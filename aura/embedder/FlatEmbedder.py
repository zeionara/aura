from transformers import AutoTokenizer, AutoModel

from torch import Tensor, ones_like, cat as torch_cat
import torch.nn.functional as F

from .BaseModel import BaseModel
from .util import to_cuda


def average_pool(  # compute average token embeddings for each input document
    last_hidden_states: Tensor,
    attention_mask: Tensor
) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def split_into_batches(arr, batch_size):
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    batches = []

    for i in range(0, len(arr), batch_size):
        batches.append(arr[i:i + batch_size])

    return batches


def join_table_batch_dict(batch_dict, max_length: int = 512):
    flattened = batch_dict['input_ids'].view(1, -1)
    
    truncated = flattened[:, :max_length]
    
    ones_tensor = ones_like(truncated)
    
    return {'input_ids': truncated, 'attention_mask': ones_tensor}


def extract_element_embeddings(embeddings: Tensor):
    embeddings_cpu = embeddings.detach().cpu()

    array = embeddings_cpu.numpy()

    return [array[i] for i in range(array.shape[0])]


class FlatEmbedder:

    def __init__(self, model: BaseModel, cuda: bool = True):
        self.tokenizer = AutoTokenizer.from_pretrained(model.value)

        self.pad = self.tokenizer.convert_tokens_to_ids(['<pad>'])[0]

        self.model_name = model.value
        self.model = AutoModel.from_pretrained(model.value)

        self.cuda = cuda

        if cuda:
            self.model.to('cuda')

    def join_batch_dicts(self, batch_dicts: list[dict]):
        max_cols = max(batch_dict['input_ids'].shape[1] for batch_dict in batch_dicts)

        all_padded_input_ids = []
        all_padded_attention_masks = []

        for batch_dict in batch_dicts:
            input_ids = batch_dict['input_ids']
            attention_mask = batch_dict['attention_mask']

            pad_input_ids = max_cols - input_ids.shape[1]
            padded_input_ids = F.pad(input_ids, (0, pad_input_ids), value = self.pad)

            pad_attention_mask = max_cols - attention_mask.shape[1]
            padded_attention_mask = F.pad(attention_mask, (0, pad_attention_mask), value = 0)

            all_padded_input_ids.append(padded_input_ids)
            all_padded_attention_masks.append(padded_attention_mask)

        return {
            'input_ids': torch_cat(all_padded_input_ids, dim = 0),
            'attention_mask': torch_cat(all_padded_attention_masks, dim = 0)
        }

    def embed(self, elements: list[dict], batch_size: int = 8, max_length: int = 512):
        batches = split_into_batches(elements, batch_size)

        for batch in batches:
            if all(element['type'] == 'paragraph' for element in batch):
                batch_dict = self.tokenizer([item['text'] for item in batch], max_length = max_length, padding = True, truncation = True, return_tensors = 'pt')
            else:
                batch_dicts = []

                for element in batch:
                    if element['type'] == 'paragraph':
                        batch_dict = self.tokenizer(element['text'], max_length = max_length, padding = True, truncation = True, return_tensors = 'pt')
                        batch_dicts.append(batch_dict)
                    elif element['type'] == 'table':
                        cells = [
                            cell['text']
                            for row in element['rows']
                            for cell in row
                            if cell.get('text')
                        ]
                        batch_dict = self.tokenizer(cells, max_length = max_length, padding = True, truncation = True, return_tensors = 'pt')
                        batch_dicts.append(join_table_batch_dict(batch_dict))
                    else:
                        raise ValueError(f'Unknown element type: {element["type"]}')

                batch_dict = self.join_batch_dicts(batch_dicts)

            if self.cuda:
                batch_dict = to_cuda(batch_dict)

            outputs = self.model(**batch_dict)

            text_embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            text_embeddings = F.normalize(text_embeddings, p = 2, dim = 1)

            for embedding, element in zip(extract_element_embeddings(text_embeddings), batch):
                if (flat_embeddings := element['embeddings'].get('flat')) is None: 
                    element['embeddings'] = {'flat': {self.model_name: embedding.tolist()}}
                else:
                    flat_embeddings[self.model_name] = embedding.tolist()
