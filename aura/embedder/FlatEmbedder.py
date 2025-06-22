from transformers import AutoTokenizer, AutoModel

from torch import Tensor
import torch.nn.functional as F

from .BaseModel import BaseModel
from .util import to_cuda


def average_pool(  # compute average token embeddings for each input document
    last_hidden_states: Tensor,
    attention_mask: Tensor
) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class FlatEmbedder:

    def __init__(self, model: BaseModel, cuda: bool = True):
        self.tokenizer = AutoTokenizer.from_pretrained(model.value)

        # for token, index in self.tokenizer.vocab.items():
        #     if index < 10:
        #         print(token, index)

        # dd

        self.model = AutoModel.from_pretrained(model.value)

        self.cuda = cuda

        if cuda:
            self.model.to('cuda')

    def embed(self, items: list[dict]):
        input_texts = [
            item['text'] for item in items
            if item['type'] == 'paragraph'
        ]

        batch_dict = self.tokenizer(input_texts, max_length = 512, padding = True, truncation = True, return_tensors = 'pt')

        if self.cuda:
            batch_dict = to_cuda(batch_dict)

        outputs = self.model(**batch_dict)

        text_embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        text_embeddings = F.normalize(text_embeddings, p = 2, dim = 1)

        print(text_embeddings)
