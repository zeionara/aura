import re

from transformers import AutoTokenizer, AutoModel
from torch import tensor as torch_tensor, stack as torch_stack, device, Tensor
import torch.nn.functional as F

from .AttentionTableEmbedder import AttentionTableEmbedder
from .FlatEmbedder import embedding_to_list
from .BaseModel import BaseModel
from .util import to_cuda


class StructuredEmbedder:
    def __init__(self, model: BaseModel, cuda: bool = True):
        self.tokenizer = AutoTokenizer.from_pretrained(model.value)
        self.model = AttentionTableEmbedder()
        self.base_model = model.value
        self.cuda = cuda

        if cuda:
            self.model.to('cuda')
            self.device = device('cuda')
        else:
            self.device = 'cpu'


    def has_flat_embedding(self, element: dict):
        if 'embeddings' not in element:
            return False

        if 'flat' not in element['embeddings']:
            return False

        return self.base_model in element['embeddings']['flat']


    def tokenize_table(self, table: dict, max_length: int, batch_size: int):
        id_to_embedding = {}
        id_to_cols = {}

        is_first_row = True

        row_embeddings = []
        col_embeddings = []

        for row in table['rows']:
            cell_embeddings = []
            j = 0

            for cell in row:
                if cell.get('text') is None:
                    if is_first_row:
                        raise ValueError('Placeholders are not allowed in the first table row')

                    cell_id = cell['id']

                    cell_embeddings.append(id_to_embedding[cell_id])
                    j += id_to_cols[cell_id]
                else:
                    cell_id = cell['id']

                    if cell_id in id_to_embedding:
                        raise ValueError(f'Duplicate cell ids are not allowed ({cell_id})')

                    if not self.has_flat_embedding(cell):
                        raise ValueError(f'Please, apply flat embedder first for model {self.base_model}')

                    id_to_embedding[cell_id] = cell_embedding = cell['embeddings']['flat'][self.base_model]
                    id_to_cols[cell_id] = cell_cols = cell['cols']

                    cell_embeddings.append(cell_embedding)

                    if is_first_row:
                        for _ in range(cell_cols):
                            col_embeddings.append([cell_embedding])
                    else:
                        for _ in range(cell_cols):
                            try:
                                col_embeddings[j].append(cell_embedding)
                            except IndexError:
                                raise

                            j += 1

            row_embeddings.append(torch_tensor(cell_embeddings, device = self.device))  # tensor with row-wise cell embeddings

            if is_first_row:
                is_first_row = False

        col_embeddings = [torch_tensor(cell_embeddings, device = self.device) for cell_embeddings in col_embeddings]

        return row_embeddings, col_embeddings

    def set_element_embedding(self, element: dict, embedding: Tensor):
        if 'embeddings' not in element:
            print(element)
        if (structured_embeddings := element['embeddings'].get('structured')) is None: 
            element['embeddings']['structured'] = {self.base_model: embedding_to_list(embedding)}
        else:
            structured_embeddings[self.base_model] = embedding_to_list(embedding)

    def embed(self, elements: list[dict], batch_size: int = 8, max_length: int = 512):  # TODO: Add batch size
        for element in elements:
            if element['type'] == 'paragraph':
                if (flat_embeddings := element['embeddings'].get('flat')) is not None and flat_embeddings.get(self.base_model) is not None:
                    continue
                else:
                    print(element)
                    raise ValueError(f'Please, apply flat embedder first for model {self.base_model}')
            elif element['type'] == 'table':
                embedding = self.model(
                    *self.tokenize_table(element, batch_size = batch_size, max_length = max_length)
                )
                self.set_element_embedding(element, embedding)

        self.model.save('/tmp/weights.pkl')
