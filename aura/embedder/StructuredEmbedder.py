import re

from transformers import AutoTokenizer, AutoModel
from torch import Tensor, stack as torch_stack
import torch.nn.functional as F

from .AttentionTableEmbedder import AttentionTableEmbedder
from .FlatEmbedder import average_pool, split_into_batches
from .BaseModel import BaseModel
from .util import to_cuda


class StructuredEmbedder:
    def __init__(self, model: BaseModel, cuda: bool = True):
        self.tokenizer = AutoTokenizer.from_pretrained(model.value)
        self.model = AttentionTableEmbedder(model.value, cuda = cuda)
        self.cuda = cuda

        if cuda:
            self.model.to('cuda')


    def has_flat_embedding(self, element: dict):
        if 'embeddings' not in element:
            return False

        if 'flat' not in element['embeddings']:
            return False

        return self.model.model_name in element['embeddings']['flat']


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
                        raise ValueError(f'Please, apply flat embedder first for model {self.model.model_name}')

                    id_to_embedding[cell_id] = cell_embedding = cell['embeddings']['flat'][self.model.model_name]
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

            row_embeddings.append(Tensor(cell_embeddings))  # tensor with row-wise cell embeddings

            if is_first_row:
                is_first_row = False

        col_embeddings = [Tensor(cell_embeddings) for cell_embeddings in col_embeddings]

        if table['rows'][0][0]['text'] == 'Контролируемый параметр':
            print('Row-wise embeddings:')

            for tensor in row_embeddings:
                print(tensor.shape)

            print('Column-wise embeddings:')

            for tensor in col_embeddings:
                print(tensor.shape)

        # batches = []

        # input_texts = []
        # n_levels = 0

        # for row in table['rows']:
        #     n_levels += 1

        #     for cell in row:
        #         if (text := cell.get('text')):
        #             input_texts.append(text)
        #         else:
        #             input_texts.append('')

        #         if (n_cols := cell.get('cols', 1)) > 1:
        #             for _ in range(n_cols - 1):
        #                 input_texts.append('')

        #     if n_levels >= batch_size:
        #         batch_dict = self.tokenizer(input_texts, max_length = max_length, padding = True, truncation = True, return_tensors = 'pt')

        #         if self.cuda:
        #             batch_dict = to_cuda(batch_dict)

        #         batches.append(
        #             (
        #                 batch_dict,
        #                 n_levels
        #             )
        #         )

        #         input_texts = []
        #         n_levels = 0

        # if n_levels > 0:
        #     batch_dict = self.tokenizer(input_texts, max_length = max_length, padding = True, truncation = True, return_tensors = 'pt')

        #     if self.cuda:
        #         batch_dict = to_cuda(batch_dict)

        #     batches.append(
        #         (
        #             batch_dict,
        #             n_levels
        #         )
        #     )

        # return batches
        

    def embed(self, elements: list[dict], batch_size: int = 8, max_length: int = 512):  # TODO: Add batch size
        for element in elements:
            if element['type'] == 'paragraph':
                if (flat_embeddings := element['embeddings'].get('flat')) is not None and flat_embeddings.get(self.model.model_name) is not None:
                    continue
                else:
                    print(element)
                    raise ValueError(f'Please, apply flat embedder first for model {self.model.model_name}')
            elif element['type'] == 'table':
                # batch_dict, n_levels = self.tokenize_table(element, max_length = max_length, batch_size = batch_size)

                self.tokenize_table(element, batch_size = batch_size, max_length = max_length)

                # outputs_lists = []

                # for batch_dict, n_levels in self.tokenize_table(element, max_length, batch_size):
                #     outputs = self.model(n_levels = n_levels, **batch_dict)
                #     outputs_lists.append(outputs)

                # stacked_outputs = torch_stack(outputs_lists)
                # average_outputs = stacked_outputs.sum(dim = 0) / stacked_outputs.shape[0]

                # if (structured_embeddings := element['embeddings'].get('structured')) is None: 
                #     element['embeddings']['structured'] = {self.model.model_name: average_outputs.detach().cpu().numpy().tolist()}
                # else:
                #     structured_embeddings[self.model.model_name] = average_outputs.detach().cpu().numpy().tolist()

        dd
