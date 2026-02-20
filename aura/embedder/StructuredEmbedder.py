import re
from logging import getLogger
from json import dumps

from transformers import AutoTokenizer, AutoModel
from torch import tensor as torch_tensor, stack as torch_stack, device, Tensor, save, optim, load
import torch.nn.functional as F

from .AttentionTableEmbedder import AttentionTableEmbedder, DEFAULT_INPUT_DIM
from .FlatEmbedder import embedding_to_list
from .BaseModel import BaseModel
from .util import to_cuda, drop_embeddings
from ..Subset import Subset
from .TableTracker import TableTracker


logger = getLogger(__name__)


class StructuredEmbedder:
    def __init__(self, model: BaseModel, cuda: bool = True, input_dim: int = DEFAULT_INPUT_DIM, path: str = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model.value)
        self.model = AttentionTableEmbedder(input_dim = input_dim)

        if path is not None:
            print(f'Loading model from {path}...')

            state_dict = load(path)
            self.model.load_state_dict(state_dict)

        self.base_model = model.value
        self.cuda = cuda
        self.table_tracker = TableTracker()

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

    def tokenize_table(self, table: dict, max_length: int, batch_size: int):  # TODO: Implement support for batch_size and max_length
        id_to_embedding = {}
        id_to_cols = {}

        is_first_row = True

        row_embeddings = []
        col_embeddings = []

        document = table.get('document', 'unknown')
        label = table.get('label')
        broken = False

        rows = table['rows']
        n_rows = len(rows)
        i = 0

        while i < n_rows:
            row = rows[i]

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

                    if not self.has_flat_embedding(cell):
                        raise ValueError(f'Please, apply flat embedder first for model {self.base_model}')

                    if cell_id not in id_to_embedding:
                        id_to_embedding[cell_id] = cell_embedding = cell['embeddings']['flat'][self.base_model]
                    id_to_cols[cell_id] = cell_cols = cell['cols']

                    cell_embeddings.append(cell_embedding)

                    if is_first_row:
                        for _ in range(cell_cols):
                            col_embeddings.append([cell_embedding])
                    else:
                        break_outer_loop = False

                        for _ in range(cell_cols):
                            try:
                                col_embeddings[j].append(cell_embedding)
                            except IndexError:
                                if not broken:
                                    broken = True

                                if row[0].get('skipped'):
                                    break_outer_loop = True
                                    i += 1
                                    break

                                logger.warning(
                                    '%s - Table %s - Can\'t insert col embedding in row %d. Trying to replicate the structure of the last valid row...',
                                    document,
                                    label,
                                    i,
                                )

                                if i > 0 and len(rows[i - 1]) == len(row) and not row[0].get('updated'):
                                    for k in range(len(row)):
                                        row[k]['cols'] = rows[i - 1][k].get('cols', 1)

                                    row[0]['updated'] = True

                                    break_outer_loop = True
                                    break

                                if len(row) == 1 and not row[0].get('updated'):
                                    row[0]['cols'] = sum(c['cols'] for c in rows[i - 1])

                                    row[0]['updated'] = True

                                    break_outer_loop = True
                                    break

                                if len(row) > 1 and len(rows[i - 1]) == 1 and not row[0].get('updated'):
                                    row[-1]['cols'] = rows[i - 1][0]['cols'] - sum(c['cols'] for c in row)

                                    row[0]['updated'] = True

                                    break_outer_loop = True
                                    break

                                for k in range(len(col_embeddings)):
                                    col_embeddings[k] = col_embeddings[k][:-1]  # delete col embeddings for the overflowing row

                                logger.warning(
                                    '%s - Table %s - Number of cells in row %d and row %d are different (%d != %d). Skipping this row.',
                                    document,
                                    label,
                                    i - 1,
                                    i,
                                    len(rows[i - 1]),
                                    len(row)
                                )

                                row[0]['skipped'] = True
                                i += 1

                                break_outer_loop = True
                                break

                            j += 1

                        if break_outer_loop:
                            break
            else:  # if row is not overflowing
                row_tensor = torch_tensor(cell_embeddings, device = self.device)
                row_embeddings.append(row_tensor)  # tensor with row-wise cell embeddings
                i += 1

            if is_first_row:
                is_first_row = False

        col_embeddings = [torch_tensor(cell_embeddings, device = self.device) for cell_embeddings in col_embeddings]

        for col_tensor in col_embeddings:
            if any([item < 1 for item in col_tensor.shape]):
                # logger.error('%s - Table %s contains incorrect number of columns', table.get('document', 'unknown'), table['label'])
                raise ValueError(f'Table {document}.{label} %s contains incorrect number of columns')
                # return None

        self.table_tracker.register_table(
            document,
            label,
            n_rows=len(rows),
            broken=broken,
            n_rows_skipped=len([row for row in rows if row[0].get('skipped')]) if broken else None,
            n_rows_updated=len([row for row in rows if row[0].get('updated')]) if broken else None
        )

        return row_embeddings, col_embeddings

    def set_element_embedding(self, element: dict, embedding: Tensor):
        if 'embeddings' not in element:
            print(element)
        if (structured_embeddings := element['embeddings'].get('structured')) is None:
            element['embeddings']['structured'] = {self.base_model: embedding_to_list(embedding)}
        else:
            structured_embeddings[self.base_model] = embedding_to_list(embedding)

    def embed(self, elements: list[dict], batch_size: int = 8, max_length: int = 512):
        for element in elements:
            if element['type'] == 'paragraph':
                if (flat_embeddings := element['embeddings'].get('flat')) is not None and flat_embeddings.get(self.base_model) is not None:
                    continue
                else:
                    print(element)
                    raise ValueError(f'Please, apply flat embedder first for model {self.base_model}')
            elif element['type'] == 'table':
                inputs = self.tokenize_table(element, batch_size = batch_size, max_length = max_length)

                if inputs is None:
                    continue

                embedding = self.model(
                    *inputs
                )
                self.set_element_embedding(element, embedding)

    def train(self, elements: list[dict], optimizer: optim.Adam, batch_size: int = 8, max_length: int = 512, epochs: int = 20, table_tracker_export_path: str = 'assets/broken-tables.yml'):
        paragraph_id_to_embedding = {}

        for element in elements:
            if element['type'] == 'paragraph' and (subset := element['subset']) is not None and Subset(subset) == Subset.TRAIN:
                paragraph_id_to_embedding[element['id']] = torch_tensor(element['embeddings']['flat'][self.base_model], device = self.device)

        self.model.train()

        for i in range(epochs):
            total_loss = 0
            batch_count = 0

            for element in elements:
                if element['type'] == 'paragraph':
                    if (flat_embeddings := element['embeddings'].get('flat')) is not None and flat_embeddings.get(self.base_model) is not None:
                        continue
                    raise ValueError(f'Please, apply flat embedder first for model {self.base_model}')
                elif element['type'] == 'table':
                    if 'context' not in element:
                        continue

                    inputs = self.tokenize_table(element, batch_size = batch_size, max_length = max_length)  # row-wise and column-wise cell embeddings

                    if inputs is None:
                        continue  # skip such table

                    output = self.model(*inputs)  # table embedding

                    batch_loss = 0
                    positive_count = 0

                    for paragraph in element['context'].keys():
                        if (paragraph_embedding := paragraph_id_to_embedding.get(paragraph)) is not None:
                            cosine_sim = F.cosine_similarity(output, paragraph_embedding, dim = 0)
                            distance = 1 - ((cosine_sim + 1) / 2)

                            batch_loss += distance
                            positive_count += 1

                    if positive_count > 0:
                        avg_batch_loss = batch_loss / positive_count

                        optimizer.zero_grad()
                        avg_batch_loss.backward()
                        optimizer.step()

                        total_loss += avg_batch_loss.item()
                        batch_count += 1

                        # print(f'Batch loss: {avg_batch_loss.item():.4f}')

            if i == 0:
                n_broken, mean_correctness_ratio = self.table_tracker.count_broken()
                n_total = self.table_tracker.count()

                self.table_tracker.export(table_tracker_export_path)

                print(f'Broken: {n_broken} / {n_total} ({n_broken / n_total * 100:.3f}%)')
                print(f'Mean correctness ratio of the broken tables = {mean_correctness_ratio:.3f}')

            if batch_count > 0:
                print(f'[{i} / {epochs}] Average epoch loss: {total_loss / batch_count:.4f}')

    def save(self, path: str):
        save(self.model.state_dict(), path)
