import re
from os import path as os_path, mkdir, walk
from tqdm import tqdm

from logging import getLogger, INFO, ERROR
from time import time

from .VllmClient import VllmClient
from .util import make_system_prompt, make_annotation_prompt, read_elements, dict_to_string, string_to_dict, dict_to_json_file, get_xml, dict_to_json_file
from .document import Paragraph, Cell, Table
from .document.ZipFile import ZipFile


logger = getLogger(__name__)
logger.setLevel(INFO)


TABLE_TITLE_PATTERN = re.compile(r'(?:таблица|форма|приложение)\s+([^ ]+)')
EXCEPTIONAL_TABLE_LABELS = frozenset({'библиография'})


def generate_batches(items: list[int], n: int):
    if n is None or n < 1 or n > len(items):
        return [items]
    return [items[i: i + n] for i in range(0, len(items), n)]


class Annotator:
    def __init__(self, llms: list[VllmClient]):
        self.llms = llms

    def annotate(self, input_path: str, output_path: str, batch_size: int = None, n_batches: int = None, table_label_search_window: int = 10, dry_run: bool = False):
        if not os_path.isdir(output_path):
            mkdir(output_path)

        llms = self.llms

        for root, _, files in walk(input_path):
            for file in files:
                if not file.endswith('.docx'):
                    continue

                tables = []
                paragraphs = []

                annotations = {}
                offset = 0

                table_label_candidates = []

                elements = read_elements(os_path.join(root, file))
                output_file = file[:-5] + '.json'

                for element in elements:
                    if element.tag.endswith('}p'):
                        paragraph = Paragraph.from_xml(element)

                        if paragraph:
                            paragraphs.append(paragraph)
                            table_label_candidates.append(paragraph.text)
                    else:
                        i = 0
                        label = None

                        table = Table.from_xml(element)
                        seen_candidates = []

                        if table.n_cols < 2 or table.n_rows < 2:
                            continue

                        while i < table_label_search_window:
                            # try:
                            if len(table_label_candidates) > 0:
                                title = table_label_candidates.pop().lower()
                            else:
                                break
                            # except IndexError:
                            #     logger.error(
                            #         'Can\'t pop from an empty list (table size = %d x %d, total paragraphs count = %d, total tables count = %d, file = %s)',
                            #         table.n_rows,
                            #         table.n_cols,
                            #         len(paragraphs),
                            #         len(tables),
                            #         file
                            #     )
                            #     break

                            seen_candidates.append(title)

                            i += 1

                            match = TABLE_TITLE_PATTERN.match(title)

                            if match is None:
                                if title in EXCEPTIONAL_TABLE_LABELS:
                                    label = title
                            else:
                                label = match.group(1)
                                break

                        if label is None:  # Try to find the table header in the table itself
                            for cell in table.cells:
                                title = cell.lower()
                                match = TABLE_TITLE_PATTERN.match(title)

                                if match is not None:
                                    label = match.group(1)
                                    break

                        if label is None:
                            logger.error(
                                '%s - Can\'t find table label (table size = %d x %d, total paragraphs count = %d, total tables count = %d) among candidates: %s',
                                file,
                                table.n_rows,
                                table.n_cols,
                                len(paragraphs),
                                len(tables),
                                ", ".join(f'"{candidate}"' for candidate in seen_candidates)
                            )
                        else:
                            annotations[label] = {
                                'type': 'table',
                                'paragraphs': []
                            }
                            table.label = label
                            logger.info('%s - Found table %s', file, label)

                        table_label_candidates = []

                        tables.append(table)

                logger.info('%s - Found %d tables and %d paragraphs', file, len(tables), len(paragraphs))

                if dry_run:
                    continue

                batched_paragraphs = generate_batches(paragraphs, batch_size)

                if n_batches is not None:
                    batched_paragraphs = batched_paragraphs[:n_batches]

                for table in tables:
                    for llm in llms:
                        llm.reset()

                    if table.label is None or table.label not in annotations:
                        continue

                    start = time()

                    prompt = make_annotation_prompt(
                        table = Cell.serialize_rows(
                            table.rows,
                            with_embeddings = False
                        )
                    )

                    for llm in llms:
                        completion = llm.complete(prompt)

                    pbar = tqdm(batched_paragraphs, desc = f'Annotating table {table.label}')

                    for paragraphs_batch in pbar:
                        iteration = 0

                        while iteration < len(llms):
                            completion = llm.complete(
                                dict_to_string(
                                    [
                                        {
                                            'id': paragraph.id,
                                            'text': paragraph.text
                                        }
                                        for paragraph in paragraphs_batch
                                    ]
                                ),
                                add_to_history = False
                            )

                            paragraph_scores = string_to_dict(completion)

                            for i, paragraph in enumerate(paragraphs_batch):
                                result = paragraph_scores.get(paragraph.id)

                                if result is None:
                                    logger.warning('Paragraph "%s" is missing relevance score', paragraph['text'])
                                else:
                                    if iteration == 0:
                                        annotations[table.label]['paragraphs'].append(
                                            {
                                                'id': paragraph.id,
                                                'text': paragraph.text,
                                                'scores': {
                                                    llms[iteration].label: result
                                                }
                                            }
                                        )
                                    else:
                                        annotations[table.label]['paragraphs'][offset + i]['scores'][llms[iteration].label] = result

                            iteration += 1

                        offset += len(batched_paragraphs)

                    logger.warning(f'Annotation completed in in {time() - start:.3f} seconds')

                dict_to_json_file(
                    annotations,
                    os_path.join(output_path, output_file)
                )
