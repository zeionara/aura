import re
from time import time
from string import punctuation
from os import path as os_path, mkdir, walk
from logging import getLogger, INFO, ERROR, WARNING

from tqdm import tqdm

from .VllmClient import VllmClient
from .util import make_annotation_prompt, read_elements, dict_to_string, string_to_dict, dict_to_json_file, normalize_spaces
from .document import Paragraph, Cell, Table


logger = getLogger(__name__)
logger.setLevel(INFO)


TABLE_TITLE_PATTERN = re.compile(r'(?:таблица|форма|приложение)\s+([^ ]+)')
EXCEPTIONAL_TABLE_LABELS = frozenset({'библиография'})
FORBIDDEN_TABLE_LABELS = frozenset({'технического', 'справкио'})

PUNCTUATION_TRANSLATOR = str.maketrans('', '', punctuation)


def generate_batches(items: list[int], n: int):
    if n is None or n < 1 or n > len(items):
        return [items]
    return [items[i: i + n] for i in range(0, len(items), n)]


def remove_punctuation(input_string):
    return input_string.translate(PUNCTUATION_TRANSLATOR)


def is_part_of_form(table_label_candidates: list[str]):
    for candidate in table_label_candidates:
        if len(candidate) < 1:
            continue

        unique_chars = set(candidate)

        if len(unique_chars) < 2 and unique_chars.pop() == '_':
            return True

    return False


def make_table_header(table_label_candidates: list[str]):
    label_length = 0
    label = None

    for candidate in table_label_candidates:
        parts = normalize_spaces(candidate).split(' ')

        if len(parts) > 4:
            label_length = 4
            label = remove_punctuation(f'{parts[1]} {parts[2]} {parts[3]} {parts[4]}')
            continue

        if len(parts) > 3 and label_length < 3:
            label_length = 3
            label = remove_punctuation(f'{parts[1]} {parts[2]} {parts[3]}')
            continue

        if len(parts) > 2 and label_length < 2:
            label_length = 2
            label = remove_punctuation(f'{parts[1]} {parts[2]}')
            continue

        if len(parts) > 1 and label_length <= 1:
            label_length = 1
            label = remove_punctuation(parts[1])
            continue

        if len(parts) > 0 and label_length < 1:
            label_length = 1
            label = remove_punctuation(parts[0])
            continue

    return label


class Annotator:
    def __init__(self, llms: list[VllmClient]):
        self.llms = llms

    def annotate(self, input_path: str, output_path: str, batch_size: int = None, n_batches: int = None, table_label_search_window: int = 20, dry_run: bool = False):
        if not os_path.isdir(output_path):
            mkdir(output_path)

        llms = self.llms

        n_tables = 0
        n_paragraphs = 0

        for root, _, files in walk(input_path):
            for file in files:
                if not file.endswith('.docx'):
                    continue

                tables = []
                paragraphs = []

                annotations = {}
                offset = 0
                form_count = 0
                missing_label_count = 0

                table_label_candidates = []
                seen_labels = set()

                elements = read_elements(os_path.join(root, file))

                output_file = file[:-5] + '.json'
                output_filename = os_path.join(output_path, output_file)

                for element in elements:
                    if element.tag.endswith('}p'):
                        paragraph = Paragraph.from_xml(element)

                        if paragraph:
                            paragraphs.append(paragraph)
                            table_label_candidates.append(paragraph.text)

                            n_paragraphs += 1
                    else:
                        if len(paragraphs) < 1:
                            continue

                        i = 0
                        label = None

                        table = Table.from_xml(element)
                        seen_candidates = []

                        if table.n_cols < 2 or table.n_rows < 2:
                            continue

                        while i < table_label_search_window:
                            if len(table_label_candidates) > 0:
                                title = table_label_candidates.pop().lower()
                            else:
                                break

                            seen_candidates.append(title)

                            i += 1

                            match = TABLE_TITLE_PATTERN.match(title)

                            if match is None:
                                if title in EXCEPTIONAL_TABLE_LABELS:
                                    label = title
                            elif match.group(1) not in FORBIDDEN_TABLE_LABELS:
                                label = match.group(1)

                                if 'форма' in title:
                                    label = f'форма {label}'

                                if 'приложение' in title:
                                    label = f'приложение {label}'

                                break

                        if label is None:  # Try to find the table header in the table itself
                            for cell in table.cells:
                                if cell is None:
                                    continue

                                title = cell.lower()
                                match = TABLE_TITLE_PATTERN.match(title)

                                if match is not None:
                                    label = match.group(1)
                                    break

                        if is_part_of_form(seen_candidates):
                            form_count += 1
                            label = f'часть формы {form_count}'

                        if label is None:  # Try to generate table name from seen candidates
                            label = make_table_header(seen_candidates)
                            logger.warning('%s - Had to generate table name "%s"', file, label)

                        if label is None:
                            missing_label_count += 1
                            label = f'таблица без названия {missing_label_count}'

                        label = label.replace('(справочное)', '')
                        label = label.replace('(рекомендуемое)', '')

                        while label in seen_labels:
                            label = f'{label}_'

                        seen_labels.add(label)

                        annotations[label] = {
                            'type': 'table',
                            'paragraphs': []
                        }
                        table.label = label
                        logger.info('%s - Found table %s', file, label)

                        table_label_candidates = []

                        tables.append(table)

                        n_tables += 1

                logger.info('%s - Found %d tables and %d paragraphs', file, len(tables), len(paragraphs))

                if dry_run or os_path.isfile(output_filename):
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
                                    logger.warning('Paragraph "%s" is missing relevance score', paragraph.text)
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

                    logger.warning('Annotation completed in in %.3f seconds', time() - start)

                dict_to_json_file(
                    annotations,
                    output_filename
                )

        logger.info('Total - Found %d tables and %d paragraphs', n_tables, n_paragraphs)
