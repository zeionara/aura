import re
from time import time
from string import punctuation
from os import path as os_path, mkdir, walk
from logging import getLogger, INFO, ERROR, WARNING, DEBUG

from .LLMClient import LLMClient
from .util import make_annotation_prompt, read_elements, dict_to_string, string_to_dict, dict_to_json_file, normalize_spaces, dict_from_json_file
from .document import Paragraph, Cell, Table


logger = getLogger(__name__)
logger.setLevel(DEBUG)


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


def is_already_annotated(file: str, table: str, paragraph_id: str, llm: str, annotations: dict = None):
    if annotations is None:
        return False

    if (table_annotations := annotations.get(table)) is None:
        return False

    if (paragraphs := table_annotations.get('paragraphs')) is None:
        return False

    for paragraph in paragraphs:
        if paragraph.get('id') == paragraph_id:
            if (scores := paragraph.get('scores')) is None:
                return False

            if scores.get(llm) is not None:
                logger.debug('%s - %s - Paragraph "%s" is already annotated by "%s"', file, table, paragraph.get('text'), llm)
                return True

    return False


def get_paragraph_offset(file: str, table: str, paragraphs: list[dict], paragraph: dict):
    paragraph_id = paragraph.get('id')
    offset = 0

    if paragraph_id is None:
        logger.error('%s - %s - Paragraph "%s" is missing id', file, table, paragraph.get('text'))

    for paragraph_ in paragraphs:
        if paragraph_.get('id') == paragraph_id:
            return offset
        offset += 1

    return None


def merge_annotations(file: str, paragraph_ids: list[str], lhs: dict, rhs: dict = None):
    if rhs is None:
        return lhs

    merged = {}

    for lhs_table_label, lhs_table_annotations in lhs.items():
        rhs_table_annotations = rhs.get(lhs_table_label)

        if rhs_table_annotations is None:
            merged[lhs_table_label] = lhs_table_annotations
            continue

        merged[lhs_table_label] = {
            'type': 'table',
            'paragraphs': []
        }

        for paragraph_id in paragraph_ids:
            lhs_paragraph = None
            rhs_paragraph = None

            for lhs_paragraph_candidate in lhs_table_annotations['paragraphs']:
                lhs_paragraph_id = lhs_paragraph_candidate.get('id')

                if lhs_paragraph_id == paragraph_id:
                    lhs_paragraph = lhs_paragraph_candidate

            for rhs_paragraph_candidate in rhs_table_annotations['paragraphs']:
                rhs_paragraph_id = rhs_paragraph_candidate.get('id')

                if rhs_paragraph_id == paragraph_id:
                    rhs_paragraph = rhs_paragraph_candidate

            if lhs_paragraph is None and rhs_paragraph is None:
                continue

            merged_paragraph = {
                'id': paragraph_id,
                'text': (
                    rhs_paragraph if lhs_paragraph is None else lhs_paragraph
                ).get('text'),
                'scores': {}
            }

            if lhs_paragraph is not None:
                for llm, score in lhs_paragraph['scores'].items():
                    merged_paragraph['scores'][llm] = score

            if rhs_paragraph is not None:
                for llm, score in rhs_paragraph['scores'].items():
                    if llm in merged_paragraph['scores']:
                        logger.error('%s - %s - Collided paragraph annotations by %s. Dismissing rhs', file, lhs_table_label, llm)
                    else:
                        merged_paragraph['scores'][llm] = score

            merged[lhs_table_label]['paragraphs'].append(merged_paragraph)

        # for lhs_paragraph in lhs_table_annotations['paragraphs']:
        #     lhs_paragraph_id = lhs_paragraph.get('id')

        #     for rhs_paragraph in rhs_table_annotations['paragraphs']:
        #         rhs_paragraph_id = rhs_paragraph.get('id')

        #         if lhs_paragraph_id == rhs_paragraph_id:
        #             merged_paragraph = {
        #                 'id': lhs_paragraph_id,
        #                 'text': lhs_paragraph.get('text'),
        #                 'scores': {}
        #             }

        #             for llm, score in lhs_paragraph['scores'].items():
        #                 merged_paragraph['scores'][llm] = score

        #             for llm, score in rhs_paragraph['scores'].items():
        #                 if llm in merged_paragraph['scores']:
        #                     logger.error('%s - %s - Collided paragraph annotations by %s. Dismissing rhs', file, lhs_table_label, llm)
        #                 else:
        #                     merged_paragraph['scores'][llm] = score

        #             merged[lhs_table_label]['paragraphs'].append(merged_paragraph)

    return merged


def annotate_paragraphs(llm: LLMClient, paragraphs: list[Paragraph], file: str, table: str, max_attempts: int = 3, default = None):
    if len(paragraphs) < 1:
        return []

    ids_to_annotate = set(paragraph.id for paragraph in paragraphs)
    paragraph_id_to_annotation = {}

    attempt = 1

    while ids_to_annotate:
        if attempt > max_attempts:
            for paragraph in paragraphs:
                if paragraph.id in ids_to_annotate:
                    logger.error('%s - Table %s - Failed to annotate paragraph "%s" after %d atempts, setting score to %s', file, table, paragraph.text, max_attempts, default)
                    ids_to_annotate.remove(paragraph.id)
                    paragraph_id_to_annotation[paragraph.id] = (
                        {
                            'id': paragraph.id,
                            'text': paragraph.text,
                            'scores': {
                                llm.label: default
                            }
                        }
                    )
            continue

        logger.info('%s - Table %s - Annotating %d paragraphs, attempt %d', file, table, len(ids_to_annotate), attempt)
        attempt += 1

        completion = llm.complete(
            dict_to_string(
                [
                    {
                        'id': paragraph.id,
                        'text': paragraph.text
                    }
                    for paragraph in paragraphs
                    if paragraph.id in ids_to_annotate
                ]
            ),
            add_to_history = False
        )
        logger.debug('%s - Table %s - LLM "%s" response to the list of paragraphs: "%s"', file, table, llm.label, completion)

        paragraph_scores = string_to_dict(completion)

        for paragraph in paragraphs:
            result = paragraph_scores.get(paragraph.id)

            if result is None:
                logger.warning('%s - Table %s - Paragraph "%s" is missing relevance score', file, table, paragraph.text)
            else:
                ids_to_annotate.remove(paragraph.id)

                paragraph_id_to_annotation[paragraph.id] = (
                    {
                        'id': paragraph.id,
                        'text': paragraph.text,
                        'scores': {
                            llm.label: result
                        }
                    }
                )

    return [
        paragraph_id_to_annotation[paragraph.id]
        for paragraph in paragraphs
    ]


def make_table_header(table_label_candidates: list[str]):
    label_length = 0
    label = None

    for candidate in table_label_candidates:
        parts = normalize_spaces(candidate).split(' ')

        if len(parts) > 4 and label_length < 4:
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
    def __init__(self, llms: list[LLMClient]):
        self.llms = llms

    def annotate(self, input_path: str, output_path: str, batch_size: int = None, n_batches: int = None, table_label_search_window: int = 20, dry_run: bool = False, n_files: int = None):
        if not os_path.isdir(output_path):
            mkdir(output_path)

        llms = self.llms

        n_tables = 0
        n_paragraphs = 0

        for root, _, files in walk(input_path):
            for file in files:
                if not file.endswith('.docx'):
                    continue

                if n_files is not None:
                    if n_files < 1:
                        break

                    n_files -= 1

                start_file = time()

                tables = []
                paragraphs = []

                annotations = {}
                previous_annotations = None

                form_count = 0
                missing_label_count = 0

                table_label_candidates = []
                seen_labels = set()

                elements = read_elements(os_path.join(root, file))

                output_file = file[:-5] + '.json'
                output_filename = os_path.join(output_path, output_file)

                previous_paragraph_ids = None

                if os_path.isfile(output_filename):
                    # logger.info('%s - File "%s" already exists. Moving forward', file, output_filename)
                    previous_annotations = dict_from_json_file(output_filename)

                    previous_annotations_values = list(previous_annotations.values())  # synchronize existing paragraph ids and new paragraph ids
                    if len(previous_annotations_values) > 0:
                        previous_paragraph_ids = [
                            paragraph['id']
                            for paragraph in previous_annotations_values[0]['paragraphs']
                        ]

                for element in elements:
                    if element.tag.endswith('}p'):
                        paragraph = Paragraph.from_xml(
                            element,
                            id_ = None if previous_paragraph_ids is None or len(paragraphs) >= len(previous_paragraph_ids) else previous_paragraph_ids[len(paragraphs)]
                        )

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
                        logger.info('%s - Found table %s - %d x %d', file, label, table.n_rows, table.n_cols)
                        logger.debug('%s - Table %s - %d x %d Content: %s', file, label, table.n_rows, table.n_cols, ', '.join(f'"{cell}"' for cell in table.cells))

                        table_label_candidates = []

                        tables.append(table)

                        n_tables += 1

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
                        completion = llm.complete(prompt)  # Initialize table annotation by describing the table and giving a set of instructions
                        logger.debug('%s - %s - LLM "%s" response to the initialization message: "%s"', file, table.label, llm.label, completion)

                    n_batches = len(batched_paragraphs)
                    batch_count = 0

                    for paragraphs_batch in batched_paragraphs:
                        iteration = 0

                        batch_count += 1
                        logger.info('%s - Table %s - batch %d / %d', file, table.label, batch_count, n_batches)

                        while iteration < len(llms):
                            annotated_batched_paragraphs = annotate_paragraphs(
                                llms[iteration],
                                (
                                    paragraphs_batch
                                    if previous_annotations is None else
                                    [
                                        paragraph
                                        for paragraph in paragraphs_batch
                                        if not is_already_annotated(file, table.label, paragraph.id, llms[iteration].label, previous_annotations)
                                    ]
                                ),
                                file,
                                table.label
                            )

                            for paragraph in annotated_batched_paragraphs:
                                paragraph_offset = get_paragraph_offset(file, table.label, annotations[table.label]['paragraphs'], paragraph)

                                if paragraph_offset is None:
                                    annotations[table.label]['paragraphs'].append(
                                        paragraph
                                    )
                                else:
                                    annotations[table.label]['paragraphs'][paragraph_offset]['scores'][llms[iteration].label] = annotated_batched_paragraphs[i]['scores'][llms[iteration].label]

                            # if iteration == 0:  # paragraphs from this batch have no annotations yet
                            #     annotations[table.label]['paragraphs'].extend(
                            #         annotated_batched_paragraphs
                            #     )
                            # else:
                            #     for i in enumerate(annotated_batched_paragraphs):
                            #         annotations[table.label]['paragraphs'][offset + i]['scores'][llms[iteration].label] = annotated_batched_paragraphs[i]['scores'][llms[iteration].label]

                            iteration += 1

                    logger.info('%s - Table %s - Annotation completed in %.3f seconds', file, table.label, time() - start)

                merged_annotations = merge_annotations(file, [paragraph.id for paragraph in paragraphs], annotations, previous_annotations)

                dict_to_json_file(
                    merged_annotations,
                    output_filename
                )

                logger.info('%s - Annotation completed in %.3f seconds, results saved as "%s"', file, time() - start_file, output_filename)

        logger.info('Found %d tables and %d paragraphs', n_tables, n_paragraphs)
