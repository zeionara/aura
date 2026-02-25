import re
from time import time
from string import punctuation
from os import path as os_path, mkdir, walk
from logging import getLogger
from concurrent.futures import ThreadPoolExecutor
from threading import get_ident
from json.decoder import JSONDecodeError

from .LLMClient import LLMClient
from .VllmClient import VllmClient
from .util import make_annotation_prompt, read_elements, dict_to_string, string_to_dict, dict_to_json_file, normalize_spaces
from .document import Paragraph, Cell, Table
from .Annotations import Annotations
from .AnnotationReport import AnnotationReport


logger = getLogger(__name__)


TABLE_TITLE_PATTERN = re.compile(r'(?:таблица|форма|приложение)\s+([^ ]+)')
EXCEPTIONAL_TABLE_LABELS = frozenset({'библиография'})
FORBIDDEN_TABLE_LABELS = frozenset({'технического', 'справкио'})

PUNCTUATION_TRANSLATOR = str.maketrans('', '', punctuation)


def get_previous_table_annotations(table: str, previous_annotations: dict = None):
    if previous_annotations is None or previous_annotations.get(table) is None:
        return None

    return previous_annotations[table].get('paragraphs')


def set_table_annotations(table: str, annotations: dict = None, table_annotations: list = None):
    if annotations is None or table_annotations is None:
        return None

    annotations[table]['paragraphs'] = table_annotations


def _generate_batches(items: list, batch_size: int):
    if len(items) < 1:
        return []

    if batch_size is None or batch_size < 1 or batch_size > len(items):
        return [items]

    return [items[i: i + batch_size] for i in range(0, len(items), batch_size)]


def generate_batches(items: list[Paragraph], batch_size: int, llms: list[LLMClient], n_batches: int = None, previous_annotations: dict = None):
    if previous_annotations is None:
        llm_to_items = {llm.label: items for llm in llms}
    else:
        llm_to_items = {llm.label: [] for llm in llms}

        for item in items:
            for previous_annotation in previous_annotations:
                if previous_annotation['id'] == item.id:
                    for llm in llms:
                        if previous_annotation['scores'].get(llm.label) is None:
                            llm_to_items[llm.label].append(item)
                    break
            else:
                for llm in llms:
                    llm_to_items[llm.label].append(item)

    return {
        llm: _generate_batches(items, batch_size) if n_batches is None else _generate_batches(items, batch_size)[:n_batches]
        for llm, items in llm_to_items.items()
    }


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


def merge_annotations(llm: str, annotations: list, previous_annotations: list = None, paragraph_ids: list[str] = None):
    merged_annotations = []

    if previous_annotations is None:
        for annotation in annotations:
            merged_annotations.append({
                'id': annotation['id'],
                'text': annotation['text'],
                'scores': {
                    llm: annotation['score']
                }
            })
    else:
        for paragraph_id in paragraph_ids:
            paragraph_annotation = None

            for previous_annotation in previous_annotations:
                if previous_annotation['id'] == paragraph_id:
                    paragraph_annotation = previous_annotation

                    break

            for annotation in annotations:
                if annotation['id'] == paragraph_id:
                    if paragraph_annotation is None:
                        paragraph_annotation = {
                            'id': paragraph_id,
                            'text': annotation['text'],
                            'scores': {
                                llm: annotation['score']
                            }
                        }
                    else:
                        if llm in paragraph_annotation['scores']:
                            logger.error('Overwriting LLM %s annotation results', llm)

                        paragraph_annotation['scores'][llm] = annotation['score']

                    break

            if paragraph_annotation is not None:
                merged_annotations.append(paragraph_annotation)

    return merged_annotations


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
                            'score': default
                        }
                    )
            continue

        logger.debug('%s - Table %s - Annotating %d paragraphs, attempt %d', file, table, len(ids_to_annotate), attempt)
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

        logger.debug('%s - Table %s - Paragraph scores: "%s"', file, table, str(paragraph_scores))

        for paragraph in paragraphs:
            if paragraph.id not in ids_to_annotate:
                continue

            result = paragraph_scores.get(paragraph.id)

            if result is None:
                logger.warning('%s - Table %s - Paragraph "%s" is missing relevance score', file, table, paragraph.text)
            else:
                ids_to_annotate.remove(paragraph.id)

                paragraph_id_to_annotation[paragraph.id] = (
                    {
                        'id': paragraph.id,
                        'text': paragraph.text,
                        'score': result
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


def handle_file(args):
    llm_configs, input_root, input_filename, output_path, batch_size, n_batches, table_label_search_window, ckpt_period, concurrent = args

    if concurrent:
        logger.info('Handling file %s in thread %d', input_filename, get_ident())

    if not input_filename.endswith('.docx'):
        return

    llms = None if llm_configs is None else [VllmClient(**config) for config in llm_configs]

    start_file = time()

    tables = []
    paragraphs = []

    annotations = {}
    previous_annotations = None

    form_count = 0
    missing_label_count = 0

    table_label_candidates = []
    seen_labels = set()

    elements = read_elements(os_path.join(input_root, input_filename))

    output_file = input_filename[:-5] + '.json'
    output_filename = os_path.join(output_path, output_file)

    file = f'{input_filename}'

    previous_paragraph_ids = None
    n_table_elements = 0

    if os_path.isfile(output_filename):
        try:
            previous_annotations = Annotations.from_file(output_filename)
        except JSONDecodeError:
            logger.error('Error handling file %s', output_filename)
            raise

        logger.debug('Found %d tables in the previous annotation results', previous_annotations.n_tables)

        if previous_annotations.n_tables > 0:
            previous_paragraph_ids = list(previous_annotations.paragraph_ids)

    if previous_paragraph_ids is None:
        previous_paragraph_ids = []

    for element in elements:
        if element.tag.endswith('}p'):
            paragraph = Paragraph.from_xml(
                element,
                id_ = None if previous_paragraph_ids is None or len(paragraphs) >= len(previous_paragraph_ids) else previous_paragraph_ids[len(paragraphs)]
            )

            if paragraph:
                paragraphs.append(paragraph)
                table_label_candidates.append(paragraph.text)

                if len(paragraphs) > len(previous_paragraph_ids):
                    previous_paragraph_ids.append(paragraph.id)
        else:
            n_table_elements += 1

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
                'paragraphs': [] if previous_annotations is None else previous_annotations.table_to_paragraphs.get(label, {'paragraphs': []})['paragraphs']
            }
            table.label = label
            logger.debug('%s - Found table %s - %d x %d', file, label, table.n_rows, table.n_cols)

            table_label_candidates = []

            tables.append(table)

    complete = None

    logger.info(
        '%s - Found %d tables (%d table elements) and %d paragraphs%s',
        file,
        n_tables := len(tables),
        n_table_elements,
        n_paragraphs := len(paragraphs),
        '' if llm_configs is not None else '. Annotation is complete 🟢' if (
            complete := (
                False if previous_annotations is None else previous_annotations.is_complete(n_tables, n_paragraphs)
            )
        ) else '. Annotation is incomplete 🔴'
    )

    report = AnnotationReport(file, n_tables, n_paragraphs, complete)

    tables_counter = 0

    if llm_configs is None:
        return report

    for table in tables:
        tables_counter += 1

        if table.label is None or table.label not in annotations:
            logger.warning('Missing table %s in annotations list', table.label)
            continue

        start = time()

        llm_to_batched_paragraphs = generate_batches(
            paragraphs,
            batch_size,
            llms,
            n_batches,
            previous_table_annotations := get_previous_table_annotations(table.label, None if previous_annotations is None else previous_annotations.table_to_paragraphs)
        )

        table_label = f'{table.label} [{tables_counter} / {n_tables}]'

        logger.debug('%s - Table %s - Annotation started', file, table_label)

        for llm in llms:
            if len(llm_to_batched_paragraphs[llm.label]) > 0:
                squeezed = False
                n_attempts = 2

                while n_attempts > 0:
                    llm.reset()

                    prompt = make_annotation_prompt(
                        table = Cell.serialize_rows(
                            table.rows,
                            with_embeddings = False
                        ),
                        squeeze_rows = squeezed,
                        squeeze_cols = squeezed
                    )

                    try:
                        completion = llm.complete(prompt)  # Initialize table annotation by describing the table and giving a set of instructions
                    except ValueError as e:
                        if squeezed:
                            raise ValueError(f'Failed to process squeezed table {table_label}') from e

                        logger.warning('%s - Failed the first attempt of parsing table %s', file, table_label)

                        llm.reset()

                        prompt = make_annotation_prompt(
                            table = Cell.serialize_rows(
                                table.rows,
                                with_embeddings = False
                            ),
                            squeeze_rows = True,
                            squeeze_cols = True
                        )
                        squeezed = True

                        try:
                            completion = llm.complete(prompt)  # Initialize table annotation by describing the table and giving a set of instructions
                        except ValueError:
                            logger.error('%s - Too many tokens in table %s', file, table_label)
                            raise

                    logger.debug('%s - Table %s - LLM "%s" response to the initialization message: "%s"', file, table_label, llm.label, completion)

                    batched_paragraphs = llm_to_batched_paragraphs[llm.label]
                    table_llm_annotations = []

                    n_generated_batches = len(batched_paragraphs)
                    batch_count = 0

                    paragraph_annotation_exception = None

                    for paragraphs_batch in batched_paragraphs:
                        batch_count += 1
                        logger.info('%s - Table %s - LLM %s - batch %d / %d', file, table_label, llm.label, batch_count, n_generated_batches)

                        try:
                            table_llm_annotations_batch = annotate_paragraphs(
                                llm,
                                paragraphs_batch,
                                file,
                                table_label
                            )
                        except ValueError as e:
                            logger.warning('%s - Table %s - Failed to annotate paragraphs', file, table_label)
                            paragraph_annotation_exception = e
                            break

                        table_llm_annotations.extend(table_llm_annotations_batch)

                        if ckpt_period is not None and batch_count % ckpt_period < 1:
                            merged_table_annotations = merge_annotations(llm.label, table_llm_annotations, previous_table_annotations, previous_paragraph_ids)
                            set_table_annotations(table.label, annotations, merged_table_annotations)
                            previous_table_annotations = merged_table_annotations

                            table_llm_annotations = []
                            dict_to_json_file(annotations, output_filename)
                            logger.info('%s - Table %s - LLM %s - batch %d / %d CHECKPOINT Saved to file %s', file, table_label, llm.label, batch_count, n_generated_batches, output_filename)

                    if paragraph_annotation_exception is not None:
                        if squeezed:
                            raise ValueError(f'Failed to process squeezed table {table_label}') from paragraph_annotation_exception

                        squeezed = True
                        n_attempts -= 1
                        continue

                    if len(table_llm_annotations) > 0:
                        merged_table_annotations = merge_annotations(llm.label, table_llm_annotations, previous_table_annotations, previous_paragraph_ids)
                        set_table_annotations(table.label, annotations, merged_table_annotations)
                        previous_table_annotations = merged_table_annotations

                    break
                else:
                    logger.debug('%s - Table %s - LLM "%s" Skip initialization because there are no data to handle', file, table_label, llm.label)
                    set_table_annotations(table.label, annotations, previous_table_annotations)

        logger.info('%s - Table %s - Annotation completed in %.3f seconds', file, table_label, time() - start)

    dict_to_json_file(annotations, output_filename)

    logger.info('%s - Annotation completed in %.3f seconds, results saved as "%s"', file, time() - start_file, output_filename)

    return report


class Annotator:
    def __init__(self, llm_configs: list[dict], n_workers: int = 16):
        self.llm_configs = llm_configs
        self.workers = ThreadPoolExecutor(max_workers = n_workers)

    def annotate(
        self, input_path: str, output_path: str,
        batch_size: int = None, n_batches: int = None,
        table_label_search_window: int = 20, ckpt_period: int = None,
        concurrent: bool = True
    ):
        if not os_path.isdir(output_path):
            mkdir(output_path)

        iterables = []

        for root, _, files in walk(input_path):
            for file in files:
                iterables.append(
                    (
                        self.llm_configs,
                        root,
                        file,
                        output_path,
                        batch_size,
                        n_batches,
                        table_label_search_window,
                        ckpt_period,
                        concurrent
                    )
                )

        reports = []

        if concurrent is False or self.llm_configs is None:
            for iterable in iterables:
                reports.append(handle_file(iterable))
        else:
            self.workers.map(handle_file, iterables)

        if len(reports) > 0:
            logger.info('Total n tables: %d', sum(report.n_tables for report in reports))
            logger.info('Total n paragraphs: %d', sum(report.n_paragraphs for report in reports))
            logger.info('Total n documents with tables: %d / %d', sum(report.n_tables > 0 for report in reports), n_documents := len(reports))
            logger.info('Total n incomplete documents: %d / %d', sum(not report.complete for report in reports), n_documents)
