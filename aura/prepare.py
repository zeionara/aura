from os import walk, path as os_path, mkdir
from json import dump, load
from pathlib import Path
from queue import Queue
from random import seed as random_seed
from logging import getLogger

from zipfile import ZipFile
from lxml import etree

from .util import get_comments, get_elements, get_xml
from .document import Paragraph, Table, INDENT
from .Subset import Subset


logger = getLogger(__name__)


MODEL = 'mistral-24b'
SCALE = 1.0


def prepare(input_path: str, output_path: str, annotations_path: str, train_fraction: float, seed: int, relevance_threshold: float):
    random_seed(seed)

    if annotations_path is None:
        return prepare_source(input_path, output_path, train_fraction)

    return prepare_annotations(input_path, output_path, annotations_path, train_fraction, relevance_threshold)


def make_comment(comment_id: int, table_label: str, score_value: float, note: str = None):
    return {
        "id": comment_id,
        "body": {
            "target": table_label,
            "score": score_value * SCALE,
            "note": note
        }
    }


def make_paragraph_element(paragraph_id: str, text: str, train_fraction: float, comments: list[dict] = None):
    element = {
        'type': 'paragraph',
        'id': paragraph_id,
        'text': text,
        'style': None,
        'embeddings': {},
        'subset': None if comments is None else Subset.random(train_fraction).value
    }

    if comments is not None:
        element['comments'] = comments

    return element


def prepare_annotations(input_path: str, output_path: str, annotations_path: str, train_fraction: float, relevance_threshold: float):
    if not os_path.isdir(output_path):
        mkdir(output_path)

    n_files = 0

    for root, _, files in walk(annotations_path):
        for file in files:
            n_files += 1

    i_file = 1

    for root, _, files in walk(annotations_path):
        for file in files:

            path = os_path.join(root, file)

            output_file = os_path.join(output_path, f'{Path(file).stem}.json')
            input_file = os_path.join(input_path, f'{Path(file).stem}.docx')

            # Parse tables from docx

            with ZipFile(input_file) as docx_zip:
                content = etree.XML(
                    docx_zip.read('word/document.xml')
                )

            xml_elements = get_elements(content)

            table_elements = []
            table_contexts = []

            table_elements_empty = []

            first_element = True

            for element, _ in xml_elements:
                if not element.tag.endswith('}p') and not first_element:
                    table = Table.from_xml(element)

                    if table.n_cols < 2 or table.n_rows < 2:
                        table_elements_empty.append(table.json)
                    else:
                        table_elements.append(table.json)

                if first_element:
                    first_element = False

            # Parse paragraphs from annotations

            with open(path, 'r', encoding = 'utf-8') as file_handler:
                annotations = load(file_handler)

            paragraph_id_to_comments = {}

            n_paragraphs = None
            n_tables = 0
            comment_id = 1

            for table_label, table_content in annotations.items():
                n_tables += 1

                paragraph_id_to_score = {}

                paragraphs = table_content['paragraphs']

                n_paragraphs_with_missing_scores = 0
                n_paragraphs_for_current_table = len(paragraphs)

                for paragraph in paragraphs:
                    model_scores = paragraph['scores'][MODEL]

                    if model_scores is None or 'score' not in model_scores:
                        n_paragraphs_with_missing_scores += 1
                        continue

                    if not isinstance((score_value := model_scores['score']), float):
                        try:
                            score_value = float(model_scores['score'])
                        except ValueError:
                            logger.warning('%s - %s - incorrect score: %s', file, table_label, score_value)
                            n_paragraphs_with_missing_scores += 1
                            continue

                    if score_value > relevance_threshold:
                        paragraph_id = paragraph['id']

                        paragraph_id_to_score[paragraph_id] = score_value

                        if (comments := paragraph_id_to_comments.get(paragraph_id)):
                            comments.append(
                                make_comment(comment_id, table_label, score_value, model_scores['comment'])
                            )
                        else:
                            paragraph_id_to_comments[paragraph_id] = [
                                make_comment(comment_id, table_label, score_value, model_scores['comment'])
                            ]

                        comment_id += 1

                if n_paragraphs_with_missing_scores > 0:
                    logger.warning(
                        '%s - Table %s - %d/%d paragraphs are missing score',
                        file,
                        table_label,
                        n_paragraphs_with_missing_scores,
                        n_paragraphs_for_current_table
                    )

                table_contexts.append(
                    {
                        'label': table_label,
                        'context': paragraph_id_to_score
                    }
                )

                if n_paragraphs is None:
                    n_paragraphs = n_paragraphs_for_current_table
                    continue

                assert n_paragraphs == n_paragraphs_for_current_table, (
                    f'{file} - {table_label} '
                    'number of paragraphs in table is not equal to the number of paragraphs in other tables '
                    f'({n_paragraphs_for_current_table} != {n_paragraphs})'
                )

            if n_tables != (n_table_elements := len(table_elements)):
                logger.error(
                    '%s - number of tables in annotations file is not equal to the number of table elements in source file (%d != %d)',
                    file, n_tables, n_table_elements
                )
                continue

            elements = []

            for paragraph in paragraphs:
                paragraph_id = paragraph['id']

                elements.append(
                    make_paragraph_element(
                        paragraph['id'],
                        paragraph['text'],
                        train_fraction,
                        paragraph_id_to_comments.get(paragraph_id)
                    )
                )

            for table_element, table_context in zip(table_elements, table_contexts):
                element = dict(table_element)

                element['label'] = table_context['label']
                element['comments'] = [
                    {
                        'id': comment_id,
                        'body': {
                            'id': table_context['label']
                        }
                    }
                ]
                element['context'] = table_context['context']

                elements.append(element)

                comment_id += 1

            for table_element in table_elements_empty:
                elements.append(table_element)

            logger.debug(
                'Found %d tables and %d paragraphs in file %s (%d/%d)',
                n_tables,
                0 if n_paragraphs is None else n_paragraphs,
                file,
                i_file,
                n_files
            )
            i_file += 1

            with open(output_file, 'w', encoding = 'utf-8') as file:
                dump(
                    {
                        'elements': elements
                    }, file, indent = INDENT, ensure_ascii = False
                )


def prepare_source(input_path: str, output_path: str, train_fraction: float):
    for root, _, files in walk(input_path):
        for file in files:

            path = os_path.join(root, file)

            output_file = os_path.join(output_path, f'{Path(file).stem}.json')

            if not os_path.isdir(output_path):
                mkdir(output_path)

            comments, content = get_comments(path)
            elements = get_elements(content, comments)

            records = []

            label_to_paragraph_ids = {}
            paragraph_id_to_xml = {}
            paragraph_id_to_element = {}

            paragraphs = Queue()

            # 1. Handle paragraph comments, group by target (table) id

            for element, comments in elements:
                if element.tag.endswith('}p'):

                    # style = get_paragraph_style(element)

                    # if style is None:
                    #     print(get_xml(element))

                    paragraph = Paragraph.from_xml(element, comments)
                    paragraphs.put(paragraph)

                    if paragraph:
                        if comments is not None:
                            paragraph.subset = Subset.random(train_fraction)
                            for comment in comments:
                                if (label := comment.body.target) is not None:
                                    if (paragraph_ids := label_to_paragraph_ids.get(comment.body.target)) is None:
                                        label_to_paragraph_ids[label] = [(paragraph.id, comment.body.score)]
                                    else:
                                        paragraph_ids.append((paragraph.id, comment.body.score))

                                    paragraph_id_to_xml[paragraph.id] = get_xml(element)
                                    paragraph_id_to_element[paragraph.id] = element

            # 2. Handle table comments, extract linked paragraphs

            for element, comments in elements:
                if element.tag.endswith('}p'):
                    paragraph = paragraphs.get()

                    if paragraph:
                        records.append(paragraph.json)
                else:
                    label = None

                    if comments is not None:
                        for comment in comments:
                            if comment.body.id is not None:
                                label = comment.body.id

                    context = None

                    if label in label_to_paragraph_ids:
                        table_paragraphs = []

                        for paragraph, score in sorted(label_to_paragraph_ids.get(label), key = lambda entry: entry[1], reverse = True):
                            if context is None:
                                context = {paragraph: score}
                            else:
                                context[paragraph] = score

                            xml = paragraph_id_to_xml.get(paragraph)

                            if xml is None:
                                logger.warning('Missing xml for paragraph %s', paragraph)
                            else:
                                table_paragraphs.append(paragraph_id_to_element[paragraph])

                    elif label is not None:
                        logger.warning('Table %s is missing annotations', label)

                    table = Table.from_xml(
                        element,
                        label,
                        comments,
                        context
                    )

                    records.append(table.json)

            with open(output_file, 'w', encoding = 'utf-8') as file:
                dump(
                    {
                        'elements': records
                    }, file, indent = INDENT, ensure_ascii = False
                )
