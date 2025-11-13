import re
from os import path as os_path, mkdir, walk
from tqdm import tqdm

from logging import getLogger
from time import time

from .VllmClient import VllmClient
from .util import make_system_prompt, make_annotation_prompt, read_elements, dict_to_string, string_to_dict, dict_to_json_file, get_xml, dict_to_json_file
from .document import Paragraph, Cell, Table
from .document.ZipFile import ZipFile


logger = getLogger(__name__)


TABLE_TITLE_PATTERN = re.compile(r'таблица\s+([^ ]+)\s+.*')


def generate_batches(items: list[int], n: int):
    if n is None or n < 1 or n > len(items):
        return [items]
    return [items[i: i + n] for i in range(0, len(items), n)]


class Annotator:
    def __init__(self, llms: list[VllmClient]):
        self.llms = llms

    def annotate(self, input_path: str, output_path: str, batch_size: int = None, n_batches: int = None, table_label_search_window: int = 5):
        if not os_path.isdir(output_path):
            mkdir(output_path)

        llms = self.llms

        for root, _, files in walk(input_path):
            for file in files:
                if not file.endswith('.docx'):
                    continue

                comment_id = 0

                tables = []
                paragraphs = []

                annotations = {}
                offset = 0

                table_label_cahdidates = []

                elements = read_elements(os_path.join(root, file))
                # file_with_comments = ZipFile(os_path.join(root, file))
                output_file = file[:-5] + '.json'

                for element in elements:
                    if element.tag.endswith('}p'):
                        paragraph = Paragraph.from_xml(element)

                        if paragraph:
                            paragraphs.append(paragraph)
                            table_label_cahdidates.append(paragraph.text)
                            # paragraphs.append({
                            #     'id': paragraph.id,
                            #     'text': paragraph.text
                            # })
                    else:
                        i = 0
                        label = None

                        while i < table_label_search_window:
                            title = table_label_cahdidates.pop().lower()
                            i += 1

                            match = TABLE_TITLE_PATTERN.match(title)

                            if match is not None:
                                label = match.group(1)
                                break

                        if label is None:
                            logger.error('Couldn\'t find table label')
                        else:
                            annotations[label] = {
                                'type': 'table',
                                'paragraphs': []
                            }

                        table_label_search_window = []

                        table = Table.from_xml(element, label = label)

                        tables.append(table)

                batched_paragraphs = generate_batches(paragraphs, batch_size)

                if n_batches is not None:
                    batched_paragraphs = batched_paragraphs[:n_batches]

                for table in tables:
                    for llm in llms:
                        llm.reset()

                    if table.label is None or table.label not in annotations:
                        continue

                    # file_with_comments.insert_comment(table.xml, table.label, comment_id = comment_id, tag = 'tbl')

                    # comment_id += 1

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
                                    # score = result.get('score')
                                    # comment = result.get('comment')

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

                                    # if score > score_threshold:
                                    #     file_with_comments.insert_comment(
                                    #         paragraph.xml,
                                    #         f'{table.label} {score:.3f}' if comment is None else f'{table.label} {score:.3f} {comment}',
                                    #         comment_id = comment_id
                                    #     )
                                    #     comment_id += 1

                            iteration += 1

                        offset += len(batched_paragraphs)

                    logger.warning(f'Annotation completed in in {time() - start:.3f} seconds')

                dict_to_json_file(
                    annotations,
                    os_path.join(output_path, output_file)
                )

                # file_with_comments.save('assets/test-live.docx')
