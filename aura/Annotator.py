import re
from os import path as os_path, mkdir, walk
from tqdm import tqdm

from logging import getLogger
from time import time

from .VllmClient import VllmClient
from .util import make_system_prompt, make_annotation_prompt, read_elements, dict_to_string, string_to_dict, dict_to_json_file, get_xml
from .document import Paragraph, Cell, Table
from .document.ZipFile import ZipFile


logger = getLogger(__name__)


TABLE_TITLE_PATTERN = re.compile(r'таблица\s+([^ ]+)\s+.*')


def generate_batches(items: list[int], n: int):
    if n is None or n < 1 or n > len(items):
        return [items]
    return [items[i: i + n] for i in range(0, len(items), n)]


class Annotator:
    def __init__(self, host: str, port: int, model: str):
        self.llm = VllmClient(host, port, model, make_system_prompt())

    def annotate(self, input_path: str, output_path: str, batch_size: int = None, n_batches: int = None, table_label_search_window: int = 5, score_threshold: float = 0.5):
        if not os_path.isdir(output_path):
            mkdir(output_path)

        llm = self.llm

        for root, _, files in walk(input_path):
            for file in files:
                if not file.endswith('.docx'):
                    continue

                comment_id = 0

                tables = []
                paragraphs = []

                table_label_cahdidates = []

                elements = read_elements(os_path.join(root, file))
                file_with_comments = ZipFile(os_path.join(root, file))

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
                        label = 'undefined'

                        while i < table_label_search_window:
                            title = table_label_cahdidates.pop().lower()
                            i += 1

                            match = TABLE_TITLE_PATTERN.match(title)

                            if match is not None:
                                label = match.group(1)
                                break

                        table_label_search_window = []

                        table = Table.from_xml(element, label = label)

                        tables.append(table)

                batched_paragraphs = generate_batches(paragraphs, batch_size)

                if n_batches is not None:
                    batched_paragraphs = batched_paragraphs[:n_batches]

                for table in tables:
                    llm.reset()

                    file_with_comments.insert_comment(table.xml, table.label, comment_id = comment_id, tag = 'tbl')

                    comment_id += 1

                    start = time()

                    prompt = make_annotation_prompt(
                        table = Cell.serialize_rows(
                            table.rows,
                            with_embeddings = False
                        )
                    )

                    completion = llm.complete(prompt)

                    for paragraphs_batch in tqdm(batched_paragraphs):
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

                        for paragraph in paragraphs_batch:
                            result = paragraph_scores.get(paragraph.id)

                            if result is None:
                                logger.warning('Paragraph "%s" is missing relevance score', paragraph['text'])
                            else:
                                score = result.get('score')
                                comment = result.get('comment')

                                if score > score_threshold:
                                    file_with_comments.insert_comment(
                                        paragraph.xml,
                                        f'{table.label} {score:.3f}' if comment is None else f'{table.label} {score:.3f} {comment}',
                                        comment_id = comment_id
                                    )
                                    comment_id += 1

                    logger.warning(f'Annotated table in {time() - start:.3f} seconds')

                file_with_comments.save('assets/test-live.docx')
