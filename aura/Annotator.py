from os import path as os_path, mkdir, walk
from tqdm import tqdm

from logging import getLogger
from time import time

from .VllmClient import VllmClient
from .util import make_system_prompt, make_annotation_prompt, read_elements, dict_to_string, string_to_dict, dict_to_json_file
from .document import Paragraph, Cell, Table


logger = getLogger(__name__)


def generate_batches(items: list[int], n: int):
    if n is None or n < 1 or n > len(items):
        return [items]
    return [items[i: i + n] for i in range(0, len(items), n)]


class Annotator:
    def __init__(self, host: str, port: int, model: str):
        self.llm = VllmClient(host, port, model, make_system_prompt())

    def annotate(self, input_path: str, output_path: str, batch_size: int = None):

        if not os_path.isdir(output_path):
            mkdir(output_path)

        tables = []
        paragraphs = []

        for root, _, files in walk(input_path):
            for file in files:
                if not file.endswith('.docx'):
                    continue

                elements = read_elements(os_path.join(root, file))

                for element in elements:
                    if element.tag.endswith('}p'):
                        paragraph = Paragraph.from_xml(element)

                        if paragraph:
                            paragraphs.append({
                                'id': paragraph.id,
                                'text': paragraph.text
                            })
                    else:
                        table = Table.from_xml(element)

                        tables.append(
                            Cell.serialize_rows(
                                table.rows,
                                with_embeddings = False
                            )
                        )

        llm = self.llm

        batched_paragraphs = generate_batches(paragraphs, batch_size)

        # print(batched_paragraphs[0])

        for table in tables:
            llm.reset()

            start = time()

            prompt = make_annotation_prompt(
                table = table
            )

            completion = llm.complete(prompt)

            for paragraphs_batch in tqdm(batched_paragraphs):
                completion = llm.complete(
                    dict_to_string(
                        paragraphs_batch
                    ),
                    add_to_history = False
                )

                paragraph_scores = string_to_dict(completion)

                for paragraph in paragraphs_batch:
                    score = paragraph_scores.get(paragraph['id'])

                    if score is None:
                        logger.warning('Paragraph "%s" is missing relevance score', paragraph['text'])
                    else:
                        paragraph['score'] = float(score)

            logger.warning(f'Annotated table in {time() - start:.3f} seconds')

            dict_to_json_file({'paragraphs': paragraphs}, 'assets/scores.json')
