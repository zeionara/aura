from os import path as os_path

from logging import getLogger, DEBUG, INFO

from .util import dict_from_json_file


logger = getLogger(__name__)
# logger.setLevel(DEBUG)
logger.setLevel(INFO)


class Annotations:
    def __init__(self, table_to_paragraphs: dict = None):
        self.table_to_paragraphs = {} if table_to_paragraphs is None else table_to_paragraphs

        self.table_labels = list(self.table_to_paragraphs.keys())

        table_to_paragraph_ids = {}

        for table_label in self.table_labels:
            table_paragraph_ids = []

            if (table := self.table_to_paragraphs.get(table_label)) is None:
                raise ValueError(f'Undefined table {table_label}')

            if (paragraphs := table.get('paragraphs')) is None:
                raise ValueError(f'Table {table_label} does not have associated paragraph annotations')

            for paragraph in paragraphs:
                # if isinstance(paragraph, str):
                #     print(paragraph)
                if (paragraph_id := paragraph.get('id')) is None:
                    raise ValueError(f'Paragraph has no id: {paragraph}')

                table_paragraph_ids.append(paragraph_id)

            table_to_paragraph_ids[table_label] = table_paragraph_ids

        self._table_to_paragraph_ids = table_to_paragraph_ids

    @classmethod
    def from_file(cls, path: str):
        if os_path.isfile(path):
            return cls(
                table_to_paragraphs = dict_from_json_file(path)
            )

        raise ValueError(f'File {path} does not exist')

    @property
    def n_tables(self):
        return len(self.table_labels)

    @property
    def paragraph_ids(self):
        paragraph_ids = []

        for table_label, table_paragraph_ids in self._table_to_paragraph_ids.items():
            if len(table_paragraph_ids) > len(paragraph_ids):
                paragraph_ids = table_paragraph_ids

            logger.debug('Table %s has %d annotated paragraphs', table_label, len(table_paragraph_ids))

        logger.debug('The most annotated table has %d annotated paragraphs', len(paragraph_ids))

        return paragraph_ids

    def is_complete(self, n_tables: int, n_paragraphs: int):
        if len(self.table_labels) < n_tables:
            return False

        if len(self.table_labels) > n_tables:
            raise ValueError(f'Number of tables in the annotation list is greater than provided number ({len(self.table_labels)} > {n_tables})')

        for _, paragraph_ids in self._table_to_paragraph_ids.items():
            if len(paragraph_ids) < n_paragraphs:
                return False

            if len(paragraph_ids) > n_paragraphs:
                raise ValueError(f'Number of paragraphs in the annotation list is greater than provided number ({len(paragraph_ids)} > {n_paragraphs})')

        return True
