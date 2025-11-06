import statistics

from lxml import etree

from .util import get_text


def get_average(values: list[int]):
    return sum(values) / len(values) if len(values) > 0 else None


def get_stdev(values: list[int]):
    return statistics.stdev(values) if len(values) > 1 else 0 if len(values) > 0 else None


class TableStats:
    def __init__(self, label: str, paragraphs: list[str]):
        self.label = label
        self.paragraphs = paragraphs

        paragraph_lengths = [len(paragraph) for paragraph in paragraphs]

        self.n_paragraphs = len(paragraph_lengths)

        self.paragraph_length_average = get_average(paragraph_lengths)
        self.paragraph_length_stdev = get_stdev(paragraph_lengths)


class DocumentStats:
    def __init__(self, label: str):
        self.label = label

        self.n_tables = 0
        self.label_to_table_stats = {}

    def append_table(self, label: str, paragraphs: list[etree.Element]):
        if label in self.label_to_table_stats:
            raise ValueError(f'There are two tables with the same label \'{label}\' in document \'{self.label}\'')

        self.label_to_table_stats[label] = TableStats(label, [get_text(paragraph) for paragraph in paragraphs])
        self.n_tables += 1

    @property
    def paragraph_lengths(self):
        return [
            len(paragraph)
            for stats in self.label_to_table_stats.values()
            for paragraph in stats.paragraphs
        ]

    @property
    def n_paragraphs(self):
        return [stats.n_paragraphs for stats in self.label_to_table_stats.values()]

    @property
    def paragraph_length_average(self):
        return get_average(self.paragraph_lengths)

    @property
    def paragraph_length_stdev(self):
        return get_stdev(self.paragraph_lengths)

    @property
    def n_paragraphs_average(self):
        return get_average(self.n_paragraphs)

    @property
    def n_paragraphs_stdev(self):
        return get_stdev(self.n_paragraphs)


class Stats:
    def __init__(self):
        self.label_to_document_stats = {}
        self.n_documents = 0

    def append_document(self, label: str):
        if label in self.label_to_document_stats:
            raise ValueError(f'There are two documents with the same label \'{label}\'')

        self.label_to_document_stats[label] = DocumentStats(label)
        self.n_documents += 1

    def append_table(self, document: str, table: str, paragraphs: list[etree.Element]):
        if document not in self.label_to_document_stats:
            self.append_document(document)

        return self.label_to_document_stats[document].append_table(table, paragraphs)

    @property
    def paragraph_lengths(self):
        return [
            len(paragraph)
            for document_stats in self.label_to_document_stats.values()
            for table_stats in document_stats.label_to_table_stats.values()
            for paragraph in table_stats.paragraphs
        ]

    @property
    def n_paragraphs(self):
        return [
            table_stats.n_paragraphs
            for document_stats in self.label_to_document_stats.values()
            for table_stats in document_stats.label_to_table_stats.values()
        ]

    @property
    def n_tables(self):
        return [
            stats.n_tables
            for stats in self.label_to_document_stats.values()
        ]

    @property
    def paragraph_length_average(self):
        return get_average(self.paragraph_lengths)

    @property
    def paragraph_length_stdev(self):
        return get_stdev(self.paragraph_lengths)

    @property
    def n_paragraphs_average(self):
        return get_average(self.n_paragraphs)

    @property
    def n_paragraphs_stdev(self):
        return get_stdev(self.n_paragraphs)

    @property
    def n_tables_average(self):
        return get_average(self.n_tables)

    @property
    def n_tables_stdev(self):
        return get_stdev(self.n_tables)

    def __iter__(self):
        return iter(self.label_to_document_stats.items())
