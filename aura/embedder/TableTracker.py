from yaml import dump
from numpy import mean
from dataclasses import dataclass, asdict


@dataclass
class Nrows:
    original: int
    skipped: int | None = None
    updated: int | None = None

    @property
    def correctness_ratio(self):
        return (self.original - self.skipped) / self.original


@dataclass
class Table:
    broken: bool
    n_rows: Nrows


class TableTracker:
    def __init__(self):
        self.n_tables = 0

        self.document_to_tables = {}

    def register_table(self, document: str, table: str, n_rows: int, broken: bool = False, n_rows_skipped: int = None, n_rows_updated: int = None):
        if (tables := self.document_to_tables.get(document)) is None:
            self.document_to_tables[document] = {
                table: Table(
                    broken=broken,
                    n_rows=Nrows(
                        original=n_rows,
                        skipped=n_rows_skipped,
                        updated=n_rows_updated
                    )
                )
            }
        else:
            if table not in tables:
                tables[table] = Table(
                    broken=broken,
                    n_rows=Nrows(
                        original=n_rows,
                        skipped=n_rows_skipped,
                        updated=n_rows_updated
                    )
                )

    def count(self):
        i = 0

        for document in self.document_to_tables:
            for _ in self.document_to_tables.get(document):
                i += 1

        return i

    def count_broken(self):
        i = 0

        ratios = []

        for document in self.document_to_tables:
            for table in self.document_to_tables.get(document).values():
                if table.broken:
                    ratios.append(table.n_rows.correctness_ratio)

                    i += 1

        return i, mean(ratios)

    def export(self, path: str, broken_only: bool = True):
        data = {}

        for document, tables in self.document_to_tables.items():
            n_tables = 0
            document_data = {}

            for label, table in tables.items():
                if not broken_only or table.broken:
                    n_tables += 1

                    table_data = asdict(table)

                    if broken_only:
                        table_data.pop('broken')

                    document_data[label] = table_data

            if n_tables > 0:
                data[document] = document_data

        with open(path, 'w', encoding='utf-8') as file:
            dump(data, file, indent=2, allow_unicode=True)
