from __future__ import annotations

from uuid import uuid4

from .ReferentiableObject import ReferentiableObject


class Cell(ReferentiableObject):
    def __init__(self, text: str, n_rows = 1, n_cols = 1, id_: str = None):
        self.text = text

        self.n_rows = n_rows
        self.n_cols = n_cols

        super().__init__(id_)

    def increment_n_rows(self):
        self.n_rows += 1

    @classmethod
    def from_json(cls, json: dict, id_to_cell: dict = None):
        cell_id = json['id']

        if (text := json.get('text')) is None:
            cell = Placeholder(id_to_cell[cell_id])
        else:
            cell = cls(text, json.get('rows'), json.get('cols'), cell_id)

        id_to_cell[cell_id] = cell

        return cell

    @property
    def id(self):
        if self._id is None:
            self._id = str(uuid4())

        return self._id

    @classmethod
    def merge_horizontally(cls, row: list[Cell]):
        merged_row = []

        current_text = None
        current_group = None

        for cell in row:
            if current_text is None:
                current_text = cell.text
                current_group = [cell]
            elif cell.text == current_text:
                current_group.append(cell)
            else:
                merged_row.append(cls(current_text, n_rows = 1, n_cols = len(current_group)))

                current_text = cell.text
                current_group = [cell]

        if current_group is not None:
            merged_row.append(cls(current_text, n_rows = 1, n_cols = len(current_group)))

        return merged_row

    @classmethod
    def merge_vertically(cls, rows: list[list[Cell]]):
        merged_rows = []

        last_row = None

        last_row_offset = None
        current_row_offset = None

        for row in rows:
            merged_row = []

            if last_row is None:
                last_row = row
                merged_rows.append(row)
                continue

            last_row_offset = 0
            current_row_offset = 0
            last_row_index = 0

            for cell in row:
                last_row_cell = last_row[last_row_index] if last_row_index < len(last_row) else None

                if last_row_cell is not None and last_row_cell == cell and last_row_offset == current_row_offset:
                    merged_row.append(last_row_cell.make_placeholder())
                    last_row_cell.increment_n_rows()
                else:
                    merged_row.append(cell)

                current_row_offset += cell.n_cols

                if last_row_cell is not None:
                    last_row_offset += last_row_cell.n_cols

                last_row_index += 1

            merged_rows.append(merged_row)
            last_row = merged_row

        return merged_rows

    def __repr__(self):
        return f'{self.text} {self.n_rows}x{self.n_cols}'

    def __eq__(self, other):
        return self.text == other.text and self.n_cols == other.n_cols

    def make_placeholder(self):
        return Placeholder(origin = self)

    def serialize(self, with_embeddings: bool = True):
        cell = {
            'id': self.id,
            'text': self.text,
            'rows': self.n_rows,
            'cols': self.n_cols
        }

        if with_embeddings:
            cell['embeddings'] = {}

        return cell

    @staticmethod
    def serialize_rows(rows: list[list[Cell]], with_embeddings: bool = True):
        data = {
            'rows': [
                [
                    cell.serialize(with_embeddings)
                    for cell in row
                ]
                for row in rows
            ]
        }

        return data

    @staticmethod
    def deserialize_rows(rows: list[list[dict]]):
        id_to_cell = {}

        return [
            [
                Cell.from_json(cell, id_to_cell)
                for cell in row
            ]
            for row in rows
        ]


class Placeholder:
    def __init__(self, origin: Cell):
        self.origin = origin

    def __eq__(self, other):
        return self.origin == other

    def increment_n_rows(self):
        return self.origin.increment_n_rows()

    def make_placeholder(self):
        return self

    @property
    def text(self):
        return None

    @property
    def n_rows(self):
        return self.origin.n_rows

    @property
    def n_cols(self):
        return self.origin.n_cols

    def serialize(self):
        return {
            'id': self.origin.id
        }

    def __repr__(self):
        return f'{self.origin.text} 👻 {self.n_rows}x{self.n_cols}'
