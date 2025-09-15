import re
from typing import ClassVar
from json import dump as dump_json

from lxml import etree

from .ReferentiableObject import ReferentiableObject
from .Item import Item, INCLUDE_XML, INDENT
from .Cell import Cell
from .Comment import Comment

from .util import get_aligned_cell
from ..util.string import normalize_spaces, drop_space_around_punctuation
from ..util.xml import WORD_NAMESPACE, should_merge_vertically, get_horizontal_span_size, get_text  # , get_xml


NOTE_PATTERN = re.compile(r'([*]+)\s+([^*]+[^*\s])')


class Table(ReferentiableObject, Item):
    type_label: ClassVar[str] = 'table'

    def __init__(self, xml: etree.Element, rows: list[list[Cell]], label: str, comments: tuple[Comment], context: list[str] = None, id_: str = None):
        self.rows = rows
        self.xml = xml
        self.embeddings = {}
        self.comments = comments
        self.context = context

        # self.id = id_
        self.label = label

        self._stats = None

        super().__init__(id_)

    @classmethod
    def from_xml(cls, xml: etree.Element, label: str = None, comments: tuple[Comment] = None, context: list[str] = None):
        rows = []
        last_row = None

        for row in xml.xpath('.//w:tr', namespaces = WORD_NAMESPACE):
            cells = []

            col_offset = 0

            for cell in row.xpath('.//w:tc', namespaces = WORD_NAMESPACE):
                if last_row is not None and should_merge_vertically(cell):
                    cells.append(
                        placeholder := get_aligned_cell(last_row, col_offset).make_placeholder()
                    )

                    (origin := placeholder.origin).n_rows += 1
                    col_offset += origin.n_cols
                else:
                    horizontal_span_size = get_horizontal_span_size(cell)

                    n_cols = 1 if horizontal_span_size is None else horizontal_span_size

                    col_offset += n_cols

                    cells.append(
                        Cell(
                            drop_space_around_punctuation(
                                normalize_spaces(
                                    get_text(cell)
                                )
                            ),
                            n_cols = n_cols
                        )
                    )

            last_row = cells
            rows.append(cells)

        # Parse notes - just add the note text in brackets after the original cell content without removing the anchor symbol(s)

        notes = {}

        for row in rows:
            for cell in row:
                if cell.text is not None:
                    for note in NOTE_PATTERN.findall(cell.text):
                        notes[note[0]] = note[1]

        for row in rows:
            for cell in row:
                for key in sorted(notes.keys(), key = len, reverse = True):
                    if cell.text is not None and cell.text.endswith(key):
                        cell.text = normalize_spaces(f'{cell.text} ({notes[key]})')

        return cls(xml, rows, label, comments, context)

    @property
    def content(self):
        return [
            [
                cell.text
                for cell in row
            ]
            for row in self.rows
        ]

    def to_json(self, path: str = None, indent: int = INDENT):
        data = Cell.serialize_rows(self.rows)

        if INCLUDE_XML:
            data['xml'] = str(self.soup)

        if (id_ := self.id) is not None:
            data['id'] = id_

        if (label := self.label) is not None:
            data['label'] = label

        if (comments := self.comments) is not None:
            data['comments'] = [comment.json for comment in comments]

        if (context := self.context) is not None:
            data['context'] = context

        data['type'] = self.type_label
        data['embeddings'] = self.embeddings

        if path is not None:
            with open(path, 'w', encoding = 'utf-8') as file:
                dump_json(data, file, indent = indent, ensure_ascii = False)

        return data

    @property
    def json(self):
        return self.to_json()
