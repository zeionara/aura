from dataclasses import dataclass

from lxml import etree

from ..util.string import normalize_spaces


class CommentBody:
    def __init__(self, text: str):
        parts = normalize_spaces(text).split(' ', maxsplit = 2)

        if len(parts) < 2:
            self.id = parts[0]

            self.target = None
            self.score = None
            self.note = None
        else:
            self.id = None

            self.target = parts[0]
            self.score = float(parts[1])

            self.note = parts[2] if len(parts) > 2 else None

    @property
    def is_table(self):
        return self.id is not None

    def __repr__(self):
        return f"CommentBody(id={self.id}, target={self.target}, score={self.score}, note={self.note})"


@dataclass
class Comment:
    id: int
    body: CommentBody
    target: etree.Element = None
