from typing import ClassVar
from json import dump

from lxml import etree


from .ReferentiableObject import ReferentiableObject
from .Item import Item, INDENT, INCLUDE_XML

from ..util.xml import get_text, get_paragraph_style
from ..util.string import normalize_spaces


class Paragraph(ReferentiableObject, Item):
    type_label: ClassVar[str] = 'paragraph'

    def __init__(self, xml: etree.Element, text: str, style: str, id_: str = None):
        self.xml = xml
        self.text = text

        self.style = style

        super().__init__(id_)

    @classmethod
    def from_xml(cls, xml: etree.Element):
        text = normalize_spaces(get_text(xml))

        if not text:
            return None

        return cls(
            xml,
            text = text,
            style = get_paragraph_style(xml)
        )

    @property
    def content(self):
        return self.text

    def to_json(self, path: str = None, indent: int = INDENT):
        data = {
            'type': self.type_label,
            'id': self.id,
            'text': self.text,
            'style': self.style,
            'embedding': '<a long array of numbers>'
        }

        if INCLUDE_XML:
            data['xml'] = str(self.soup)

        if path is not None:
            with open(path, 'w', encoding = 'utf-8') as file:
                dump(data, file, indent = indent, ensure_ascii = False)

        return data

    @property
    def json(self):
        return self.to_json()
