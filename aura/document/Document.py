import os
import re
import uuid

from docx import Document as DocxDocument
from docx.oxml import parse_xml
from docx.text.paragraph import Paragraph
from docx.table import Table


DOCX_ELEMENT_TEMPLATE_PATH = 'assets/docx-element-templates'

with open(os.path.join(DOCX_ELEMENT_TEMPLATE_PATH, 'header.xml'), 'r', encoding = 'utf-8') as file:
    HEADER_ELEMENT_TEMPLATE = file.read()

with open(os.path.join(DOCX_ELEMENT_TEMPLATE_PATH, 'paragraph.xml'), 'r', encoding = 'utf-8') as file:
    PARAGRAPH_ELEMENT_TEMPLATE = file.read()

BOOKMARK_START = re.compile(r'<w:bookmarkStart[^>]*/>')
BOOKMARK_END = re.compile(r'<w:bookmarkEnd[^>]*/>')


class Document:
    def __init__(self):
        self.elements = []
        self.docx = DocxDocument()

    def _remove_bookmarks(self, xml_string: str):
        cleaned = BOOKMARK_START.sub('', xml_string)
        cleaned = BOOKMARK_END.sub('', cleaned)

        return cleaned

    def append(self, element):
        element = self._remove_bookmarks(element)

        self.elements.append(
            element
        )

    def append_h1(self, text: str):
        self.append(HEADER_ELEMENT_TEMPLATE.format(text = text, style = 'Heading1', id = str(uuid.uuid4())))

    def append_paragraph(self, text: str):
        self.append(PARAGRAPH_ELEMENT_TEMPLATE.format(text = text, id = str(uuid.uuid4())))

    def append_h2(self, text: str):
        self.append(HEADER_ELEMENT_TEMPLATE.format(text = text, style = 'Heading2', id = str(uuid.uuid4())))

    def append_h3(self, text: str):
        self.append(HEADER_ELEMENT_TEMPLATE.format(text = text, style = 'Heading3', id = str(uuid.uuid4())))

    def append_h4(self, text: str):
        self.append(HEADER_ELEMENT_TEMPLATE.format(text = text, style = 'Heading4', id = str(uuid.uuid4())))

    def append_h5(self, text: str):
        self.append(HEADER_ELEMENT_TEMPLATE.format(text = text, style = 'Heading5', id = str(uuid.uuid4())))

    def to_docx(self, filename):
        self.docx.element.body.clear_content()

        for xml_string in self.elements:
            xml_element = parse_xml(xml_string)
            if xml_element.tag.endswith('p'):  # Paragraph
                paragraph = Paragraph(xml_element, self.docx.element.body)
                self.docx.element.body.append(paragraph._element)
            elif xml_element.tag.endswith('tbl'):  # Table
                table = Table(xml_element, self.docx.element.body)
                self.docx.element.body.append(table._element)

        self.docx.save(filename)
