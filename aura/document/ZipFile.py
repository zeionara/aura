import os
from zipfile import ZipFile as ZipFileBase
from lxml import etree
from datetime import datetime

from .Document import DOCX_ELEMENT_TEMPLATE_PATH
from ..util import get_condensed_xml, replace_last_occurrence


with open(os.path.join(DOCX_ELEMENT_TEMPLATE_PATH, 'comment.xml'), 'r', encoding = 'utf-8') as file:
    COMMENT_TEMPLATE = file.read()


class ZipFile:
    def __init__(self, path: str):
        self.path = path

        with ZipFileBase(path) as docx_zip:
            try:
                self.comments = etree.XML(
                    docx_zip.read('word/comments.xml')
                )
            except KeyError:
                self.comments = None

            self.document = docx_zip.read('word/document.xml').decode('utf-8')

    def insert_comment(self, xml: etree.Element, comment_text: str, comment_id: int = 0, author: str = 'Zeio Nara', date: datetime = None):
        if date is None:
            date = datetime.now()

        condensed_xml = get_condensed_xml(xml)

        condensed_xml_with_comment = replace_last_occurrence(
            condensed_xml.replace('<w:r ', f'<w:commentRangeStart w:id="{comment_id}"/><w:r ', 1),
            '</w:r>',
            f'</w:r><w:commentRangeEnd w:id="{comment_id}"/>'
        )

        self.document = self.document.replace(condensed_xml, condensed_xml_with_comment)

        comment = COMMENT_TEMPLATE.format(
            author = author,
            id = id,
            date = date.strftime('%Y-%m-%dT%H:%M:%SZ'),
            text = comment_text
        )

        print(comment)
