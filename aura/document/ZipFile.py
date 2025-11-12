import os
from zipfile import ZipFile as ZipFileBase
from lxml import etree
from datetime import datetime, timezone, timedelta

from .Document import DOCX_ELEMENT_TEMPLATE_PATH
from ..util import get_condensed_xml, replace_last_occurrence, WORD_NAMESPACES, get_xml


TZ_OFFSET = timezone(timedelta(hours=3), name='MSK')


with open(os.path.join(DOCX_ELEMENT_TEMPLATE_PATH, 'comment.xml'), 'r', encoding = 'utf-8') as file:
    COMMENT_TEMPLATE = file.read()


def dumps(xml: etree.Element):
    return etree.tostring(
        xml,
        pretty_print=True,
        encoding='utf-8',
        xml_declaration = True,
        standalone = True
    ).decode('utf-8')


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
        # 1. Update word/document.xml

        condensed_xml = get_condensed_xml(xml)

        condensed_xml_with_comment = replace_last_occurrence(
            condensed_xml.replace('<w:r ', f'<w:commentRangeStart w:id="{comment_id}"/><w:r ', 1),
            '</w:r>',
            f'</w:r><w:commentRangeEnd w:id="{comment_id}"/>'
        )

        self.document = self.document.replace(condensed_xml, condensed_xml_with_comment)

        # 2. Update word/comments.xml

        if self.comments is None:
            comments = etree.Element(f'{{{WORD_NAMESPACES["w"]}}}comments', nsmap = WORD_NAMESPACES)
        else:
            comments = self.comments

        if date is None:
            date = datetime.now(tz = TZ_OFFSET)

        comment = COMMENT_TEMPLATE.format(
            author = author,
            id = comment_id,
            date = date.strftime('%Y-%m-%dT%H:%M:%SZ'),
            text = comment_text
        )
        comment = etree.fromstring(comment)

        comments.append(comment)

        self.comments = etree.tostring(
            comments,
            pretty_print=True,
            encoding='utf-8',
            xml_declaration = True,
            standalone = True
        ).decode('utf-8')

    def save(self, path=None):
        if path is None:
            path = self.path

        temp_path = path + '.tmp'

        # found_document_rels = False

        with ZipFileBase(self.path, 'r') as source_zip:
            init_comments = self.comments is not None and 'word/comments.xml' not in [item.filename for item in source_zip.infolist()]

            with ZipFileBase(temp_path, 'w') as target_zip:
                # Copy all files from source to target, replacing the specific files
                for item in source_zip.infolist():
                    if item.filename == 'word/document.xml':
                        # Replace with updated document
                        target_zip.writestr(item, self.document)
                    elif item.filename == 'word/comments.xml':
                        # Replace with updated comments if they exist
                        if self.comments is not None:
                            target_zip.writestr(item, self.comments)
                    elif item.filename == '[Content_Types].xml' and init_comments:
                        content_types = etree.XML(source_zip.read(item))

                        content_types.append(
                            etree.fromstring(
                                '<Override ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.comments+xml" PartName="/word/comments.xml"/>'
                            )
                        )

                        target_zip.writestr(item, dumps(content_types))
                    elif item.filename == 'word/_rels/document.xml.rels' and init_comments:
                        rels = etree.XML(source_zip.read(item))

                        rels.append(
                            etree.fromstring('<Relationship Id="rId100" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments" Target="comments.xml"/>')
                        )

                        target_zip.writestr(item, dumps(rels))

                        # target_zip.writestr(item, '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId8" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/fontTable" Target="fontTable.xml"/><Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings" Target="settings.xml"/><Relationship Id="rId7" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" Target="footer1.xml"/><Relationship Id="rId2" Type="http://schemas.microsoft.com/office/2007/relationships/stylesWithEffects" Target="stylesWithEffects.xml"/><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/><Relationship Id="rId6" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/header" Target="header1.xml"/><Relationship Id="rId5" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/image1.png"/><Relationship Id="rId4" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/webSettings" Target="webSettings.xml"/><Relationship Id="rId9" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="theme/theme1.xml"/><Relationship Id="rId10" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments" Target="comments.xml"/></Relationships>')
                    # elif item.filename == '_rels/.rels':
                    #     target_zip.writestr('_rels/.rels', '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="theme/theme1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments" Target="comments.xml"/></Relationships>')
                    else:
                        # Copy all other files as-is
                        target_zip.writestr(item, source_zip.read(item.filename))

                    # if item.filename == '_rels/document.xml.rels':
                    #     found_document_rels = True

                # if not found_document_rels:
                #     target_zip.writestr('_rels/.rels', '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="theme/theme1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments" Target="comments.xml"/></Relationships>')

                # If comments.xml didn't exist in original but we created it, add it now
                if init_comments:
                    target_zip.writestr('word/comments.xml', self.comments)

        os.replace(temp_path, path)
