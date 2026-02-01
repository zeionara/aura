import re
from lxml import etree

from .string import normalize_spaces, reduce_spaces  # , replace_last_occurrence


WORD_NAMESPACE = {
    'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
}
WORD_NAMESPACES = {
    'mc': "http://schemas.openxmlformats.org/markup-compatibility/2006",
    'o': "urn:schemas-microsoft-com:office:office",
    'r': "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    'm': "http://schemas.openxmlformats.org/officeDocument/2006/math",
    'v': "urn:schemas-microsoft-com:vml",
    'wp': "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    'w10': "urn:schemas-microsoft-com:office:word",
    'w': "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    'wne': "http://schemas.microsoft.com/office/word/2006/wordml",
    'sl': "http://schemas.openxmlformats.org/schemaLibrary/2006/main",
    'a': "http://schemas.openxmlformats.org/drawingml/2006/main",
    'pic': "http://schemas.openxmlformats.org/drawingml/2006/picture",
    'c': "http://schemas.openxmlformats.org/drawingml/2006/chart",
    'lc': "http://schemas.openxmlformats.org/drawingml/2006/lockedCanvas",
    'dgm': "http://schemas.openxmlformats.org/drawingml/2006/diagram",
    'wps': "http://schemas.microsoft.com/office/word/2010/wordprocessingShape",
    'wpg': "http://schemas.microsoft.com/office/word/2010/wordprocessingGroup",
    'w14': "http://schemas.microsoft.com/office/word/2010/wordml",
    'w15': "http://schemas.microsoft.com/office/word/2012/wordml",
    'w16': "http://schemas.microsoft.com/office/word/2018/wordml",
    'w16cex': "http://schemas.microsoft.com/office/word/2018/wordml/cex",
    'w16cid': "http://schemas.microsoft.com/office/word/2016/wordml/cid",
    'cr': "http://schemas.microsoft.com/office/comments/2020/reactions",
    None: "http://schemas.microsoft.com/office/tasks/2019/documenttasks"
}
XMLNS_PROPERTY_PATTERN = re.compile(r'xmlns:[a-z0-9]+="[^" ]+"')


# def insert_comment(xml: etree.Element, comment_text: str, comment_id: int = 0):
#     condensed_xml = get_condensed_xml(xml)
#
#     condensed_xml_with_comment = replace_last_occurrence(
#         condensed_xml.replace('<w:r ', f'<w:commentRangeStart w:id="{comment_id}"/><w:r ', 1),
#         '</w:r>',
#         f'</w:r><w:commentRangeEnd w:id="{comment_id}"/>'
#     )
#
#     print(condensed_xml_with_comment)


def get_text(root: etree.Element):
    return root.xpath('string(.)', namespaces = WORD_NAMESPACE)


def get_xml(root: etree.Element):
    return etree.tostring(root, encoding = str)


def get_condensed_xml(root: etree.Element):
    xml = etree.tostring(root, encoding = str)
    return reduce_spaces(XMLNS_PROPERTY_PATTERN.sub('', xml)).replace(' >', '>')


def iterchildren(element: etree.Element):
    for child in element.iterchildren():
        yield child

        for grandchild in iterchildren(child):
            yield grandchild


def get_elements(content: etree.XML, comments: dict = None):
    elements = list(content.xpath('//w:p|//w:tbl', namespaces = WORD_NAMESPACE))

    # Make sure that only the topmost elements appear in list

    for element in list(elements):
        leaf = element

        while element is not None:
            if (element := element.getparent()) in elements:
                elements.remove(leaf)
                break

    elements_with_comments = []

    for element in elements:
        if comments is None:
            elements_with_comments.append((element, None))
        else:
            element_comments = []

            for comment in list(comments.values()):
                if comment.target == element:
                    element_comments.append(
                        comments.pop(comment.id)
                    )
                elif comment.target in iterchildren(element):
                    comment.target = element
                    element_comments.append(
                        comments.pop(comment.id)
                    )

            elements_with_comments.append((element, tuple(element_comments) if len(element_comments) > 0 else None))

    if comments is not None and (n_comments := len(comments.values())) > 0:
        raise ValueError(f'Left {n_comments} unresolved comments')

    return elements_with_comments


def get_paragraph_style(paragraph: etree.Element):
    styles = paragraph.xpath('.//w:pStyle', namespaces = WORD_NAMESPACE)

    if len(styles) > 0:
        values = styles[0].xpath('.//@w:val', namespaces = WORD_NAMESPACE)
        if len(values) > 0:
            return values[0]

    return None


def should_merge_vertically(cell: etree.Element):
    vmerges = cell.xpath('.//w:vMerge', namespaces = WORD_NAMESPACE)

    if len(vmerges) > 0:
        values = vmerges[0].xpath('.//@w:val', namespaces = WORD_NAMESPACE)

        if len(values) > 0:
            return values[0] != 'restart'
        else:
            return True

    return False


def get_horizontal_span_size(cell: etree.Element):
    hspans = cell.xpath('.//w:gridSpan', namespaces = WORD_NAMESPACE)

    if len(hspans) > 0:
        values = hspans[0].xpath('.//@w:val', namespaces = WORD_NAMESPACE)
        if len(values) > 0:
            return int(values[0])

    return None
