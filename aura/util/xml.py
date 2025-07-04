from lxml import etree


WORD_NAMESPACE = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}


def get_text(root: etree.Element):
    return root.xpath('string(.)', namespaces = WORD_NAMESPACE)


def get_xml(root: etree.Element):
    return etree.tostring(root, encoding = str)


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

    if (n_comments := len(comments.values())) > 0:
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
