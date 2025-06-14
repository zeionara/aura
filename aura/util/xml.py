from lxml import etree


WORD_NAMESPACE = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}


def get_xml_text(root):
    return root.xpath('string(.)', namespaces = WORD_NAMESPACE)


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
