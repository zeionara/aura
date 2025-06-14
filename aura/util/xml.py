from lxml import etree


WORD_NAMESPACE = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}


def get_xml_text(root):
    return root.xpath('string(.)', namespaces = WORD_NAMESPACE)

    # text_parts = []

    # for element in root.iter():
    #     if element.text:
    #         text_parts.append(element.text)
    #     if element.tail:
    #         text_parts.append(element.tail)

    # return "".join(text_parts).strip()


def iterchildren(element: etree.Element):
    for child in element.iterchildren():
        yield child

        for grandchild in iterchildren(child):
            yield grandchild
