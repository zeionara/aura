from zipfile import ZipFile

from lxml import etree

from .xml import WORD_NAMESPACE, get_xml_text, iterchildren

from ..document import Comment, CommentBody


def get_comments(docxFileName):
    comments = {}

    with ZipFile(docxFileName) as docx_zip:
        comments_body_xml = etree.XML(
            docx_zip.read('word/comments.xml')
        )
        comments_target_xml = etree.XML(
            docx_zip.read('word/document.xml')
        )

    comments_body = comments_body_xml.xpath('//w:comment', namespaces = WORD_NAMESPACE)
    comments_target = comments_target_xml.xpath('//w:commentRangeStart', namespaces = WORD_NAMESPACE)

    assert len(comments_body) == len(comments_target), 'Number of comments body elements is not equal to the number of comments target, which is unexpected'

    for c in comments_body:
        comment_body = c.xpath('string(.)', namespaces = WORD_NAMESPACE)
        comment_id = c.xpath('@w:id', namespaces = WORD_NAMESPACE)[0]

        comments[comment_id] = Comment(
            id = comment_id,
            body = CommentBody(
                comment_body
            )
        )

    for comment_target in comments_target:
        comment_id = comment_target.xpath('@w:id', namespaces = WORD_NAMESPACE)[0]
        comment = comments[comment_id]

        reference_tag = '}tbl' if comment.body.is_table else '}p'

        while not comment_target.tag.endswith(reference_tag):
            comment_target = comment_target.getparent()

        comment.target = comment_target

    return comments, comments_target_xml


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
