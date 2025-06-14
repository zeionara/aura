from zipfile import ZipFile

from lxml import etree

from .xml import WORD_NAMESPACE

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
