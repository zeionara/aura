from dataclasses import dataclass


@dataclass
class AnnotationReport:
    document: str
    n_tables: int
    n_paragraphs: int
