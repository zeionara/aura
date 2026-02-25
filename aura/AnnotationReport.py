from dataclasses import dataclass


@dataclass
class AnnotationReport:
    document: str
    n_tables: int
    n_paragraphs: int
    n_generated_labels: int
    n_missing_labels: int
    complete: bool = None
