from typing import ClassVar

from dataclasses import dataclass


INDENT = 2
INCLUDE_XML = False


@dataclass
class Item:
    type_label: ClassVar[str] = None

    content: str | list[str]
