from enum import Enum

from .FlatEmbedder import FlatEmbedder
from .StructuredEmbedder import StructuredEmbedder
from .BaseModel import BaseModel


class EmbedderType(Enum):
    FLAT = 'flat'
    STRUCTURED = 'structured'
