from .zip import get_comments, read_elements
from .xml import get_text, get_elements, get_paragraph_style, get_xml, get_condensed_xml, WORD_NAMESPACE, WORD_NAMESPACES
from .string import normalize_spaces, drop_space_around_punctuation, read, dict_to_string, string_to_dict, replace_last_occurrence
from .prompt import make_system_prompt, make_annotation_prompt
from .file import dict_to_json_file
