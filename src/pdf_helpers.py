from enum import Enum
import hashlib
import pandas as pd
from io import StringIO
from unstructured.documents.elements import Element, ElementType, Table

class ElementCategory(Enum):
    TEXTUAL = "Textual"
    TABLE = "Table"
    IMAGE = "Image"
    OTHER = "Other"

def classify_element(element_type):
    categories = {
        ElementCategory.TEXTUAL: [
            ElementType.TITLE,
            ElementType.TEXT,
            ElementType.UNCATEGORIZED_TEXT,
            ElementType.NARRATIVE_TEXT,
            ElementType.BULLETED_TEXT,
            ElementType.PARAGRAPH,
            ElementType.ABSTRACT,
            ElementType.FIELD_NAME,
            ElementType.VALUE,
            ElementType.LINK,
            ElementType.COMPOSITE_ELEMENT,
            ElementType.FIGURE_CAPTION,
            ElementType.CAPTION,
            ElementType.LIST_ITEM,
            ElementType.LIST_ITEM_OTHER,
            ElementType.ADDRESS,
            ElementType.EMAIL_ADDRESS,
            ElementType.FORMULA,
            ElementType.HEADER,
            ElementType.HEADLINE,
            ElementType.SUB_HEADLINE,
            ElementType.PAGE_HEADER,
            ElementType.SECTION_HEADER,
            ElementType.PAGE_FOOTER,
        ],
        ElementCategory.IMAGE: [
            ElementType.IMAGE,
            ElementType.PICTURE,
        ],
        ElementCategory.TABLE: [
            ElementType.TABLE,
        ]
    }

    for category, options in categories.items():
        if element_type in options:
            return category
    
    return ElementCategory.OTHER

def get_hash(string: str) -> int:
    return hashlib.sha256(string.encode('utf-8')).hexdigest()

def get_id(url: str, chunk_number: int) -> str:
    hash = get_hash(url)
    return f'{hash}-{chunk_number}'

def get_table_id(table_markdown: str) -> str:
    return get_hash(table_markdown)

def convert_table_to_dataframe(table: Table) -> pd.DataFrame:
    return pd.read_html(StringIO(table.metadata.text_as_html))[0]

def convert_table_to_markdown(table: Table) -> str:
    return convert_table_to_dataframe(table).to_markdown(index=False)

def is_page_number(element: Element) -> bool:
    return (element.category == ElementType.PAGE_NUMBER 
            or (element.category == ElementType.UNCATEGORIZED_TEXT and element.text == str(element.metadata.page_number)))