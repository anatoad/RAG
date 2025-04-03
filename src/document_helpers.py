from enum import Enum
import hashlib
import pandas as pd
from io import StringIO
from unstructured.documents.elements import Element, ElementType, Table
import re
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

diacritics = {
    'ţ': 'ț',
}

def normalize_romanian_diacritics(text):
    normalized_text = text
    if isinstance(text, str):
        for char, value in diacritics.items():
            normalized_text = normalized_text.replace(char, value)
    return normalized_text

def convert_table_to_dataframe(table: Table) -> pd.DataFrame:
    df = pd.read_html(StringIO(table.metadata.text_as_html), encoding='utf-8')[0]
    # normalize diacritics for every string in the dataframe
    df.columns = [normalize_romanian_diacritics(col) for col in df.columns]
    df = df.map(lambda x: normalize_romanian_diacritics(x))

    # cleanup missing values
    df.columns = ['' if "Unnamed" in str(col) else str(col) for col in df.columns]
    df = df.fillna('')

    return df

def is_page_number(element: Element) -> bool:
    return (element.category == ElementType.PAGE_NUMBER 
            or (element.category == ElementType.UNCATEGORIZED_TEXT and element.text == str(element.metadata.page_number)))

def table_contains_html(table: Table) -> bool:
    pattern = r">([^<]+)<"
    matches = re.findall(pattern, table.metadata.text_as_html)
    return matches and any(match.strip() for match in matches)
