from pathlib import Path
import os

# OpenSearch configuration
ADMIN_PASSWD = os.environ['OPENSEARCH_INITIAL_ADMIN_PASSWORD']
INDEX_NAME = 'rag-knn-index'
MODEL_ID = 'g0Krd5UB_e6dONcEC5dk'

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SPACY_MODEL = "ro_core_news_lg"

OPENAI_MODEL = "gpt-4o-mini"

# Define paths dynamically relative to this file
SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

DATA_DIR = BASE_DIR / "data"
OPENSEARCH_CONFIG_DIR = BASE_DIR / "opensearch-config"
OCR_DIR = DATA_DIR / "_ocr"
TEMP_DIR = BASE_DIR / "temp"
CONTENT_DIR = DATA_DIR / "content"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
OCR_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)
CONTENT_DIR.mkdir(parents=True, exist_ok=True)
