from pathlib import Path
import os

# OpenSearch configuration
OPENSEARCH_ADDRESS = "localhost"
ADMIN_PASSWD = os.environ['OPENSEARCH_INITIAL_ADMIN_PASSWORD']

MODEL_GROUP_NAME = "Unstructured"
PIPELINE_NAME = "ingest-pipeline"
INDEX_NAME = "knn-index"
MODEL_URL = "huggingface/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_ID = "g0Krd5UB_e6dONcEC5dk"

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANKER_ENDPOINT = "http://localhost:12456/score"

SPACY_MODEL = "ro_core_news_lg"

OPENAI_MODEL = "gpt-4o-mini"
TEMPERATURE = 1e-8

# Define paths dynamically relative to this file
SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
OPENSEARCH_CONFIG_DIR = BASE_DIR / "opensearch-config"
CONTENT_DIR = DATA_DIR / "content"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
CONTENT_DIR.mkdir(parents=True, exist_ok=True)
