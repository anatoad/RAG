from pathlib import Path
import os

# OpenSearch configuration
ADMIN_PASSWD = os.environ['OPENSEARCH_INITIAL_ADMIN_PASSWORD']
INDEX_NAME = 'rag-knn-index'
MODEL_ID = 'g0Krd5UB_e6dONcEC5dk'

# Get the absolute path of the directory where this file is located
SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Define paths dynamically relative to this file
DATA_DIR = BASE_DIR / "data"
OCR_DIR = BASE_DIR / "data" / "_ocr"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
OCR_DIR.mkdir(parents=True, exist_ok=True)

# Print paths for debugging
if __name__ == "__main__":
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"src: {SRC_DIR}")
    print(f"data: {DATA_DIR}")
    print(f"ocr: {OCR_DIR}")
