from pathlib import Path
import json
import os
import sys
import logging
import settings
from textwrap import TextWrapper
import pandas as pd

def get_logger(
    name,
    log_dir=None,
    level=logging.INFO,
    format=f"%(asctime)s %(levelname)s: %(message)s",
    stderr=False
) -> logging.Logger:
    log_dir = log_dir or settings.LOGS_DIR
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        fileHandler = logging.FileHandler(f'{log_dir}/{name}.log')
        fileHandler.setFormatter(logging.Formatter(format))
        logger.addHandler(fileHandler)

        if stderr:
            streamHandler = logging.StreamHandler(sys.stderr)
            streamHandler.setFormatter(logging.Formatter(format))
            logger.addHandler(streamHandler)

    return logger

def get_text_wrapper():
    wrapper = TextWrapper()
    wrapper.width = 160
    wrapper.replace_whitespace = False
    wrapper.break_long_words = False
    wrapper.break_on_hyphens = False
    return wrapper

def get_directories(path: str) -> list[str]:
    return sorted([d.name for d in Path(path).iterdir() if d.is_dir()])

def get_files_by_extension(path: str, extension: str) -> list[str]:
    return sorted([f.name for f in Path(path).glob(f"*{extension}")])

def read_metadata(metadata_filepath: str) -> dict:
    with open(metadata_filepath, "r") as jsonFile:
        try:
            data = json.load(jsonFile)
        except:
            data = {}

    return data

def write_to_json_file(filepath: str, data: dict) -> None:
    with open(filepath, "w") as jsonFile:
        json.dump(data, jsonFile, indent=4, sort_keys=True)

