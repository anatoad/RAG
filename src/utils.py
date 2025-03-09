from pathlib import Path
import json
import hashlib

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

def get_hash(string: str) -> int:
    return hashlib.sha256(string.encode('utf-8')).hexdigest()

def get_id(url: str, chunk_number: int) -> str:
    hash = get_hash(url)
    return f'{hash}-{chunk_number}'
