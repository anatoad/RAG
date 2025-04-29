# import json
# import os

# DATA_DIR = "/home/ana/ACS/rag/data"

# dirs = os.listdir(DATA_DIR)

# for dir in dirs:
#     metadata_filepath = f"{DATA_DIR}/{dir}/metadata.json"

#     metadata = {}
#     with open(metadata_filepath, "r") as file:
#         metadata = json.load(file)

#     files = os.listdir(f"{DATA_DIR}/{dir}")
#     files = [file for file in files]

#     metadata = [item for item in metadata if item['path'] in files]

#     with open(metadata_filepath, "w") as file:
#         json.dump(metadata, file, indent=4)