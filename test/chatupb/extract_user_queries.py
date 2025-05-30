import json
import os

DIR = "chatupb-convs/"

def load_conversation(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_user_queries(conversation):
    messages = conversation.get("messages", [])
    return [msg["content"] for msg in messages if msg["role"] == "user"]

if __name__ == "__main__":
    files = [file for file in os.listdir(DIR) if file.endswith(".json")]
    queries = []
    for file in files:
        conversation = load_conversation(os.path.join(DIR, file))
        queries.extend(extract_user_queries(conversation))
    
    with open("user_queries.json", "w", encoding="utf-8") as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)
