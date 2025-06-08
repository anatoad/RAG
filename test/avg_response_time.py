import json

filepath = {
    "gpt-4o-mini": "/home/ana/ACS/rag/test/logs/gpt-4o-mini/qa.json",
    "deepseek": "/home/ana/ACS/rag/test/logs/deepseek/qa.json",
    "gemma-3n-e4b-it": "/home/ana/ACS/rag/test/logs/gemma-3n-e4b-it/qa.json"
}


def average_response_time(filepath: str) -> float:
    with open(filepath, "r", encoding="utf-8") as file:
        data = json.load(file)

    response_times = [entry["response_time"] for entry in data if "response_time" in entry]

    if not response_times:
        return 0.0

    return sum(response_times) / len(response_times)

for model_name, filepath in filepath.items():
    print(f"[{model_name}]: {average_response_time(filepath)}")
