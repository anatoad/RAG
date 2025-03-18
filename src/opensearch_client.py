import json
from logging import Logger
from utils import get_logger
from config import ADMIN_PASSWD, INDEX_NAME, MODEL_ID
from opensearchpy import OpenSearch

def replace_placeholders(object: dict, **kwargs):
    if isinstance(object, dict):
        return {key: replace_placeholders(val, **kwargs) for key, val in object.items()}
    elif isinstance(object, list):
        return [replace_placeholders(item, **kwargs) for item in object]
    elif isinstance(object, str):
        return object.format(**kwargs)
    return object

class OpenSearchClient:
    def __init__(self, host: str = 'localhost', port: int = 9200, logger: Logger = None) -> None:
        self.host = host
        self.port = port
        self._logger = logger or get_logger("opensearch-client")
        self.client = self._connect_to_opensearch()

    def _connect_to_opensearch(self) -> None:
        if hasattr(self, 'client') and self.client:
            return self.client
        try:
            # Create the client with SSL/TLS and hostname verification disabled.
            client = OpenSearch(
                hosts = [{'host': self.host, 'port': self.port}],
                http_compress = True, # enables gzip compression for request bodies
                http_auth = ('admin', ADMIN_PASSWD),
                use_ssl = True,
                verify_certs = False,
                ssl_assert_hostname = False,
                ssl_show_warn = False,
            )
            self._logger.info(f"Connected to OpenSearch")
            self._logger.info(json.dumps(client.info(), indent=4))
            return client
        except Exception as e:
            self._logger.error(f"Could not connect to Opensearch: {e}")
            return None

    def _perform_request(self, method: str, endpoint: str, body: dict):
        try:
            response = self.client.transport.perform_request(method, endpoint, body=body)
            self._logger.info(f"{method} {endpoint}")
            if body: self._logger.info(json.dumps(body, indent=4, ensure_ascii=False))
            self._logger.info(f"RESPONSE:\n{json.dumps(response, indent=4, ensure_ascii=False)}")
            return response
        except Exception as e:
            self._logger.error("Error during request", exc_info=True)
            return None
        
    def _update_cluster_settings(self, settings: json):
        self._logger.info(f"Update ML-related cluster settings")
        endpoint = "_cluster/settings"
        return self._perform_request("PUT", endpoint, body=settings)
    
    def _load_json_config(self, filepath: str, **kwargs):
        with open(filepath, 'r') as file:
            json_config = json.load(file)

        json_config = replace_placeholders(json_config, **kwargs)
        return json_config

    def get_task(self, task_id: str):
        self._logger.info(f"Get task, task_id={task_id}")
        endpoint = f"/_plugins/_ml/tasks/{task_id}"
        return self._perform_request("GET", endpoint, body={})

    def register_model_group(self, group_name: str, description: str = "", access_mode: str = "public"):
        self._logger.info("Register model group")
        endpoint = f"/_plugins/_ml/model_groups/_register"
        body = {
            "name": group_name,
            "description": description,
            "access_mode": access_mode,
        }
        return self._perform_request("POST", endpoint, body)

    def get_model_groups(self):
        self._logger.info("Get all model groups")
        endpoint = "/_plugins/_ml/model_groups/_search"
        return self._perform_request("POST", endpoint, body={})
    
    def get_model_group_id(self, group_name: str):
        self._logger.info(f"Get model group id, group_name={group_name}")
        endpoint = "/_plugins/_ml/model_groups/_search"
        body = {
            "query": {
                "bool": {
                    "must": [
                        {
                        "terms": {
                            "name": [group_name]
                        }
                        }
                    ]
                }
            }
        }
        response = self._perform_request("GET", endpoint, body=body)
        if response["hits"]["hits"]:
            return response["hits"]["hits"][0]["_id"]
        return None

    def delete_model_group(self, group_name: str):
        group_id = self.get_model_group_id(group_name)
        if group_id:
            self._logger.info(f"Delete model group, group_name={group_name}")
            endpoint = f"/_plugins/_ml/model_groups/{group_id}"
            return self._perform_request("DELETE", endpoint, body={})
        return None

    def register_model(self, model_name: str, version: str, group_name: str):
        self._logger.info(f"Register model, model_name={model_name}, group_name={group_name}")
        group_id = self.get_model_group_id(group_name)
        if group_id:
            endpoint = "/_plugins/_ml/models/_register"
            body = {
                "name": model_name,
                "version": version,
                "model_group_id": group_id,
                "model_format": "TORCH_SCRIPT"
            }
            return self._perform_request("POST", endpoint, body=body)
        self._logger(f"Model group {group_name} not found")
        return None
    
    def get_model_id(self, task_id: str):
        response = self.get_task(task_id)
        return response["model_id"]
    
    def get_models(self):
        endpoint = f"/_plugins/_ml/models/_search"
        body = {
            "query": {
                "match_all": {}
            },
            "size": 1000
        }
        return self._perform_request("POST", endpoint, body=body)

    def deploy_model(self, model_id: str):
        self._logger.info(f"Deploy model, model_id = {model_id}")
        endpoint = f"/_plugins/_ml/models/{model_id}/_deploy"
        response = self.client.transport.perform_request("POST", endpoint)
        return response

    def check_index_exists(self, index_name: str):
        self._logger.info(f"Check if index exists, index_name = {index_name}")
        return self.client.indices.exists(index=index_name)

    def create_ingest_pipeline(self, pipeline_id: str, description:str, processors: list[dict]):
        endpoint = f"_ingest/pipeline/{pipeline_id}"
        body = {
            "description": description,
            "processors": processors
        }
        response = client.transport.perform_request("PUT", endpoint, body=body)
        return response

    def get_elements_count(self, index_name: str):
        self._logger.info(f"Get elements count, index_name = {index_name}")
        response = self.client.count(index=index_name)
        return response['count']

    def get_all_elements(self, index_name: str):
        self._logger.info(f"Get all elements, index_name = {index_name}")
        query = {"query": {"match_all": {}}}
        return self.client.search(
            index=index_name,
            body=query,
            _source_excludes=["embedding"]  # exclude text embedding from the response
        )

    def semantic_search(self, index_name: str, query_text: str, k: int = 3):
        self._logger.info(f"Semantic search, query_text = {query_text}")
        query = {
            "size": k,
            "query": {
                "neural": {
                        "embedding": {
                            "query_text": query_text,
                            "model_id": MODEL_ID,
                            "k": k
                        }
                    }
            }
        }
        return self.client.search(
            index=index_name,
            body=query,
            _source_excludes=["embedding"],
        )
    
    def create_index(self, index_name: str, body: json):
        self._logger.info(f"Create KNN index, index_name = {index_name}")
        return self.client.search(
            index=index_name,
            body=body,
            _source_excludes=["embedding"]
        )

if __name__ == "__main__":
    client = OpenSearchClient()

    # client.register_model_group("dummy")

    # response = client.register_model("huggingface/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "1.0.1", "dummy")
    # task_id = response["task_id"]
    # task_id = "gNEpqZUBJq99Lce6HLWB"
    # # client.get_task(task_id)

    # model_id = client.get_model_id(task_id)clus
    # client.deploy_model(model_id)

    # print(client.check_index_exists(INDEX_NAME))
    # print(client.get_elements_count(INDEX_NAME))
    # print(client.get_all_elements(INDEX_NAME))
    # print(json.dumps(client.semantic_search(INDEX_NAME, "regulamentul de acordare al burselor pentru studenti"), indent=4))

    # client.delete_model_group("dummy")
    # client.get_models()
    # print(json.dumps(client.get_model_groups(), indent=4))

    # client._load_json_config("../opensearch-config/ingest-pipeline.json", model_id=MODEL_ID)