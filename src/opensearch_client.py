import json
from logging import Logger
from utils import get_logger
import settings
from opensearchpy import OpenSearch, helpers
from opensearchpy.exceptions import TransportError
from jinja2 import Template
import time

class OpenSearchClient:
    def __init__(self, host: str = settings.OPENSEARCH_ADDRESS, port: int = 9200, logger: Logger = None) -> None:
        self.host = host
        self.port = port
        self._logger = logger or get_logger("opensearch-client")
        self.client: OpenSearch = self._connect_to_opensearch()

    def _connect_to_opensearch(self) -> None:
        if hasattr(self, 'client') and self.client:
            return self.client
        try:
            # Create the client with SSL/TLS and hostname verification disabled.
            client = OpenSearch(
                hosts = [{'host': self.host, 'port': self.port}],
                http_compress = True, # enables gzip compression for request bodies
                http_auth = ('admin', settings.ADMIN_PASSWD),
                use_ssl = True,
                verify_certs = False,
                ssl_assert_hostname = False,
                ssl_show_warn = False,
            )
            self._logger.info(f"Connected to OpenSearch")
            if client:
                self._logger.info(json.dumps(client.info(), indent=4))
            return client
        except Exception as e:
            self._logger.error(f"Could not connect to Opensearch: {e}")
            return None

    def _perform_request(self, method: str, endpoint: str, body: dict, params: dict | None = None, verbose: bool = True):
        try:
            response = self.client.transport.perform_request(method, endpoint, body=body, params=params)
            if verbose:
                self._logger.info(f"{method} {endpoint}")
                if body: self._logger.info(json.dumps(body, indent=4, ensure_ascii=False))
                self._logger.info(f"Response:\n{json.dumps(response, indent=4, ensure_ascii=False)}")
            return response
        except Exception as e:
            if verbose:
                self._logger.error("Error during request", exc_info=True)
            return None
        
    def _update_cluster_settings(self, settings: json):
        self._logger.info(f"Update ML-related cluster settings")
        endpoint = "/_cluster/settings"
        return self._perform_request("PUT", endpoint, body=settings)
    
    def _load_json_config(self, filepath: str, **kwargs):
        with open(filepath, 'r') as file:
            template = Template(file.read())
        json_config = template.render(**kwargs)
        return json_config

    def _wait_for_task_to_finish(self, task_id, timeout=60000, wait_time=5):
        """
        Waits for a task to complete, with retries.
        """
        start_time = time.time()
        
        while True:
            try:
                response = self.get_task(task_id)
                task_status = response.get('state')
                
                if task_status == 'COMPLETED':
                    self._logger.info(f"Task {task_id} completed successfully.")
                    return response
                else:
                    self._logger.info(f"Task {task_id} is still in progress. Waiting...")
                    
                    # Check if timeout has been reached
                    if time.time() - start_time > timeout:
                        self._logger.error(f"Task {task_id} did not complete within the timeout period of {timeout} seconds.")
                        break
                    
                    time.sleep(wait_time)
            except TransportError as e:
                self._logger.error(f"Error while checking task {task_id}: {e}")
                time.sleep(10)
    
    def _wait_for_model_to_register(self, model_name, group_name: str, timeout=60000, wait_time=5):
        """
        Waits for a task to complete, with retries.
        """
        start_time = time.time()
        
        while True:
            try:
                response = self.get_model(model_name, group_name, verbose=False)
                # self._logger.info(json.dumps(response, indent=4))
                task_status = response['_source'].get('model_state')
                
                if task_status == 'REGISTERED':
                    self._logger.info(f"Model registered successfully.")
                    return response
                else:
                    self._logger.info(f"Model is still registering. Waiting...")
                    
                    # Check if timeout has been reached
                    if time.time() - start_time > timeout:
                        self._logger.error(f"Model did not register within the timeout period of {timeout} seconds.")
                        break
                    
                    time.sleep(wait_time)
            except TransportError as e:
                self._logger.error(f"Error occured while waiting for model to register", exc_info=True)
                time.sleep(10)

    def get_task(self, task_id: str):
        self._logger.info(f"Get task, task_id={task_id}")
        endpoint = f"/_plugins/_ml/tasks/{task_id}"
        return self._perform_request("GET", endpoint, body={})

    def register_model_group(self, group_name: str, description: str = "", access_mode: str = "public"):
        self._logger.info("Register model group")
        model_group_id = self.get_model_group_id(group_name)
        if model_group_id:
            self._logger.info(f'Model group with name "{group_name}" already exists. Id: {model_group_id}')
            return model_group_id

        endpoint = f"/_plugins/_ml/model_groups/_register"
        body = {
            "name": group_name,
            "description": description,
            "access_mode": access_mode,
        }
        response = self._perform_request("POST", endpoint, body)
        if response:
            settings.MODEL_GROUP_ID = response.get("model_group_id")
            return response.get("model_group_id")
        return None

    def get_model_groups(self):
        self._logger.info("Get all model groups")
        endpoint = "/_plugins/_ml/model_groups/_search"
        response = self._perform_request("POST", endpoint, body={})
        if response:
            return response["hits"]["hits"]
        return []
    
    def get_model_group_id(self, group_name: str, verbose: bool = True):
        if verbose:
            self._logger.info(f"Get model group id, group_name={group_name}")
        endpoint = "/_plugins/_ml/model_groups/_search"
        body = {
            "query": {
                "match": {
                    "name": group_name
                }
            }
        }
        response = self._perform_request("GET", endpoint, body=body, verbose=verbose)
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

    def register_model(self, model_name: str, version: str, group_name: str) -> str:
        """
        Register model to the model group. Wait for task to finish. Return model_id.
        """
        self._logger.info(f"Register model, model_name={model_name}, group_name={group_name}")
        if self.get_model(model_name, group_name):
            self._logger.info(f"Model already exists in model group {group_name}")
            self._wait_for_model_to_register(model_name, group_name)
            return

        group_id = self.get_model_group_id(group_name)
        if group_id:
            endpoint = "/_plugins/_ml/models/_register"
            body = {
                "name": model_name,
                "version": version,
                "model_group_id": group_id,
                "model_format": "TORCH_SCRIPT"
            }
            response = self._perform_request("POST", endpoint, body=body)
            task_id = response["task_id"]
            response = self._wait_for_task_to_finish(task_id=task_id)
            settings.MODEL_ID = response.get("model_id")
            return response.get("model_id")
        self._logger(f"Model group {group_name} not found")

        return None
    
    def get_model(self, model_name: str, group_name: str, verbose: bool = True) -> str:
        model_group_id = self.get_model_group_id(group_name, verbose)
        if not model_group_id:
            if verbose: self._logger.error(f'No model group with name "{group_name}" found.')
            return False

        endpoint = "/_plugins/_ml/models/_search"
        body = {
            "query": {
                "bool": {
                "must": [
                    {
                    "match": { "name": model_name }
                    },
                    {
                    "match": { "model_group_id": model_group_id }
                    }
                ]
                }
            }
        }
        
        response = self._perform_request("POST", endpoint, body=body, verbose=verbose)
        if response:
            return response["hits"]["hits"][0]
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
        response = self._perform_request("POST", endpoint, body=body)
        if response:
            return response["hits"]["hits"]
        return response

    def deploy_model(self, model_id: str):
        self._logger.info(f"Deploy model, model_id = {model_id}")
        endpoint = f"/_plugins/_ml/models/{model_id}/_deploy"
        return self._perform_request("POST", endpoint, body={})

    def create_ingest_pipeline(
        self, 
        pipeline_id: str,
        description:str,
        model_id: str | None = None,
        processors: list[dict] | None = None,
    ):
        endpoint = f"/_ingest/pipeline/{pipeline_id}"
        if not processors:
            processors = [
                {
                    "text_embedding": {
                        "model_id": model_id,
                        "field_map": {
                            "text": "embedding"
                        }
                    }
                }
            ]
        body = {
            "description": description,
            "processors": processors
        }
        return self._perform_request("PUT", endpoint, body=body)

    def get_ingest_pipelines(self):
        endpoint = "/_ingest/pipeline"
        return self._perform_request("GET", endpoint, body={})
    
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
            _source_excludes=["embedding"],  # exclude text embedding from the response
            explain=True,
        )

    def semantic_search(self, index_name: str, query_text: str, k: int = 3, model_id: str = settings.MODEL_ID):
        self._logger.info(f"Semantic search, query_text = {query_text}")
        query = {
            "size": k,
            "query": {
                "neural": {
                        "embedding": {
                            "query_text": query_text,
                            "model_id": model_id,
                            "k": k
                        }
                    }
            }
        }
        try:
            response = self.client.search(
                index=index_name,
                body=query,
                _source_excludes=["embedding"],
            )
            self._logger.info("Semantic search performed successfully")
            return response["hits"]["hits"]
        except Exception as e:
            self._logger.error("Error occured during semantic search", exc_info=True)
            return None

    def index_exists(self, index_name: str):
        self._logger.info(f"Check if index exists, index_name = {index_name}")
        return self.client.indices.exists(index=index_name)

    def get_indices(self):
        self._logger.info(f"Get all indices")
        endpoint = "/_cat/indices"
        response = self._perform_request("GET", endpoint, body={})
        return [] if not response else response.split('\n')

    def create_index(self, index_name: str, body: dict | None = None):
        self._logger.info(f"Create KNN index, index_name = {index_name}")      
        endpoint = f"/{index_name}"

        # get default knn-index template config
        if not body:
            filepath = (settings.OPENSEARCH_CONFIG_DIR / "knn-index.json").as_posix()
            body = self._load_json_config(filepath, pipeline=settings.PIPELINE_NAME)

        return self._perform_request("PUT", endpoint, body=body)

    def get_documents_from_index(self, index_name: str) -> list[str]:
        self._logger.info(f"Get documents from index {index_name}")
        query = {"query": {"match_all": {}}}
        response = self.client.search(
            index=index_name,
            body=query,
            _source_includes=["filename"],
            explain=True
        )
        if response:
            docs = response["hits"]["hits"]
            doc_names = sorted(list(set([doc["_source"]["filename"] for doc in docs])))
            return doc_names

        return None

    def delete_index(self, index_name: str):
        self._logger.info(f"Delete index {index_name}")
        if self.check_index_exists(index_name):
            try:
                response = self.client.indices.delete(index=index_name)
                self._logger.info(f"Index {index_name} successfully deleted")
                return response
            except Exception as e:
                self._logger.error(f"Error deleting index {index_name}", exc_info=True)
        else:
            self._logger.info(f"Index {index_name} does not exist.")

    def ingest_data_bulk(self, data):
        self._logger.info("Ingest data bulk")
        try:
            ret = helpers.parallel_bulk(
                self.client, 
                actions=data, 
                chunk_size=10, 
                raise_on_error=True,
                raise_on_exception=False,
                # max_chunk_bytes=20 * 1024 * 1024,
                request_timeout=60
            )
            self._logger.info("Successfully performed parallel bulk ingestion")
            return ret
        except Exception as e:
            self._logger.error("Error during parallel bulk ingestion", exc_info=True)
            return None

    def delete_document(self, index: str, filename: str):
        self._logger.info(f"Delete document {filename}")
        try:
            body = {
                "query": {
                    "term": {
                        "filename": filename
                    }
                }
            }
            ret = self.client.delete_by_query(
                index=index,
                body=body
            )
            self._logger.info(f"Successfully deleted document {filename}", exc_info=True)
            return ret
        except Exception as e:
            self._logger.error(f"Error deleting document {filename}", exc_info=True)
            return None
