import json
import logging
import settings
from langchain_core.documents import Document
from opensearch_client import OpenSearchClient

class Retriever:
    def __init__(
        self,
        logger: logging.Logger,
        index: str = settings.INDEX_NAME,
        model_id: str = settings.MODEL_ID,
        k: int = 10,
    ) -> None:
        """
        Initialize the Retriever with OpenSearch connection details.
        """
        self._logger = logger
        self._k = k
        self._index = index
        self._model_id = model_id
        self._client = OpenSearchClient(logger=self._logger)
        self._client._connect_to_opensearch()

    def retrieve_documents(self, query_text: str, k: int = None) -> list[Document]:
        """
        Retrieve documents relevant to the query_text using the OpenSearch client.
        """
        response = self._client.semantic_search(
            index_name=self._index,
            query_text=query_text,
            k=k or self._k
        )

        documents = [
            Document(
                page_content=document["_source"]["text"],
                metadata={
                    "id": document["_source"]["id"],
                    "url": document["_source"]["url"],
                    "filename": document["_source"]["filename"],
                    "page_number": document["_source"]["page_number"],
                }
            )
            for document in response["hits"]["hits"]
        ]

        return documents

    def format_documents(self, documents: list[Document]) -> str:
        """
        Format the retrieved documents into a string for inclusion in the prompt.
        """
        return "\n\n".join(
            [
            f"Source: {doc.metadata.get('url')}\nContent: {doc.page_content}"
            for doc in documents
            ]
        )

    def _print_document(self, document: Document) -> None:
        print(json.dumps(
            {
                "page_content": document.page_content,
                "metadata": document.metadata
            },
            indent=4,
            ensure_ascii=False,
        ))
    
    def print_documents(self, documents: list[Document]) -> None:
        for document in documents:
            self._print_document(document)
