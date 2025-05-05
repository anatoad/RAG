import json
import logging
import settings
import utils
from langchain_core.documents import Document
from opensearch_client import OpenSearchClient
import boto3
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

class Retriever:
    def __init__(
        self,
        logger: logging.Logger | None = None,
        index: str = settings.INDEX_NAME,
        model_id: str = settings.MODEL_ID,
        k: int = 15,
        top_k: int = 10
    ) -> None:
        """
        Initialize the Retriever with OpenSearch connection details.
        """
        self._logger = logger or utils.get_logger(__name__)
        self._k = k
        self._top_k = top_k
        self._index = index
        self._model_id = model_id
        self._client = OpenSearchClient(logger=self._logger)
        self._client._connect_to_opensearch()
        self._predictor = Predictor(
            endpoint_name=settings.SAGEMAKER_RERANKER_ENDPOINT,
            serializer=JSONSerializer(),
            deserializer=JSONDeserializer(),
        )
        self._wrapper = utils.get_text_wrapper()
        self._SCORE_THRESHOLD = 0.07

    def retrieve_documents(self, query_text: str, k: int = None) -> list[Document]:
        """
        Retrieve documents relevant to the query_text using the OpenSearch client.
        """
        documents = self._client.semantic_search(
            index_name=self._index,
            query_text=query_text,
            k=k or self._k
        )

        # Handle tables
        retrieved_documents = [
            document 
            for document in documents
            if document["_score"] >= self._SCORE_THRESHOLD
        ]

        # Sort document chunks - group by document, maintain order
        retrieved_documents = sorted(retrieved_documents, key=lambda doc: doc["_id"])

        table_ids = set()
        documents = []

        for document in retrieved_documents:
            table_id = document["_source"]["table_id"]
            if table_id and table_id in table_ids: # table can be split into multiple chunks
                continue

            if not table_id:
                page_content = document["_source"]["text"]
            elif table_id not in table_ids:
                table_ids.add(table_id)
                page_content = document["_source"]["table_text"]

            documents.append(
                Document(
                    page_content=page_content,
                    metadata={
                        "score": document["_score"],
                    } | {
                        key: value
                        for key, value in document["_source"].items() if "embedding" not in key
                    }
                )
            )

        return documents

    def rerank(self, query: str, documents: list[Document], top_k: int | None = None) -> list[Document]:
        docs = [doc.page_content for doc in documents]
        payload = {
            "inputs": [
                {"text": query, "text_pair": doc}
                for doc in docs
            ]
        }
        results = self._predictor.predict(payload)
        scores = [item["score"] for item in results]
        for doc, score in list(zip(documents, scores)):
            doc.metadata["rerank_score"] = score
        # keep only top_k documents with the highest score, keep order unchanged
        top_k = top_k or self._top_k
        best_idx = sorted(
            sorted(
                range(len(documents)),
                key=lambda i: documents[i].metadata["rerank_score"],
                reverse=True
            )[:top_k]
        )
        return [documents[i] for i in best_idx]

    def format_document(self, document: Document) -> str:
        source = document.metadata.get('url')
        page_number = document.metadata.get("page_number")
        content = document.page_content

        doc_str = f"Sursa: {source}"
        if page_number:
            doc_str += f", pagina: {page_number}"
        doc_str += f"\nConținut: {content}"

        return doc_str

    def format_documents(self, documents: list[Document]) -> str:
        """
        Format the retrieved documents into a string for inclusion in the prompt.
        """
        return "\n\n".join(
            [
            self.format_document(doc)
            for doc in documents
            ]
        )

    def _print_document(self, document: Document) -> None:
        print(self._wrapper.fill(
            json.dumps(
                {
                    "page_content": document.page_content.replace("\n", '\n'),
                    "metadata": document.metadata
                },
                indent=4,
                ensure_ascii=False,
                separators=(',', ":"),
            )
        ))
    
    def print_documents(self, documents: list[Document]) -> None:
        docs = sorted(documents, key=lambda doc: doc.metadata["score"], reverse=True)
        for document in docs:
            self._print_document(document)
