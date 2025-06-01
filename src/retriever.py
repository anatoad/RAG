import json
import logging
import settings
import utils
from langchain_core.documents import Document
from opensearch_client import OpenSearchClient

class Retriever:
    def __init__(
        self,
        logger: logging.Logger | None = None,
        index: str = settings.INDEX_NAME,
        model_id: str = settings.MODEL_ID,
        k: int = 15
    ) -> None:
        """
        Initialize the Retriever with OpenSearch connection details.
        """
        self._logger = logger or utils.get_logger(__name__)
        self._k = k
        self._index = index
        self._model_id = model_id
        self._client = OpenSearchClient(logger=self._logger)
        self._client._connect_to_opensearch()
        self._wrapper = utils.get_text_wrapper()
        self._SCORE_THRESHOLD = 0.07

    def retrieve_documents(self, query_text: str, k: int = None) -> list[Document]:
        """
        Retrieve documents relevant to the query_text using the OpenSearch client.
        """
        retrieved_documents = self._client.semantic_search(
            index_name=self._index,
            query_text=query_text,
            k=k or self._k
        )
        retrieved_documents = sorted(retrieved_documents, key=lambda doc: doc["_score"], reverse=True)
        documents = []
        for document in retrieved_documents:
            if not document["_source"].get("table_id"):
                page_content = document["_source"]["text"]
            else:
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

    def format_document(self, document: Document) -> str:
        source = document.metadata.get('url')
        page_number = document.metadata.get("page_number")
        content = document.page_content

        doc_str = f"Sursa: {source}"
        if page_number:
            doc_str += f", pagina: {page_number}"
        doc_str += f"\nConținut: {content}"

        return doc_str

    def filter_documents(self, documents: list[Document]) -> list[Document]:
        # Sort document chunks - group by document, maintain order
        ordered_docs = sorted(documents, key=lambda doc: doc.metadata["id"])

        # Filter out documents with the same table_id
        table_ids = set()
        filtered_docs = []

        for document in ordered_docs:
            table_id = document.metadata.get("table_id")
            if table_id and table_id in table_ids: # table can be split into multiple chunks
                continue

            filtered_docs.append(document)
            table_ids.add(table_id)

        documents = list(filter(lambda doc: doc.metadata.get("score", 0) >= self._SCORE_THRESHOLD, filtered_docs))

        return documents

    def format_documents(self, documents: list[Document]) -> str:
        """
        Format the retrieved documents into a string for inclusion in the prompt.
        """

        filtered_docs = self.filter_documents(documents)

        return "\n\n".join(
            [
            self.format_document(doc)
            for doc in filtered_docs
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
