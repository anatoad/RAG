from langchain_core.documents import Document
import json
import requests
from logging import Logger
from settings import RERANKER_ENDPOINT, RERANKER_MODEL
import utils

class Reranker:
    def __init__(
        self,
        logger: Logger | None = None,
        url: str = RERANKER_ENDPOINT,
        model: str = RERANKER_MODEL
    ) -> None:
        self._logger = logger or utils.get_logger(__name__)
        self._wrapper = utils.get_text_wrapper()
        self._url = url
        self._model = model
        self._headers = {
            "Content-Type": "application/json"
        }
    
    def rerank_documents(self, question: str, documents: list[Document]) -> list:
        """
        Rerank documents based on the question and the documents' content.
        """
        passages = [doc.page_content for doc in documents]
        sentence_pairs = [
            [question, passage]
            for passage in passages
        ]

        # send the request to the reranker
        request_body = {
            "sentence_pairs": sentence_pairs,
            "normalize": True,
        }

        response = requests.post(
            url=self._url,
            headers=self._headers,
            data=json.dumps(request_body)
        )
        
        if response.status_code != 200:
            raise Exception(f"Reranker request failed: {response.text}")

        results = response.json()
        
        # update document scores
        for doc, score in zip(documents, results["scores"]):
            doc.metadata["rerank_score"] = score

        # the id of a document has the format <hash>-<chunk_id>
        # sort by rerank score, then by id, so as to keep the order of chunks from the same document
        sorted_documents = sorted(
            documents,
            key=lambda x: (x.metadata["rerank_score"], x.metadata["id"].split("-")[1]),
            reverse=True
        )

        return sorted_documents
