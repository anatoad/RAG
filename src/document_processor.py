from abc import ABC, abstractmethod
from pathlib import Path
import spacy
from spacy import Language
from transformers import AutoTokenizer
from logging import Logger
from urllib.parse import unquote
import settings
from utils import get_logger
import os
import re
import json
from document_helpers import *

class DocumentProcessor(ABC):
    def __init__(
        self, 
        path: str | Path,
        url: str,
        nlp: Language = None,
        tokenizer: AutoTokenizer = None,
        max_tokens: int = 512,
        logger: Logger = None
    ) -> None:
        self.path = path if isinstance(path, str) else path.as_posix()
        # decode percent-encoded/URL-encoded filename -> get diacritics
        self.filename = unquote(Path(self.path).name)
        self.url = url
        self.nlp = nlp or spacy.load(settings.SPACY_MODEL)
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(settings.MODEL_NAME)
        self.max_tokens = max_tokens or tokenizer.model_max_length
        self.sentences = None
        self.chunks = []
        self.num_chunks = 0
        self._logger = logger or get_logger(Path(__file__).resolve().stem)

    @abstractmethod
    def process(self) -> None:
        """
        Implementation of document processing logic.
        This should include partitioning, cleanup and chunking.
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        pass

    @abstractmethod
    def perform_chunking(self) -> None:
        pass

    def format_data(self, index_name: str) -> list[dict[str, str]]:
        """
        Format data for OpenSearch bulk ingestion.
        """
        if not self.chunks: return None

        return [
            {"_index": index_name, "_id": chunk["id"]} | chunk
            for chunk in self.chunks
        ]
    
    def export_chunked_document(self, output_filepath: str | None = None):
        if not self.chunks:
            self._logger.error(f"Cannot export chunks to json. No chunks found. Please call process() first.")
            return

        if not output_filepath:
            dir_path = Path(self.path).parent
            filename = f"{Path(self.path).stem}.json"
            output_filepath = os.path.join(dir_path, filename)

        with open(output_filepath, "w+") as file:
            json.dump(self.chunks, file, indent=4, ensure_ascii=False)
        
        self._logger.info(f"Saved chunked document to {output_filepath}")

    def _sentencize(self, text: str, page_number: int | None = None) -> list[str]:
        """
        Perform sentence segmentation using spaCy model trained for Romanian language.
        https://spacy.io/models/ro#ro_core_news_lg
        """
        sentences = []
        text = self._cleanup_text(text)

        try:
            doc = self.nlp(text)
            sents = [str(sentence) for sentence in doc.sents]
            sentences = self._format_sentences(sents, page_number=page_number)
        except Exception as e:
            self._logger.error(f"An unexpected error occurred: {str(e)}", exc_info=True)

        return sentences

    def _split_sentences_into_chunks(
        self,
        sentences: list[dict[str, str | int]],
        table_id: str | None = None,
        table_text: str | None = None,
    ) -> None:
        """
        Split text into chunks of sentences, taking into account the maximum sequence length of
        the model (max_tokens). This is the context window for embedding purposes - any text
        exceeding that will get truncated, the information will be lost.
        """
        if not sentences: return []
        chunks = []
        current_chunk = []
        current_tokens_count = 0
        current_page = sentences[0]["page_number"]

        for sentence in sentences:
            sentence_tokens = self.tokenizer.encode(sentence["text"], add_special_tokens=False)
            sentence_tokens_count = len(sentence_tokens)
    
            # save current chunk
            if current_tokens_count + sentence_tokens_count > self.max_tokens:
                self.num_chunks += 1
                chunks.append(
                    self._format_chunk(
                        sentences=current_chunk,
                        chunk_number=self.num_chunks,
                        page_number=current_page,
                        table_id=table_id,
                        table_text=table_text,
                    )
                )
                current_chunk = []
                current_tokens_count = 0
                current_page = sentence["page_number"]
            
            # split current sentence if it is too long
            if sentence_tokens_count > self.max_tokens:
                tokens = self.tokenizer.tokenize(sentence["text"])
                for i in range(0, len(tokens), self.max_tokens):
                    substring = self.tokenizer.convert_tokens_to_string(tokens[i:i+self.max_tokens])
                    self.num_chunks += 1
                    chunks.append(
                        self._format_chunk(
                            sentences=[substring],
                            chunk_number=self.num_chunks,
                            page_number=current_page,
                            table_id=table_id,
                            table_text=table_text,
                        )
                    )
                continue

            current_chunk.append(sentence["text"])
            current_tokens_count += sentence_tokens_count

        if current_chunk:
            self.num_chunks += 1
            chunks.append(
                self._format_chunk(
                    sentences=current_chunk,
                    chunk_number=self.num_chunks,
                    page_number=current_page,
                    table_id=table_id,
                    table_text=table_text
                )
            )
        
        return chunks

    def _cleanup_text(self, text: str) -> str:
        # remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # clean up table of contents
        text = re.sub(r'\.{4,}\s*\d+', '', text).strip()

        return text

    def _format_sentences(self, sentences: list[str], page_number: int) -> list[dict[str, str | int]]:
        return [
            {
                "text": sentence,
                "page_number": page_number
            }
            for sentence in sentences
        ]

    def _format_chunk(
        self,
        sentences: list[str],
        chunk_number: int,
        page_number: int,
        table_id: str | None = None,
        table_text: str | None = None,
    ) -> dict[str, str | int]:
        return {
            "id": get_id(self.url, chunk_number),
            "text": " ".join(sentences),
            "url": self.url,
            "type": self.type,
            "filename": self.filename,
            "page_number": page_number,
            "table_id": table_id,
            "table_text": table_text,
        }
