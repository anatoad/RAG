from pathlib import Path
from spacy import Language
from transformers import AutoTokenizer
from logging import Logger
from document_processor import DocumentProcessor

class TxtProcessor(DocumentProcessor):
    def __init__(
        self, 
        path: str | Path,
        url: str,
        nlp: Language = None,
        tokenizer: AutoTokenizer = None,
        max_tokens: int = 512,
        logger: Logger = None
    ) -> None:
        super().__init__(path, url, nlp, tokenizer, max_tokens, logger)
        self.type = "txt"
        self.text = None
    
    def _read_file(self) -> str:
        try:
            with open(self.path, "r") as file:
                self.text = ' '.join(file.readlines())
        except Exception as e:
            self._logger.error(f"Error occured trying to read file", exc_info=True)

    def cleanup(self) -> None:
        self.text = self._cleanup_text(self.text)

    def perform_chunking(self):
        sentences = self._sentencize(self.text)
        self.chunks = self._split_sentences_into_chunks(sentences)

    def process(self) -> None:
        self._read_file()
        self.cleanup()
        self.perform_chunking()
