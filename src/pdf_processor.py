from spacy import Language
import re
import urllib
import ocrmypdf
import pymupdf
import settings
from utils import *
from pdf_helpers import *
import img2pdf
from pdf2image import convert_from_path
from transformers import AutoTokenizer
from bs4 import BeautifulSoup
os.environ["EXTRACT_TABLE_AS_CELLS"] = "True"
os.environ["TABLE_IMAGE_CROP_PAD"] = "10"
# os.environ["EXTRACT_IMAGE_BLOCK_CROP_VERTICAL_PAD"] = "0"
# os.environ["EXTRACT_IMAGE_BLOCK_CROP_HORIZONTAL_PAD"] = "0"
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.coordinates import PixelSpace
import spacy

class PdfProcessor:
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
        self.ocr_path = None
        self.bw_path = None
        # decode percent-encoded/URL-encoded filename -> get diacritics
        self.filename = urllib.parse.unquote(self.path.split('/')[-1])
        self.url = url
        self.nlp = nlp or spacy.load(settings.SPACY_MODEL)
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(settings.MODEL_NAME)
        self.max_tokens = max_tokens or tokenizer.model_max_length
        self.document = self._init_document(path)
        self.sentences = None
        self.chunks = []
        self.num_chunks = 0
        self.elements = None
        self._logger = logger or get_logger(Path(__file__).resolve().name.split(".")[0])

    def _init_document(self, path: str) -> None:
        return  pymupdf.open(path, filetype="pdf")
        
    def _needs_ocr(self) -> bool:
        return not any([page.get_text().strip() for page in self.document])

    def _perform_ocr(self, input_path: str, output_path: str, force_ocr: bool = False, redo_ocr: bool = False) -> None:
        ocrmypdf.ocr(
            input_file=input_path,
            output_file=output_path,
            output_type="pdf",
            language="ron",
            deskew=not(force_ocr or  redo_ocr),
            rotate_pages=True,
            progress_bar=True,
            force_ocr=force_ocr,
            redo_ocr=redo_ocr,
            jobs=4,
        )
        self._logger.info(f"Successfully performed OCR on {input_path}. Saved at {output_path}.")

    def apply_ocr(self, bw=False) -> None:
        """
        Apply OCR, create a new pdf, replace the path.
        """
        parent_dir = Path(self.path).parent.name
        if bw and self.bw_path:
            input_path = self.bw_path
            dir = settings.OCR_DIR / "bw" / parent_dir
        else:
            input_path = self.path
            dir = settings.OCR_DIR / parent_dir

        output_path = dir / self.filename
        os.makedirs(dir, exist_ok=True)

        if not os.path.exists(output_path):
            try:
                self._perform_ocr(input_path, output_path)
            except ocrmypdf.exceptions.TaggedPDFError:
                self._logger.info(f"This PDF is marked as a Tagged PDF. Using --redo-ocr to override this error.")
                self._perform_ocr(input_path, output_path, redo_ocr=True)
            except ocrmypdf.exceptions.PriorOcrFoundError:
                self._logger.info(f"This PDF contains pages that already have text. Using --force-ocr to completely re-run the OCR regardless of the existing text.")
                self._perform_ocr(input_path, output_path, force_ocr=True)
            except Exception as e:
                self._logger.error(f"An unexpected error occurred: {str(e)}", exc_info=True)
                return
        
        self.path = output_path
        self.document = self._init_document(self.path)
        self._logger.info(f"OCR for {self.path} found at {output_path}")

    def convert_to_bw(self, output_path: str = None, dpi: int = 400, threshold: int = 128) -> None:
        """
        Convert a PDF to black and white.
        """
        if not output_path:
            dir, filename = self.path.split("/")[-2:]
            output_path = settings.DATA_DIR / "bw" / dir / filename

        pages = convert_from_path(self.path, dpi=dpi, thread_count=4)
        processed_image_paths = []

        for i, page in enumerate(pages):
            # apply threshold to get a binary image
            gray_image = page.convert("L")
            bw_image = gray_image.point(lambda x: 0 if x < threshold else 255, "1")
            
            # save temporary image file
            temp_image_path = f"temp_page_{i}.png"
            bw_image.save(temp_image_path, format="PNG")
            processed_image_paths.append(temp_image_path)
        
        # merge processed images into a single PDF
        with open(output_path, "wb") as f:
            f.write(img2pdf.convert(processed_image_paths))
        
        # clean up temporary image files
        for temp_path in processed_image_paths:
            os.remove(temp_path)
    
        self.bw_path = output_path
        self._logger.info(f"Black and white PDF saved to: {self.bw_path}")

    def sentencize(self) -> None:
        self.sentences = []
        
        for page_number, page in enumerate(self.document):
            text = self._cleanup_text(page.get_text())
            try:
                doc = self.nlp(text)
                sents = [str(sentence) for sentence in doc.sents]
                self.sentences.extend(self._format_sentences(sents, page_number + 1))
            except Exception as e:
                self._logger.error(f"An unexpected error occurred: {str(e)}", exc_info=True)

    def split_into_chunks(
        self,
        sentences: list[dict[str, str | int]],
        table_id: str | None = None,
        table_text: str | None = None,
    ) -> None:
        """
        Split text into chunks of sentences, taking into account the maximum sequence length of
        the model (max_tokens). This is the context window for embedding purposes - any text
        exceeding that will get truncated, the information will be lost.

        TODO: try chunking by title
        """
        if not sentences: return
        chunks = []
        current_chunk = []
        current_tokens_count = 0
        current_page = 1

        for sentence in sentences:
            sentence_tokens = self.tokenizer.encode(sentence["text"], add_special_tokens=False)
            sentence_tokens_count = len(sentence_tokens)
            if current_page is None:
                current_page = sentence["page_number"]

            # TODO: deal with this
            if sentence_tokens_count > self.max_tokens:
                self._logger.error(f"Token count exceeded: {sentence_tokens_count}")
                self._logger.error(f"pdf: {self.path}, page number {sentence["page_number"]}")
                self._logger.error(f"Sentence: {sentence["text"]}")

            if current_tokens_count + sentence_tokens_count > self.max_tokens:
                self.num_chunks += 1
                chunks.append(
                    self._format_chunk(
                        sentences=current_chunk,
                        chunk_number=self.num_chunks,
                        page_number=current_page
                    )
                )
                current_chunk = []
                current_tokens_count = 0
                current_page = sentence["page_number"]
            
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
    
    def process(self) -> None:
        self.partition_pdf()
        self.cleanup_elements()
        self.perform_chunking()

    def partition_pdf(self) -> None:
        """
        Parse the PDF into a list of elements using lanchain unstructured.
        Detect complex layout in the document using OCR-based and Transformer-based models.
        """
        output_dir = settings.CONTENT_DIR
        self.elements = partition_pdf(
            filename=self.path,
            url=None,                                              # run inference locally, must have unstructured[local-inference] installed
            languages=["ron"],                                     # use Romanian and English language packs for OCR
            infer_table_structure=True,                            # extract tables
            strategy="hi_res",
            extract_images_in_pdf=True,                            # mandatory to set as ``True``
            extract_image_block_types=["Image", "Table"],          # optional
            extract_image_block_to_payload=False,                  # optional
            extract_image_block_output_dir=output_dir,             # optional - only works when ``extract_image_block_to_payload=False`
            max_partition=None,
        )
        self._logger.info(f"Partitioned PDF {self.path}")

    def _sentence_segmentation(self, element: Element) -> list[str]:
        """
        Split text into sentences using spaCy.
        """
        sentences = []
        clean_text = self._cleanup_text(element.text)
        
        try:
            doc = self.nlp(clean_text)
            sents = [str(sentence) for sentence in doc.sents]
            sentences = self._format_sentences(sents, page_number=element.metadata.page_number)
        except Exception as e:
            self._logger.error(f"An unexpected error occurred: {str(e)}", exc_info=True)

        return sentences

    def perform_chunking(self) -> None:
        """
        Perform chunking using the unstructured document elements obtained after partitioning.
        Combines consecutive elements to form chunks as large as possible without exceeding the
        maximum sequence length of the embedding model (max_tokens). 

        A single element that exceeds the maximum chunk size is divided into two or more chunks using
        sentence segmentation.
        """
        if not self.elements:
            return None

        sentences = []
        for element in self.elements:
            element_category = classify_element(element.category)

            if element_category == ElementCategory.TEXTUAL:
                sentences.extend(self._sentence_segmentation(element))

            elif element_category == ElementCategory.TABLE:
                self.chunks.extend(self.split_into_chunks(sentences))
                sentences = []
                self.chunks.extend(self._get_table_chunks(element))

        # use the sentences to build text chunks
        self.chunks.extend(self.split_into_chunks(sentences))

    def _format_table_row(self, row) -> str:
        return " ; ".join(row) + " ; "

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
            "type": "pdf",
            "filename": self.filename,
            "page_number": page_number,
            "table_id": table_id,
            "table_text": table_text,
        }

    def format_data(self, index_name: str) -> list[dict[str, str]]:
        """
        Format data for OpenSearch bulk ingestion.
        """
        return [
            {"_index": index_name, "_id": chunk["id"]} | chunk
            for chunk in self.chunks
        ]

    def _get_table_chunks(self, table) -> str:
        df = convert_table_to_dataframe(table)

        # get table header + rows
        records = []
        records.append(self._format_table_row(df.columns))
        records.extend([self._format_table_row(row) for _, row in df.iterrows()])

        sentences = self._format_sentences(records, page_number=table.metadata.page_number)

        table_text = self._format_table_text(table)
        table_id = get_table_id(table_text)
        table_chunks = self.split_into_chunks(sentences, table_id, table_text)
        
        return table_chunks
    
    def _format_table_text(self, table: Table) -> str:
        """
        
        """
        markdown = convert_table_to_dataframe(table).to_markdown(index=False)
        # remove whitespace
        markdown = re.sub(r' +', ' ', markdown)
        markdown = re.sub(r'\|:[-]{3,}', '|:--', markdown)

        return markdown

    def _plot_page_with_boxes(self, page: pymupdf.Page, elements: list) -> None:
        # page -> pixmap -> PIL image
        pix = page.get_pixmap()
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # create a coordinate system for the page
        coordinate_system = PixelSpace(width=image.width, height=image.height)
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image)

        colors = {
            "Title": "red",
            "Table": "blue",
        }
        
        # draw bounding boxes for each element
        for element in elements:
            # change coordinate system to match the page
            element.convert_coordinates_to_new_system(coordinate_system, in_place=True)
            color = colors.get(element.category, "green")
            poly = patches.Polygon(
                element.metadata.coordinates.points,
                closed=True,
                fill=False,
                edgecolor=color,
                linewidth=1
            )
            ax.add_patch(poly)
        
        plt.axis('off')
        plt.show()

    def render_page(self, page_number: int) -> None:
        page = pymupdf.open(self.path).load_page(page_number - 1)
        page_elements = [element for element in self.elements if element.metadata.page_number == page_number]

        self._plot_page_with_boxes(page, page_elements)

    def cleanup_elements(self) -> list:
        if not self.elements:
            return None
        
        # remove page numbers
        self.elements = [element for element in self.elements if not is_page_number(element)]

        for element in self.elements:
            if element.category == ElementType.TABLE:
                self._cleanup_table(element)
            else:
                self._cleanup_textual_element(element)

    def _cleanup_table(self, table: Element) -> None:
        html_content = table.metadata.text_as_html
        if table.metadata.table_as_cells:
            html_content = self._fix_table_cell_text(table)

        soup = BeautifulSoup(html_content, "html.parser")

        # remove extraneous '|' characters from text
        for element in soup.find_all(["td", "th"]):
            element.string = re.sub(r"\|+", " ", element.get_text(strip=True))

        cleaned_html = soup.prettify()
        table.metadata.text_as_html = cleaned_html

    def _cleanup_text(self, text: str) -> str:
        # remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # clean up table of contents
        text = re.sub(r'\.{4,}\s*\d+', '', text).strip()

        return text

    def _cleanup_textual_element(self, element: Element) -> None:
        element.text = self._cleanup_text(element.text)
    
    def _fix_table_cell_text(self, table, delimiter=" L ") -> str:
        """
        Move text after a delimiter to the text below in the same column.
        This is very much hardcoded.
        """
        if not table.metadata.text_as_html:
            return table.metadata.text_as_html

        soup = BeautifulSoup(table.metadata.text_as_html, "html.parser")
        rows = soup.find_all("tr")

        for row_idx, row in enumerate(rows[:-1]):
            cells = row.find_all("td")
            next_row_cells = rows[row_idx + 1].find_all("td")

            for col_idx, cell in enumerate(cells):
                if not cell.string:
                    continue

                # do some cleanup - hardcoded
                cell.string = re.sub(r'^L', '', cell.string)

                if delimiter in cell.string:
                    # split the cell text using the delimiter
                    parts = cell.string.rsplit(delimiter)
                    # keep the first part in the current cell
                    cell.string = parts[0].strip()

                    # move the rest to the next row in the same column
                    if col_idx < len(next_row_cells):
                        next_row_cells[col_idx].string = parts[1].strip() + " " + next_row_cells[col_idx].string

        return str(soup)

    def _get_tables(self):
        return [element for element in self.elements if element.category == ElementType.TABLE]
