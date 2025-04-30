from spacy import Language
import re
import pymupdf
import settings
from utils import *
from document_helpers import *
from transformers import AutoTokenizer
from bs4 import BeautifulSoup
os.environ["EXTRACT_TABLE_AS_CELLS"] = "True"
os.environ["TABLE_IMAGE_CROP_PAD"] = "20"
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.coordinates import PixelSpace
from document_processor import DocumentProcessor

class PdfProcessor(DocumentProcessor):
    def __init__(
        self, 
        path: str | Path,
        url: str,
        nlp: Language = None,
        tokenizer: AutoTokenizer = None,
        max_tokens: int = 512,
        logger: logging.Logger = None
    ) -> None:
        super().__init__(path, url, nlp, tokenizer, max_tokens, logger)
        self.type = "pdf"
        self.document = self._init_document(path)
        self.num_pages = len(self.document)
        self.elements = None

    def _init_document(self, path: str) -> None:
        return pymupdf.open(path, filetype="pdf")

    def process(self) -> None:
        self.partition_pdf()
        self.cleanup()
        self.perform_chunking()

    def partition_pdf(self) -> None:
        """
        Parse the PDF into a list of elements using lanchain unstructured.
        Detect complex layout in the document using OCR-based and Transformer-based models.
        """
        output_dir = settings.CONTENT_DIR
        self.elements = partition_pdf(
            filename=self.path,
            url=None,                                       # run inference locally, must have unstructured[local-inference] installed
            languages=["ron"],                              # use Romanian Tesseract language pack for OCR
            infer_table_structure=True,                     # extract tables
            strategy="hi_res",
            extract_images_in_pdf=True,                     # mandatory to be set as ``True``
            extract_image_block_types=["Image", "Table"],   # optional
            extract_image_block_to_payload=False,           # optional
            extract_image_block_output_dir=output_dir,      # optional - only works when ``extract_image_block_to_payload=False``
            max_partition=None,
        )

        self._logger.info(f"Partitioned PDF {self.path}. Extracted content found at {output_dir}")

    def perform_chunking(self) -> None:
        """
        Perform chunking using the unstructured document elements obtained after partitioning.
        Combines consecutive elements to form chunks as large as possible without exceeding the
        maximum sequence length of the embedding model (max_tokens). 

        A single element that exceeds the maximum chunk size is divided into two or more chunks using
        sentence segmentation.
        """
        if not self.elements: return None

        sentences = []
        for element in self.elements:
            element_category = classify_element(element.category)

            if element_category == ElementCategory.TABLE and table_contains_html(element):
                self.chunks.extend(self._split_sentences_into_chunks(sentences))
                sentences.clear()
                self.chunks.extend(self._get_table_chunks(element))
            
            elif element_category in [ElementCategory.TABLE, ElementCategory.TEXTUAL]:
                sentences.extend(self._sentencize(element.text, element.metadata.page_number))

        # use the sentences to build text chunks
        self.chunks.extend(self._split_sentences_into_chunks(sentences))

    def _get_table_chunks(self, table: Table) -> str:
        df = self._convert_table_to_dataframe(table)

        df_string = ' ; '.join(df.columns.astype(str).values.flatten())
        df_string += ' ; '.join(df.astype(str).values.flatten())

        sentences = self._sentencize(df_string, table.metadata.page_number)

        table_text = self._format_table_text(table)
        table_id = get_table_id(table_text)
        table_chunks = self._split_sentences_into_chunks(sentences, table_id, table_text)
        
        return table_chunks

    def _format_table_text(self, table: Table) -> str:
        """
        Convert table to markdown format, clean and compact.
        """
        # markdown = convert_table_to_dataframe(table).to_markdown(index=False)
        df = self._convert_table_to_dataframe(table)
        markdown = df.to_markdown(index=False)

        # remove whitespace, make markdown more compact
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

    def cleanup(self) -> list:
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
        
        # handle email addresses
        email_regex = r"[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}"
        email_addresses = re.findall(email_regex, table.text)
        if email_addresses:
            for email_address in email_addresses:
                index = email_address.find("@") + 1
                domain = email_address[index:]
                cleaned_html = re.sub(r"\([\w\.-]+{}".format(domain), f"@{domain}", cleaned_html, count=1)

        table.metadata.text_as_html = cleaned_html

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
                cell.string = re.sub(r'^L ', '', cell.string)

                if delimiter in cell.string:
                    # split the cell text using the delimiter
                    parts = cell.string.rsplit(delimiter)
                    # keep the first part in the current cell
                    cell.string = parts[0].strip()

                    # move the rest to the next row in the same column
                    if col_idx < len(next_row_cells):
                        next_row_cells[col_idx].string = parts[1].strip() + " " + next_row_cells[col_idx].string

        return str(soup)

    def _convert_table_to_dataframe(self, table: Table) -> str:
        """
        Convert table to dataframe.
        TODO: fill in missing text.
        """
        df = convert_table_to_dataframe(table)

        return df

    def _get_tables(self):
        return [element for element in self.elements if element.category == ElementType.TABLE]
