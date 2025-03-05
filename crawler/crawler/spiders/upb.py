from scrapy_selenium import SeleniumRequest
from urllib.parse import urljoin
import scrapy
from pathlib import Path
from datetime import datetime
import json
import os

# Construct the path to the data directory
FILE_DIR = Path(__file__).resolve().parent
DATA_DIR = FILE_DIR.parent.parent.parent / "data"

def is_date_updated(old_date: str, new_date: str) -> bool:
    date_format = "%a, %d %b %Y %H:%M:%S %Z"
    old_datetime = datetime.strptime(old_date, date_format)
    new_datetime = datetime.strptime(new_date, date_format)

    return old_datetime < new_datetime

def file_exists(
    filename: str,
    file_path: str,
    last_modified: str,
    metadata_filepath: str
) -> bool:
    """
    Check if a file with the given name exists,
    overwrite it if the last_modified timestamp in the metadata file was updated. 
    """
    if os.path.isfile(file_path):
        data = get_metadata(filename, metadata_filepath)
        if not data or "last_modified" not in data:
            return False
        
        return not is_date_updated(data["last_modified"], last_modified)

    return False

def get_metadata(filename: str,metadata_filepath: str) -> dict:
    with open(metadata_filepath, "r+") as jsonFile:
        try:
            data = json.load(jsonFile)
        except:
            return None
    
        if filename not in data:
            return None
        
        return data[filename]

def append_to_metadata_file(
    metadata_filepath: str,
    filename: str,
    url: str,
    last_modified: str
) -> None:
    """
    Add information to metadata file.
    url: source url where pdf was downloaded from
    last_modified: timestamp when the file was last changed
        (Last-Modified field from response header)
    """
    with open(metadata_filepath, "r+") as jsonFile:
        try:
            data = json.load(jsonFile)
        except:
            data = {}

        if filename not in data:
            data[filename] = {}
        
        data[filename]["url"] = url

        if last_modified:
            data[filename]["last_modified"] = last_modified

        jsonFile.seek(0)
        json.dump(data, jsonFile, indent=4)
        jsonFile.truncate()

class UpbSpider(scrapy.Spider):
    name = "upb"
    urls = [
        # "https://upb.ro/",
        "https://upb.ro/regulamente-si-rapoarte/",
    ]

    def start_requests(self):
        for url in self.urls:
            yield SeleniumRequest(url=url, callback=self.parse)

    def parse(self, response):
        page_url = response.url

        # # Extract text from page body
        # text = "".join(response.xpath('//body//text()[not(ancestor::style) and not(ancestor::script)]').getall())
        # text = re.sub(r'\s+', ' ', text).strip()
        # with open(f"text_{self.name}.txt", "w+", encoding="utf-8") as f:
        #     f.write(text)

        # Create directory and json source file
        pdf_dir = os.path.join(DATA_DIR, self.name)
        metadata_filepath = os.path.join(pdf_dir, "metadata.json")

        os.makedirs(pdf_dir, exist_ok=True)
        if not os.path.exists(metadata_filepath):
            open(metadata_filepath, "x")
            self.logger.info(f"Created metadata file {metadata_filepath}")

        # Download pdfs
        pdf_links = response.xpath('//a[contains(@href, ".pdf")]/@href').getall()
        for link in pdf_links:
            pdf_url = urljoin(response.url, link)
            yield scrapy.Request(
                pdf_url,
                callback=self.download_pdf,
                cb_kwargs=dict(pdf_dir=pdf_dir, metadata_filepath=metadata_filepath)
            )

    def download_pdf(self, response, pdf_dir, metadata_filepath):
        filename = response.url.split("/")[-1]
        pdf_path = os.path.join(pdf_dir, filename)

        last_modified = None
        if 'Last-Modified' in response.headers:
            last_modified = response.headers['Last-Modified'].decode("utf-8")

            if file_exists(filename, pdf_path, last_modified, metadata_filepath):
                self.logger.info(f"PDF with name {filename} from url {response.url} already exists and has not been modified.")
                return

        # Save pdf
        with open(pdf_path, 'wb') as f:
            f.write(response.body)
        
        append_to_metadata_file(metadata_filepath, filename, response.url, last_modified)

        self.logger.info(f"Downloaded PDF: {pdf_path}")
