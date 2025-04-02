import scrapy
from scrapy_selenium import SeleniumRequest
from urllib.parse import urljoin, urlparse, unquote
from datetime import datetime
import trafilatura
import json
import logging
from pathlib import Path
import os
import re

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent 
DATA_DIR = BASE_DIR / "data"

def load_urls(filename: str) -> dict[str, list[str]]:
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
        return data
    except Exception as e:
        return {}

def generate_tag(dir, filename):
    return f"[{Path(dir).name} {re.sub(r'[-_.]', ' ', Path(unquote(filename)).stem)}]".upper()

def append_to_metadata_file(
    metadata_filepath: str,
    filename: str,
    url: str,
    filetype: str,
    dir: str,
    title: str | None = None,
    tag: str | None = None,
) -> None:
    """
    Add information to metadata file.
    url: source url where pdf was downloaded from
    """
    with open(metadata_filepath, "r+") as jsonFile:
        try:
            data = json.load(jsonFile)
        except:
            data = []

    if any(entry['url'] == url for entry in data):
        return
    
    tag = tag or generate_tag(dir, filename)

    data.append(
        {
            "title": title or filename,
            "path": filename,
            "url": url,
            "filetype": filetype,
            "tag": tag
        }
    )

    with open(metadata_filepath, "w") as jsonFile:
        json.dump(data, jsonFile, indent=4)

class UpbSpider(scrapy.Spider):
    name = "upb"
    urls = load_urls("urls.json")
    logger = logging.getLogger(__name__)
    TARGET_DATE = datetime(2024, 10, 1)

    def start_requests(self):
        for faculty, url_list in self.urls.items():
            # create directory
            dir = (DATA_DIR / faculty).as_posix()
            os.makedirs(dir, exist_ok=True)

            # create metadata file
            metadata_filepath = os.path.join(dir, "metadata.json")
            os.open(metadata_filepath, os.O_CREAT | os.O_WRONLY)

            for url in url_list:
                yield SeleniumRequest(
                    url=url,
                    callback=self.parse,
                    errback=self.handle_error,
                    cb_kwargs={"dir": dir, "metadata_filepath": metadata_filepath},
                    wait_time=3,
                )

    def parse(self, response, dir, metadata_filepath):
        if response.status != 200:
            self.logger.error(f"Failed to fetch {response.url}, status code: {response.status}")
            return

        self.logger.info(f"Page loaded: {response.url}, status code: {response.status}")
        
        self.logger.info("Downloading pdfs...")

        # pdf_links = response.xpath('//a[contains(@href, ".pdf")]/@href').getall()
        pdf_links = response.xpath('//a[substring(@href, string-length(@href) - 3) = ".pdf"]/@href').getall()
        self.logger.info(f"Links found: {len(pdf_links)}")
        for link in pdf_links:
            pdf_url = urljoin(response.url, link)
            yield scrapy.Request(
                pdf_url,
                callback=self.save_pdf,
                cb_kwargs=dict(dir=dir, metadata_filepath=metadata_filepath)
            )

    def extract_page_content(self, response, dir, metadata_filepath):
        html_content = response.body
        clean_markdown = trafilatura.extract(html_content, include_tables=True)
        # extracted_markdown = trafilatura.extract(html_content, output_format="markdown", include_formatting=True)

        # # cleanup bolded text
        # clean_markdown = re.sub(r'(\*\*|\_\_)(.*?)\1', r'\2', extracted_markdown)

        # # cleanup italics
        # clean_markdown = re.sub(r'(\*|_)(.*?)\1', r'\2', clean_markdown)

        if not clean_markdown:
            self.logger.info(f"No content extracted from page {response.url}")
            return
        
        self.save_markdown(dir, response.url, clean_markdown, metadata_filepath)

    def save_markdown(self, dir, url, content, metadata_filepath):
        path = urlparse(url).path.strip('/').replace('/', '_')
        if not path: path = urlparse(url).netloc.lstrip('www.').replace('/', '_')
        filename = path + ".txt"
        path = os.path.join(dir, filename)

        if os.path.isfile(path):
            self.logger.info(f"File with name {path} already exists.")
            return
        
        with open(path, "w+", encoding='utf-8') as file:
            file.write(content)
        
        append_to_metadata_file(metadata_filepath, filename, url, "html", dir)

        self.logger.info(f"Saved page content from {url} at {path}")
    
    def save_pdf(self, response, dir, metadata_filepath):
        last_modified = response.headers.get('Last-Modified')
        if last_modified:
            last_modified = last_modified.decode('utf-8')
            last_modified_date = datetime.strptime(last_modified, '%a, %d %b %Y %H:%M:%S GMT')

            if last_modified_date < self.TARGET_DATE:
                return
    
            self.logger.info(f"Last Modified: {last_modified_date}")

        filename = response.url.split("/")[-1]
        path = os.path.join(dir, filename)

        if os.path.isfile(path):
            self.logger.info(f"PDF with name {filename} from url {response.url} already exists.")
            return

        with open(path, 'wb') as file:
            file.write(response.body)
        
        append_to_metadata_file(metadata_filepath, filename, response.url, "pdf", dir)

        self.logger.info(f"Saved PDF from {response.url} at {path}")

    def handle_error(self, response):
        self.logger.error(f"Ignored request: {response.url}. Possibly timed out.")
