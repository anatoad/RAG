from typing import Generator

from scrapy_selenium import SeleniumRequest
from urllib.parse import urljoin
import scrapy

from .config import PDF_FILES_PATH
from datetime import datetime
import json
import re
import os

def compare_dates(old_date, new_date):
    date_format = "%a, %d %b %Y %H:%M:%S %Z"
    old_datetime = datetime.strptime(old_date.decode('utf-8'), date_format)
    new_datetime = datetime.strptime(new_date.decode('utf-8'), date_format)

    return old_datetime < new_datetime

def file_exists(filename, file_path, last_modified):
    # TODO: check here if a file with the same name exists
    # and if it does, check the last_modified date and if it has changed, overwrite it
    if os.path.isfile(file_path):
        return True

    return False

def append_to_metadata_file(metadata_filepath, filename, url):
    # save the source url
    with open(metadata_filepath, "r+") as jsonFile:
        try:
            data = json.load(jsonFile)
        except:
            data = {}

        if filename not in data:
            data[filename] = {}
        
        data[filename]["url"] = url

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
        pdf_dir = os.path.join(PDF_FILES_PATH, self.name)
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
            last_modified = response.headers['Last-Modified']

            # check to see if a file with the same name, from the same url already exists
            # overwrite it if the last_modified date
            if file_exists(filename, pdf_path, last_modified):
                self.logger.info(f"PDF with name {filename} from url {response.url} already exists and has not been modified.")

                return

        # save pdf
        with open(pdf_path, 'wb') as f:
            f.write(response.body)
        
        # save the source url
        append_to_metadata_file(metadata_filepath, filename, response.url)

        self.logger.info(f"Downloaded PDF: {pdf_path}")
