import os
from typing import List
from docling.document_converter import DocumentConverter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from config import constants
from utils.logging import logger

class DocumentProcessor:
    def __init__(self):
        self.headers = [("#", "Header 1"), ("##", "Header 2")]
        
    def validate_files(self, files: List) -> None:
        """Validate the total size of the uploaded files."""
        total_size = sum(os.path.getsize(f.name) for f in files)
        if total_size > constants.MAX_TOTAL_SIZE:
            raise ValueError(f"Total size exceeds {constants.MAX_TOTAL_SIZE//1024//1024}MB limit")

    def process(self, files: List) -> List:
        """Process the uploaded files into chunks."""
        self.validate_files(files)
        all_chunks = []
        
        for file in files:
            try:
                # Validate file type
                if not file.name.endswith(('.pdf', '.docx', '.txt', '.md')):
                    logger.warning(f"Skipping unsupported file type: {file.name}")
                    continue
                
                converter = DocumentConverter()
                markdown = converter.convert(file.name).document.export_to_markdown()
                
                splitter = MarkdownHeaderTextSplitter(self.headers)
                chunks = splitter.split_text(markdown)
                all_chunks.extend(chunks)
                
                logger.info(f"Processed {file.name}: {len(chunks)} chunks")
            except Exception as e:
                logger.error(f"Failed to process {file.name}: {str(e)}")
                raise RuntimeError(f"Error processing document: {e}")
                
        return all_chunks