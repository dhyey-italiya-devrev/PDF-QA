from typing import Generator, Iterator
import io
from PyPDF2 import PdfReader
import docx2txt
from PIL import Image
import pytesseract

def stream_pdf(file_obj: io.BytesIO, chunk_size: int = 4096) -> Generator[str, None, None]:
    pdf_reader = PdfReader(file_obj)
    for page in pdf_reader.pages:
        text = page.extract_text()
        if text:
            yield text

def stream_docx(file_obj: io.BytesIO) -> Generator[str, None, None]:
    # Load the entire document but yield it in chunks
    text = docx2txt.process(file_obj)
    chunk_size = 4096
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]

def stream_image_text(image_obj: io.BytesIO) -> Generator[str, None, None]:
    img = Image.open(image_obj)
    # Process image in tiles for large images
    width, height = img.size
    tile_size = 1024
    
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            tile = img.crop((x, y, min(x + tile_size, width), min(y + tile_size, height)))
            text = pytesseract.image_to_string(tile)
            if text.strip():
                yield text
