import PyPDF2
import docx2txt
from PIL import Image
import pytesseract
import streamlit as st
import io
from .stream_processor import stream_pdf, stream_docx, stream_image_text

MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

def validate_file_size(file):
    size_mb = file.size / (1024 * 1024)
    if file.size > MAX_FILE_SIZE:
        raise ValueError(f"File size ({size_mb:.1f}MB) exceeds 100MB limit")
    if file.size == 0:
        raise ValueError("File is empty")

def process_pdf(pdf_file):
    try:
        validate_file_size(pdf_file)
        pdf_bytes = io.BytesIO(pdf_file.read())
        text_generator = stream_pdf(pdf_bytes)
        return "".join(text_generator)
    except Exception as e:
        raise ValueError(f"Error processing PDF: {str(e)}")

def process_docx(docx_file):
    try:
        validate_file_size(docx_file)
        docx_bytes = io.BytesIO(docx_file.read())
        text_generator = stream_docx(docx_bytes)
        return "".join(text_generator)
    except Exception as e:
        raise ValueError(f"Error processing Word document: {str(e)}")

def process_image(image_file):
    try:
        validate_file_size(image_file)
        image_bytes = io.BytesIO(image_file.read())
        text_generator = stream_image_text(image_bytes)
        return "".join(text_generator)
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")
