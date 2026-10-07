"""
PDF parsing and text extraction utility.
Extracts clean plain text from candidate resume PDFs.
"""

import io
from pathlib import Path
from typing import Union, BinaryIO


def extract_text_from_pdf(pdf_source: Union[str, Path, BinaryIO, bytes]) -> str:
    """
    Extract readable text from a PDF file path, BytesIO stream, or raw bytes.

    Args:
        pdf_source: File path, Path object, file-like stream, or raw bytes.

    Returns:
        Cleaned plain text of the resume.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("pypdf is required. Install it using 'pip install pypdf'.")

    stream = None
    try:
        if isinstance(pdf_source, (str, Path)):
            file_path = Path(pdf_source)
            if not file_path.exists():
                raise FileNotFoundError(f"PDF file not found at: {file_path}")
            stream = open(file_path, "rb")
        elif isinstance(pdf_source, bytes):
            stream = io.BytesIO(pdf_source)
        else:
            stream = pdf_source

        reader = PdfReader(stream)
        extracted_pages = []
        for page_idx, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                extracted_pages.append(text.strip())

        full_text = "\n\n".join(extracted_pages)
        return clean_extracted_text(full_text)
    finally:
        # Close only if we opened it locally
        if isinstance(pdf_source, (str, Path)) and stream is not None:
            stream.close()


def clean_extracted_text(raw_text: str) -> str:
    """
    Normalize whitespaces, fix line-breaks, and remove non-printable characters.
    """
    if not raw_text:
        return ""

    lines = raw_text.splitlines()
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped:
            cleaned_lines.append(stripped)

    return "\n".join(cleaned_lines)

