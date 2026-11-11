"""
Resume Parser Module.
Extracts text from PDF resumes using PyMuPDF, validates document integrity,
and extracts skills and contact information.
"""

from typing import Dict, Any, Union, BinaryIO
import io
import os
import pymupdf as fitz

from modules.preprocessor import clean_text, extract_contact_info, compute_text_stats
from modules.skills_dictionary import extract_skills_from_text


class ResumeParser:
    """
    Parses and validates PDF resumes, extracting raw text, structured skills,
    contact details, and clean text for machine learning inference.
    """

    def __init__(self, min_text_length: int = 50):
        self.min_text_length = min_text_length

    def extract_text_from_pdf(self, file_source: Union[str, bytes, BinaryIO]) -> Dict[str, Any]:
        """
        Extract text from a PDF file path, raw bytes, or file-like stream using PyMuPDF.
        
        Returns:
            dict: {
                "success": bool,
                "text": str,
                "pages": int,
                "error": str or None,
                "warning": str or None
            }
        """
        doc = None
        extracted_text = ""
        page_count = 0

        try:
            # Handle different input types
            if isinstance(file_source, str):
                if not os.path.exists(file_source):
                    return {"success": False, "text": "", "pages": 0, "error": f"File not found: {file_source}", "warning": None}
                doc = fitz.open(file_source)
            elif isinstance(file_source, bytes):
                if len(file_source) == 0:
                    return {"success": False, "text": "", "pages": 0, "error": "Uploaded file is empty (0 bytes).", "warning": None}
                doc = fitz.open(stream=file_source, filetype="pdf")
            elif hasattr(file_source, "read"):
                # Streamlit UploadedFile or file-like stream
                content = file_source.read()
                if len(content) == 0:
                    return {"success": False, "text": "", "pages": 0, "error": "Uploaded file is empty.", "warning": None}
                doc = fitz.open(stream=content, filetype="pdf")
                # Reset stream pointer if possible
                if hasattr(file_source, "seek"):
                    file_source.seek(0)
            else:
                return {"success": False, "text": "", "pages": 0, "error": "Unsupported file source type.", "warning": None}

            page_count = len(doc)
            if page_count == 0:
                return {"success": False, "text": "", "pages": 0, "error": "PDF has 0 pages.", "warning": None}

            # Extract text page by page
            text_blocks = []
            for page_num in range(page_count):
                page = doc.load_page(page_num)
                page_text = page.get_text("text")
                if page_text:
                    text_blocks.append(page_text)

            extracted_text = "\n".join(text_blocks).strip()

            warning = None
            if len(extracted_text) < self.min_text_length:
                warning = (
                    "Warning: Minimal or no text extracted. "
                    "The PDF might be a scanned image or protected document."
                )

            return {
                "success": True,
                "text": extracted_text,
                "pages": page_count,
                "error": None,
                "warning": warning
            }

        except Exception as e:
            return {
                "success": False,
                "text": "",
                "pages": 0,
                "error": f"Failed to parse PDF document: {str(e)}",
                "warning": None
            }
        finally:
            if doc:
                try:
                    doc.close()
                except Exception:
                    pass

    def parse(self, file_source: Union[str, bytes, BinaryIO], filename: str = "resume.pdf") -> Dict[str, Any]:
        """
        Complete parsing pipeline: validation, text extraction, cleaning,
        contact extraction, and skill identification.
        """
        # Validate extension
        if filename and not filename.lower().endswith(".pdf"):
            return {
                "is_valid": False,
                "filename": filename,
                "error": "Invalid file type. Please upload a PDF file (.pdf).",
                "raw_text": "",
                "clean_text": "",
                "skills": {"all": [], "by_category": {}},
                "contacts": {},
                "stats": {}
            }

        extraction = self.extract_text_from_pdf(file_source)

        if not extraction["success"]:
            return {
                "is_valid": False,
                "filename": filename,
                "error": extraction["error"],
                "raw_text": "",
                "clean_text": "",
                "skills": {"all": [], "by_category": {}},
                "contacts": {},
                "stats": {}
            }

        raw_text = extraction["text"]
        cleaned_text = clean_text(raw_text)
        skills = extract_skills_from_text(raw_text)
        contacts = extract_contact_info(raw_text)
        stats = compute_text_stats(raw_text)
        stats["page_count"] = extraction["pages"]

        return {
            "is_valid": True,
            "filename": filename,
            "warning": extraction["warning"],
            "raw_text": raw_text,
            "clean_text": cleaned_text,
            "skills": skills,
            "contacts": contacts,
            "stats": stats
        }

