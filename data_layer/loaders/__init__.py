"""
Data loaders module.
"""

from .pdf_loader import load_pdf
from .text_loader import load_text
from .loader_factory import load_file

__all__ = ['load_pdf', 'load_text', 'load_file']