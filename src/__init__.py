"""Q&A System - гибридная система поиска ответов на вопросы."""

__version__ = "1.0.0"
__author__ = "Your Name"

from .data_loader import load_data, load_data_from_csv, load_data_from_excel
from .qa_system import HybridQASystem

__all__ = [
    "HybridQASystem",
    "load_data",
    "load_data_from_csv",
    "load_data_from_excel",
]
