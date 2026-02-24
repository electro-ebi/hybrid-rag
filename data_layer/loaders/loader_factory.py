import os

from .pdf_loader import load_pdf
from .text_loader import load_text
from .code_loader import load_code


CODE_EXTENSIONS = {
    ".py", ".js", ".cpp", ".c", ".java", ".ts", ".go"
}


def load_file(file_path: str) -> dict:
    """
    Load a file based on its extension and return structured data.
    """

    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".pdf":
        content = load_pdf(file_path)

    elif extension == ".txt":
        content = load_text(file_path)

    elif extension in CODE_EXTENSIONS:
        content = load_code(file_path)

    else:
        raise ValueError(f"Unsupported file type: {extension}")

    return {
        "content": content,
        "metadata": {
            "source": file_path,
            "file_type": extension
        }
    }
