from pypdf import PdfReader


def load_pdf(file_path: str) -> str:
    """
    Load a PDF file and extract raw text.
    """

    reader = PdfReader(file_path)
    

    text = ""


    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"

    return text
