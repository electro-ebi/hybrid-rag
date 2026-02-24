def load_text(file_path: str) -> str:
    """
    Load a plain text file and return its content.
    """

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    return content
