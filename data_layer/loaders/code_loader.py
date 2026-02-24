def load_code(file_path: str) -> str:
    """
    Load a code file and return its content.
    """

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    return content
