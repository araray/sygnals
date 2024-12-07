import os


def create_directory(path):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def get_file_extension(file_path):
    """Extract the file extension from a path."""
    _, ext = os.path.splitext(file_path)
    return ext.lower()


def validate_file_format(file_path, supported_formats):
    """Check if the file has a supported format."""
    ext = get_file_extension(file_path)
    if ext not in supported_formats:
        raise ValueError(f"Unsupported file format: {ext}")
