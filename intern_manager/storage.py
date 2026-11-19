"""File-saving helpers for Intern Manager uploads."""

import logging
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import BinaryIO


logger = logging.getLogger(__name__)
DATA_DIR = Path(__file__).resolve().parent / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".png", ".jpg", ".jpeg"}
ALLOWED_FILE_TYPES = {"Resume", "Certificate", "ID", "Other"}


def ensure_upload_directory() -> None:
    """Create the uploads directory if it does not exist."""
    try:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.exception("Could not create uploads directory")
        raise


def _safe_component(value: str) -> str:
    """Replace characters unsafe or unsuitable for a path component."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_-")
    return cleaned or "intern"


def get_intern_folder(intern_id: int, intern_name: str) -> Path:
    """Return the sanitized upload directory path for an intern."""
    if isinstance(intern_id, bool) or not isinstance(intern_id, int) or intern_id < 0:
        raise ValueError("intern_id must be a non-negative integer.")
    return UPLOAD_DIR / f"{intern_id}_{_safe_component(intern_name)}"


def _read_upload(uploaded_file: BinaryIO) -> bytes:
    """Read an upload while enforcing the maximum file size."""
    size = getattr(uploaded_file, "size", None)
    if size is not None and size > MAX_FILE_SIZE:
        raise ValueError("Uploaded files must be 10 MB or smaller.")

    getbuffer = getattr(uploaded_file, "getbuffer", None)
    if callable(getbuffer):
        content = bytes(getbuffer())
        if len(content) > MAX_FILE_SIZE:
            raise ValueError("Uploaded files must be 10 MB or smaller.")
        return content

    content = bytearray()
    while len(content) <= MAX_FILE_SIZE:
        chunk = uploaded_file.read(min(64 * 1024, MAX_FILE_SIZE + 1 - len(content)))
        if not chunk:
            break
        if not isinstance(chunk, (bytes, bytearray)):
            raise TypeError("Uploaded file must provide binary content.")
        content.extend(chunk)
    if len(content) > MAX_FILE_SIZE:
        raise ValueError("Uploaded files must be 10 MB or smaller.")
    return bytes(content)


def _save_content(folder: Path, stem: str, extension: str, content: bytes) -> str:
    """Save content without overwriting an existing file."""
    ensure_upload_directory()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{stem}{extension}"
    try:
        _write_exclusive(path, content)
        logger.info("Saved uploaded file with extension %s", extension)
        return str(path)
    except FileExistsError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    timestamped_stem = f"{stem}_{timestamp}"
    attempt = 0
    while True:
        suffix = f"_{attempt}" if attempt else ""
        path = folder / f"{timestamped_stem}{suffix}{extension}"
        try:
            _write_exclusive(path, content)
            logger.info("Saved uploaded file with extension %s", extension)
            return str(path)
        except FileExistsError:
            attempt += 1


def _write_exclusive(path: Path, content: bytes) -> None:
    """Write a file exclusively and remove partial output if writing fails."""
    created = False
    try:
        with path.open("xb") as destination:
            created = True
            destination.write(content)
    except FileExistsError:
        raise
    except OSError:
        if created:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                logger.exception("Could not remove partially written upload")
        logger.exception("Could not save uploaded file")
        raise


def _file_details(uploaded_file: BinaryIO) -> tuple[str, str, bytes]:
    """Validate an uploaded filename and return its safe name and content."""
    filename = getattr(uploaded_file, "name", None)
    if not isinstance(filename, str) or not filename:
        raise ValueError("Uploaded file must have a filename.")

    basename = re.split(r"[\\/]", filename)[-1]
    original_stem, extension = Path(basename).stem, Path(basename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        logger.warning("Rejected upload with unsupported extension %s", extension)
        raise ValueError(f"Unsupported file type. Allowed extensions: {allowed}.")

    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", original_stem).strip("._-")
    if not safe_stem:
        safe_stem = "upload"
    return safe_stem, extension, _read_upload(uploaded_file)


def save_intern_file(
    intern_id: int,
    intern_name: str,
    uploaded_file: BinaryIO,
    file_type: str,
) -> str:
    """Save an allowed intern document and return its path."""
    if file_type not in ALLOWED_FILE_TYPES:
        allowed = ", ".join(sorted(ALLOWED_FILE_TYPES))
        raise ValueError(f"file_type must be one of: {allowed}.")
    stem, extension, content = _file_details(uploaded_file)
    return _save_content(
        get_intern_folder(intern_id, intern_name),
        stem,
        extension,
        content,
    )


def save_photo(
    intern_id: int,
    intern_name: str,
    uploaded_file: BinaryIO,
) -> str:
    """Save an allowed intern photo as photo.<extension> and return its path."""
    _, extension, content = _file_details(uploaded_file)
    return _save_content(
        get_intern_folder(intern_id, intern_name),
        "photo",
        extension,
        content,
    )


def delete_intern_folder(intern_id: int) -> bool:
    """Remove upload directories for an intern, returning whether any were found."""
    if isinstance(intern_id, bool) or not isinstance(intern_id, int) or intern_id < 0:
        raise ValueError("intern_id must be a non-negative integer.")
    ensure_upload_directory()
    folders = [
        path
        for path in UPLOAD_DIR.iterdir()
        if path.name.startswith(f"{intern_id}_") and path.is_dir()
    ]
    for folder in folders:
        try:
            shutil.rmtree(folder)
        except OSError:
            logger.exception("Could not delete uploads folder for intern_id=%s", intern_id)
            raise
    if folders:
        logger.info("Deleted upload folder(s) for intern_id=%s", intern_id)
    return bool(folders)


def delete_stored_file(file_path: str) -> bool:
    """Delete a stored upload when its resolved path is inside the uploads directory."""
    upload_root = UPLOAD_DIR.resolve()
    stored_path = Path(file_path).resolve()
    try:
        stored_path.relative_to(upload_root)
    except ValueError as error:
        logger.warning("Rejected attempt to delete a file outside uploads directory")
        raise ValueError("File path is outside the uploads directory.") from error

    if not stored_path.is_file():
        return False
    try:
        stored_path.unlink()
    except OSError:
        logger.exception("Could not delete stored upload")
        raise
    logger.info("Deleted stored upload")
    return True
