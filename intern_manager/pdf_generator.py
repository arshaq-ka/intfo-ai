"""Generate printable intern detail reports."""

import logging
import os
import tempfile
import unicodedata
from datetime import date
from pathlib import Path
from typing import Any

from fpdf import FPDF
from fpdf.errors import FPDFException


logger = logging.getLogger(__name__)
_OUTPUT_NAME = "intern_details.pdf"


class PdfGenerationError(RuntimeError):
    """Raised when an intern detail PDF cannot be generated."""


def _font_candidates() -> list[tuple[str, Path, int]]:
    """Return available system fonts that can render Unicode text."""
    candidates: list[tuple[str, Path, int]] = []
    windows_root = os.environ.get("WINDIR")
    if windows_root:
        fonts = Path(windows_root) / "Fonts"
        candidates.extend(
            [
                ("InternArial", fonts / "arial.ttf", 0),
                ("InternMalgun", fonts / "malgun.ttf", 0),
                ("InternYaHei", fonts / "msyh.ttc", 0),
                ("InternGothic", fonts / "msgothic.ttc", 0),
                ("InternMingLiu", fonts / "mingliub.ttc", 0),
            ]
        )
    candidates.extend(
        [
            (
                "InternDejaVu",
                Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
                0,
            ),
            (
                "InternNoto",
                Path("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"),
                0,
            ),
            (
                "InternNotoCJK",
                Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
                0,
            ),
            (
                "InternArialMac",
                Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
                0,
            ),
        ]
    )
    return [candidate for candidate in candidates if candidate[1].is_file()]


def _configure_unicode_fonts(
    pdf: FPDF,
    required_glyphs: set[int],
) -> tuple[str | None, set[int]]:
    """Register Unicode fonts needed by the report and return their glyph map."""
    families: list[str] = []
    supported_glyphs: set[int] = set()
    for family, font_path, collection_font_number in _font_candidates():
        try:
            pdf.add_font(
                family,
                fname=font_path,
                collection_font_number=collection_font_number,
            )
        except (FPDFException, OSError, ValueError):
            logger.warning("Could not load Unicode font %s", font_path, exc_info=True)
            continue
        font = pdf.fonts[family.lower()]
        font_glyphs = set(font.cmap)
        if families and not font_glyphs.intersection(required_glyphs - supported_glyphs):
            continue
        families.append(family)
        supported_glyphs.update(font_glyphs)

    if not families:
        logger.warning("No Unicode system font found; unsupported characters will be replaced")
        return None, supported_glyphs
    if len(families) > 1:
        pdf.set_fallback_fonts(families[1:])
    return families[0], supported_glyphs


def _safe_text(value: Any, supported_glyphs: set[int]) -> str:
    """Normalize text and replace characters unavailable in installed fonts."""
    normalized = unicodedata.normalize("NFC", "" if value is None else str(value))
    safe_characters = []
    for character in normalized:
        if character in "\r\n":
            safe_characters.append(character)
        elif character == "\t":
            safe_characters.append("    ")
        elif ord(character) < 32:
            safe_characters.append(" ")
        elif supported_glyphs:
            safe_characters.append(
                character if ord(character) in supported_glyphs else "?"
            )
        else:
            try:
                character.encode("cp1252")
            except UnicodeEncodeError:
                safe_characters.append("?")
            else:
                safe_characters.append(character)
    return "".join(safe_characters)


def _write_field(
    pdf: FPDF,
    label: str,
    value: Any,
    unicode_font: str | None,
    supported_glyphs: set[int],
) -> None:
    """Write one wrapping label/value row."""
    pdf.set_font("Helvetica", style="B", size=10)
    pdf.cell(38, 6, f"{label}:", new_x="RIGHT", new_y="TOP")
    if unicode_font:
        pdf.set_font(unicode_font, size=10)
    else:
        pdf.set_font("Helvetica", size=10)
    text = _safe_text(value, supported_glyphs) or "(Not provided)"
    pdf.multi_cell(0, 6, text, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)


def generate_intern_pdf(
    intern: dict[str, Any],
    files: list[dict[str, Any]],
    folder: Path,
) -> Path:
    """Create an A4 intern report with details, photo, and attached-file metadata."""
    folder = Path(folder)
    temporary_path: Path | None = None
    try:
        folder.mkdir(parents=True, exist_ok=True)
        pdf = FPDF(format="A4")
        pdf.set_margins(15, 15, 15)
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        report_values = [
            intern.get(key)
            for key in (
                "full_name",
                "phone",
                "email",
                "college",
                "department",
                "skills",
                "start_date",
                "end_date",
                "status",
                "notes",
            )
        ]
        for attached_file in files:
            report_values.extend(
                [
                    attached_file.get("file_name"),
                    attached_file.get("file_type"),
                    attached_file.get("uploaded_at") or attached_file.get("uploaded_date"),
                ]
            )
        required_glyphs = {
            ord(character)
            for value in report_values
            for character in ("" if value is None else str(value))
        }
        unicode_font, supported_glyphs = _configure_unicode_fonts(
            pdf,
            required_glyphs,
        )

        pdf.set_font("Helvetica", style="B", size=20)
        pdf.cell(0, 10, "Intern Details", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", size=10)
        pdf.cell(
            0,
            7,
            f"Generated on: {date.today().isoformat()}",
            new_x="LMARGIN",
            new_y="NEXT",
        )

        photo_path = intern.get("photo_path")
        photo_added = False
        if photo_path:
            photo = Path(str(photo_path))
            if photo.is_file():
                try:
                    pdf.image(
                        photo,
                        x=160,
                        y=15,
                        w=35,
                        h=42,
                        keep_aspect_ratio=True,
                        alt_text="Intern photo",
                    )
                    photo_added = True
                except (FPDFException, OSError, ValueError):
                    logger.warning("Could not add intern photo to report", exc_info=True)

        pdf.set_y(63 if photo_added else 35)
        pdf.set_font("Helvetica", style="B", size=13)
        pdf.cell(0, 8, "Intern Information", new_x="LMARGIN", new_y="NEXT")

        fields = (
            ("Name", "full_name"),
            ("Phone", "phone"),
            ("Email", "email"),
            ("College", "college"),
            ("Department", "department"),
            ("Skills", "skills"),
            ("Start date", "start_date"),
            ("End date", "end_date"),
            ("Status", "status"),
            ("Notes", "notes"),
        )
        for label, key in fields:
            _write_field(
                pdf,
                label,
                intern.get(key),
                unicode_font,
                supported_glyphs,
            )

        pdf.ln(3)
        pdf.set_font("Helvetica", style="B", size=13)
        pdf.cell(0, 8, "Attached Files", new_x="LMARGIN", new_y="NEXT")
        if not files:
            if unicode_font:
                pdf.set_font(unicode_font, size=10)
            else:
                pdf.set_font("Helvetica", size=10)
            pdf.multi_cell(0, 6, "No attached files.")
        for attached_file in files:
            file_details = (
                f"File: {_safe_text(attached_file.get('file_name'), supported_glyphs)}"
                f" | Type: {_safe_text(attached_file.get('file_type'), supported_glyphs)}"
                f" | Uploaded: {_safe_text(attached_file.get('uploaded_at') or attached_file.get('uploaded_date'), supported_glyphs)}"
            )
            if unicode_font:
                pdf.set_font(unicode_font, size=10)
            else:
                pdf.set_font("Helvetica", size=10)
            pdf.multi_cell(0, 6, file_details, new_x="LMARGIN", new_y="NEXT")

        output_path = folder / _OUTPUT_NAME
        with tempfile.NamedTemporaryFile(
            prefix="intern_details_",
            suffix=".pdf",
            dir=folder,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        pdf.output(temporary_path)
        temporary_path.replace(output_path)
        logger.info("Generated intern detail PDF")
        return output_path
    except (FPDFException, OSError, ValueError) as error:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                logger.exception("Could not remove temporary intern PDF")
        logger.exception("Could not generate intern detail PDF")
        raise PdfGenerationError(f"Could not generate intern detail PDF: {error}") from error
