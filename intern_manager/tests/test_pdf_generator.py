"""Tests for intern detail PDF generation."""

from PIL import Image
from pypdf import PdfReader

from pdf_generator import generate_intern_pdf


def test_generate_intern_pdf_creates_folder_and_includes_details(tmp_path):
    photo_path = tmp_path / "photo.png"
    Image.new("RGB", (20, 20), color="blue").save(photo_path)
    report_folder = tmp_path / "nested" / "intern"
    intern = {
        "full_name": "王小明 — Zoë",
        "phone": "5551234567",
        "email": "intern@example.test",
        "college": "Example University",
        "department": "Engineering",
        "skills": "Python, SQL",
        "start_date": "2026-01-01",
        "end_date": "2026-06-01",
        "status": "Active",
        "notes": "Long note " * 900 + "END",
        "photo_path": str(photo_path),
    }
    files = [
        {
            "file_name": "résumé.pdf",
            "file_type": "Resume",
            "uploaded_at": "2026-01-02 10:30:00",
        }
    ]

    report_path = generate_intern_pdf(intern, files, report_folder)

    assert report_path == report_folder / "intern_details.pdf"
    assert report_path.is_file()
    reader = PdfReader(report_path)
    assert len(reader.pages) > 1
    assert "/XObject" in reader.pages[0]["/Resources"]
    content = "\n".join(page.extract_text() or "" for page in reader.pages)
    assert "Intern Details" in content
    assert "Generated on:" in content
    assert "王小明" in content
    assert "Email:" in content
    assert "Python, SQL" in content
    assert "résumé.pdf" in content
    assert "2026-01-02 10:30:00" in content
    assert "END" in content
