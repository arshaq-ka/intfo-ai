"""Tests for local PDF resume extraction and Ollama error handling."""

import json
from io import BytesIO

import app
import httpx
import ollama
import pytest
from pypdf import PdfWriter


def test_parse_resume_result_normalizes_json_fields():
    result = app._parse_resume_result(
        '```json\n{"name":"Ada Lovelace","email":"ada@example.test",'
        '"phone":"5551234567","college":"Analytical University",'
        '"skills":["Python","SQL"]}\n```'
    )

    assert result == {
        "name": "Ada Lovelace",
        "email": "ada@example.test",
        "phone": "5551234567",
        "college": "Analytical University",
        "skills": "Python, SQL",
    }


def test_extract_resume_details_uses_local_model(monkeypatch):
    monkeypatch.setattr(app, "_extract_pdf_text", lambda _pdf: "Resume text")
    received = {}

    def fake_generate(**kwargs):
        received.update(kwargs)
        return {
            "response": json.dumps(
                {
                    "name": "Ada Lovelace",
                    "email": "ada@example.test",
                    "phone": "5551234567",
                    "college": "Analytical University",
                    "skills": ["Python"],
                }
            )
        }

    monkeypatch.setattr(ollama, "generate", fake_generate)

    result = app._extract_resume_details(b"pdf bytes")

    assert result["name"] == "Ada Lovelace"
    assert result["skills"] == "Python"
    assert received["model"] == "llama3.1:8b"
    assert received["format"] == "json"
    assert "Resume text" in received["prompt"]


def test_extract_resume_details_reports_ollama_not_running(monkeypatch):
    monkeypatch.setattr(app, "_extract_pdf_text", lambda _pdf: "Resume text")

    def fail_to_connect(**_kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(ollama, "generate", fail_to_connect)

    with pytest.raises(app.ResumeExtractionError, match="Ollama is not running"):
        app._extract_resume_details(b"pdf bytes")


def test_extract_pdf_text_rejects_scanned_or_empty_pdf():
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    pdf_buffer = BytesIO()
    writer.write(pdf_buffer)

    with pytest.raises(app.ResumeExtractionError, match="No selectable text"):
        app._extract_pdf_text(pdf_buffer.getvalue())
