# Intern Manager

## Setup

From the project directory, create and activate a virtual environment, then install
the dependencies:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the requirements and start the Streamlit app:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The application creates `data/`, `data/uploads/`, and the SQLite database
automatically when it starts.

Intern detail reports are generated as `intern_details.pdf` in each intern's
upload folder and can be downloaded from **Search / View**.

## Optional resume autofill

On the **Add Intern** page, turn on **Use local Ollama to fill fields from a PDF
resume**, upload a text-based PDF, then select **Extract details from PDF**.
Review and edit the suggested name, email, phone, college, and skills before
submitting; the resume is saved as an attached Resume file.

The Python dependencies (`pypdf` and `ollama`) are installed with
`requirements.txt`. Ollama must also be running locally and have the model
downloaded. Pull the model once, then start the Ollama desktop app or run the
server in a separate terminal:

```bash
ollama pull llama3.1:8b
ollama serve
```

If Ollama is unavailable, the app displays an error and does not prevent manual
entry. Scanned PDFs without selectable text are not supported.

## Run tests

From the project directory, run:

```bash
python -m pytest
```
