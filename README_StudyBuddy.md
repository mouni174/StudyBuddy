# StudyBuddy — RAG Tutor for College Notes

StudyBuddy is a Python-based study assistant for college students. Administrators upload official study notes, and students can download those notes, ask questions grounded in the uploaded content, take quizzes, and track their learning progress.

## Features

### Admin
- Admin login and registration.
- Dashboard with student, subject, note, and activity information.
- Create and manage subjects.
- Upload official PDF, DOCX, and TXT notes.
- View uploaded notes and delete notes.
- Review student questions, answers, quiz attempts, and performance.
- Manage student accounts.

### Student
- Student registration and login.
- Browse notes by subject and download available PDF, DOCX, and TXT files.
- Ask StudyBuddy questions about indexed course notes.
- Generate and take quizzes.
- View progress, quiz results, and question history.

### Retrieval-Augmented Generation (RAG)
StudyBuddy extracts text from uploaded notes, splits it into chunks, creates embeddings, and stores them in ChromaDB. When a student asks a question, the app retrieves relevant chunks and sends the context to Gemini to generate a response. Answers can include source information from the uploaded notes.

**Important:** A file can be available for download even when no readable text was extracted. Scanned/image-only PDFs may require OCR before their contents can be indexed and used for RAG answers.

## Technology Stack
- Python
- Streamlit — web interface
- SQLite — application data
- ChromaDB — vector database
- Sentence Transformers (`all-MiniLM-L6-v2`) — text embeddings
- PyMuPDF — PDF text extraction
- python-docx — DOCX text extraction
- Google Gemini API (`google-genai`) — answer and quiz generation

## Project Structure

```text
StudyBuddy/
├── app.py
├── requirements.txt
├── .env.example
├── README.md
├── .streamlit/
│   └── config.toml          # Optional Streamlit configuration
└── data/
    ├── uploads/             # Uploaded note files
    ├── chroma/              # ChromaDB persistent data
    └── studybuddy.db        # SQLite database (created by the app)
```

Your exact folder layout may differ slightly depending on how you set up the project.

## Requirements
- Windows, macOS, or Linux
- Python 3.10 or 3.11/3.12, compatible with the versions pinned in your environment
- A Google Gemini API key
- Internet access for Gemini API calls and initial model downloads

## Installation on Windows

Open PowerShell in the project folder.

### 1. Create and activate a virtual environment

If you already have a working `venv`, activate it instead of recreating it.

```powershell
cd D:\StudyBuddy
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this in the same terminal session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

### 2. Install dependencies

Use `python -m pip` so pip runs through the active Python environment.

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you do not have a `requirements.txt` file, install the packages used by your current project version. Avoid installing a different PyTorch or Transformers build unless required by an actual error.

### 3. Configure the Gemini API key

Create a key in Google AI Studio: https://aistudio.google.com/apikey

Set it for the current PowerShell window:

```powershell
$env:GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
```

This environment-variable setting applies to that PowerShell session. If you close the terminal, set it again before launching the app.

Alternatively, create `.streamlit/secrets.toml` in the project folder:

```toml
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
```

Do not commit or share your real API key. Keep `secrets.toml` out of public repositories.

### 4. Run StudyBuddy

```powershell
streamlit run app.py
```

Streamlit will display a local URL, usually `http://localhost:8501`. Open that address in your browser.

To stop the app, return to PowerShell and press `Ctrl+C`.

## First-Time Setup

1. Open StudyBuddy in your browser.
2. Sign in with an existing administrator account, or use Admin Registration if enabled in your current version.
3. Create the required subjects, for example, **Computer Networks Notes** or **Java**.
4. Open the admin note-upload section.
5. Select the subject and upload a PDF, DOCX, or TXT file.
6. Click **Process & Index Notes** and wait for the result.
7. Sign in or register as a student.
8. Select a subject to ask questions, download notes, or take a quiz.

### Demo administrator account

Some project versions initialize this demo account:

- **Email:** `admin@studybuddy.local`
- **Password:** `Admin@123`

Use it only for a local demonstration. If the account does not work, use the administrator account created in your current database. Change or remove demo credentials before any real deployment.

## How to Use the Main Features

### Upload and index notes
Administrators upload course files and select a subject. Text-extractable content is indexed for RAG search. A successful upload does not necessarily mean the file contains extractable text.

### Download notes
Students open **Download Notes**, select a subject, and download available files. If the interface reports **File unavailable**, check that the original file still exists in the project's uploads directory and upload it again if necessary.

### Ask StudyBuddy
Students select a subject and ask a question. StudyBuddy retrieves related chunks from that subject's indexed notes and uses Gemini to compose a response. The answer depends on the quality and contents of the uploaded notes.

### Quizzes and progress
Students can generate quizzes and review their results and available activity history. Admins can review student activity and quiz performance through the admin area.

## Troubleshooting

### `GEMINI_API_KEY` is missing
Set the environment variable in the same PowerShell window used to start Streamlit, or configure `.streamlit/secrets.toml`. Restart Streamlit after changing configuration.

### Gemini returns `503 UNAVAILABLE`
This generally indicates temporary service unavailability or high demand. Wait briefly and retry. If the error persists, check the Gemini service/model availability and API quota.

### `404 NOT_FOUND` for a Gemini model
The configured model may not be available to your API key or may have changed. Verify the model name supported by your Gemini API account and update the model setting in `app.py`.

### `No readable text found`
The PDF may be scanned or image-only, protected, or otherwise difficult to extract. Try selecting and copying text from the PDF. If no text can be selected, run OCR on the PDF and upload the searchable version. The file may still be downloadable even if indexing returns zero chunks, depending on your app version.

### `File unavailable` in Download Notes
The database may contain a note record while the file itself has been moved, renamed, or deleted. Confirm the file exists under `data\uploads` and re-upload it through the admin page if needed.

### `ModuleNotFoundError`
Activate the StudyBuddy virtual environment and install requirements with:

```powershell
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

For a Streamlit file-watcher warning mentioning missing `torchvision`, the warning may be caused by optional Transformers image modules being inspected. It is separate from PDF text extraction. Only install a compatible `torchvision` build if your project actually needs it or the warning becomes a runtime error.

### Database and project data
The SQLite database, uploaded files, and ChromaDB persistent store are stored locally. Back up the entire `data` directory before replacing code or moving the project. Do not delete the database or ChromaDB folder unless you intend to reset the application's saved data.

## Security and Deployment Notes
- Keep your Gemini API key private.
- The demo admin credentials are not suitable for production.
- Public admin registration is convenient for a college demo but should be restricted in a real deployment.
- This project is intended as an educational prototype. Review authentication, authorization, file validation, upload size limits, secrets handling, and deployment security before exposing it to the public internet.

## License
No license has been specified. Add a license file if you plan to publish or distribute the project.
