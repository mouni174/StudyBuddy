# StudyBuddy — RAG Tutor for College Notes

A Python + Streamlit college project that lets an administrator upload college notes and students ask questions using RAG.

## Features

### Admin
- Admin login
- Add subjects
- Upload PDF/DOCX/TXT notes
- Index notes into ChromaDB
- View users, notes and usage statistics

### Student
- Student registration/login
- Ask questions from notes
- RAG retrieval using Sentence Transformers + ChromaDB
- Gemini-generated answers
- Source/page references
- AI-generated MCQ quizzes
- Quiz scores/history
- Chat history
- Basic progress dashboard

## 1. Install Python

Recommended: Python 3.11 or 3.12.

Check:

```powershell
python --version
```

## 2. Create virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\venv\Scripts\Activate.ps1
```

## 3. Install dependencies

```powershell
pip install -r requirements.txt
```

## 4. Set Gemini API key

PowerShell for the current terminal:

```powershell
$env:GEMINI_API_KEY="YOUR_API_KEY"
```

Or copy `.env.example` to `.env` and set the key. For simplicity, the app reads Streamlit secrets or the environment variable.

## 5. Run

```powershell
streamlit run app.py
```

The browser should open the StudyBuddy application.

## Demo Admin

```text
Email: admin@studybuddy.local
Password: Admin@123
```

Change the demo credentials/password implementation before submitting the project.

## Recommended demo flow

1. Login as Admin.
2. Add a subject such as DBMS.
3. Upload a DBMS PDF.
4. Wait for the first embedding model download.
5. Login/register as a Student.
6. Select DBMS.
7. Ask a question.
8. Show the retrieved source/page.
9. Generate a quiz.
10. Submit the quiz.
11. Show progress/history.

## RAG explanation for viva

1. Notes are loaded from PDF/DOCX/TXT.
2. Text is divided into smaller chunks.
3. Sentence Transformer converts chunks into numerical embeddings.
4. ChromaDB stores those embeddings.
5. When a student asks a question, the question is also embedded.
6. ChromaDB performs similarity search.
7. Relevant note chunks are passed to Gemini.
8. Gemini generates an answer constrained to the retrieved notes.
9. The application displays source filenames/pages.

## Important

This is an MVP intended for a college project/demo. Before production use, add stronger authentication, password reset, file validation, per-user authorization, deletion/re-indexing of documents, rate limiting, and secret management.
