import os
import sqlite3
import hashlib
import secrets
import time
import json
from pathlib import Path
from datetime import datetime

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import fitz
from docx import Document
from google import genai
from google.genai import types

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "data" / "studybuddy.db"
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
CHROMA_DIR = BASE_DIR / "data" / "chroma"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="StudyBuddy", page_icon="🎓", layout="wide")

# ---------------- DATABASE ----------------

def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def hash_password(password):
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 120000)
    return f"{salt}${digest.hex()}"


def verify_password(password, stored):
    try:
        salt, digest = stored.split("$", 1)
        check = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 120000).hex()
        return secrets.compare_digest(check, digest)
    except Exception:
        return False


def init_db():
    conn = db()
    cur = conn.cursor()
    cur.executescript("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'student',
        created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS subjects (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        description TEXT DEFAULT '',
        created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        subject_id INTEGER NOT NULL,
        uploaded_by INTEGER,
        uploaded_at TEXT NOT NULL,
        stored_path TEXT,
        FOREIGN KEY(subject_id) REFERENCES subjects(id)
    );

    CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        subject_id INTEGER,
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS quiz_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        subject_id INTEGER,
        score INTEGER NOT NULL,
        total INTEGER NOT NULL,
        created_at TEXT NOT NULL
    );
    """)

    # Backward-compatible migration: older databases don't have stored_path.
    document_columns = [row[1] for row in cur.execute("PRAGMA table_info(documents)").fetchall()]
    if "stored_path" not in document_columns:
        cur.execute("ALTER TABLE documents ADD COLUMN stored_path TEXT")

    cur.execute(
        """
        INSERT OR IGNORE INTO users
        (name, email, password_hash, role, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            "Administrator",
            "admin@studybuddy.local",
            hash_password("Admin@123"),
            "admin",
            now()
        )
    )

    conn.commit()
    conn.close()
    


def create_user(name, email, password, role="student"):
    conn = db()
    try:
        conn.execute(
            "INSERT INTO users(name,email,password_hash,role,created_at) VALUES(?,?,?,?,?)",
            (name.strip(), email.strip().lower(), hash_password(password), role, now())
        )
        conn.commit()
        return True, "Account created."
    except sqlite3.IntegrityError:
        return False, "An account with this email already exists."
    finally:
        conn.close()


def authenticate(email, password):
    conn = db()
    row = conn.execute(
        "SELECT * FROM users WHERE email=?",
        (email.strip().lower(),)
    ).fetchone()
    conn.close()
    if row and verify_password(password, row["password_hash"]):
        return dict(row)
    return None

# ---------------- AI / RAG ----------------

@st.cache_resource
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name="study_notes",
        metadata={"hnsw:space": "cosine"}
    )


def get_gemini():
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        try:
            key = st.secrets.get("GEMINI_API_KEY")
        except Exception:
            key = None
    if not key:
        return None
    return genai.Client(api_key=key)


def extract_text(path):
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        doc = fitz.open(path)
        pages = [(i + 1, page.get_text("text")) for i, page in enumerate(doc)]
        doc.close()
        return pages
    if suffix == ".docx":
        doc = Document(path)
        return [(1, "\n".join(p.text for p in doc.paragraphs))]
    if suffix == ".txt":
        return [(1, path.read_text(encoding="utf-8", errors="ignore"))]
    raise ValueError("Only PDF, DOCX and TXT files are supported.")


def chunk_text(text, chunk_size=900, overlap=150):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def index_document(file_path, filename, subject_id):
    collection = get_collection()
    embedder = get_embedder()
    pages = extract_text(file_path)

    documents, metadatas, ids = [], [], []
    counter = 0

    for page_no, text in pages:
        for chunk in chunk_text(text):
            counter += 1
            documents.append(chunk)
            metadatas.append({
                "filename": filename,
                "subject_id": str(subject_id),
                "page": str(page_no)
            })
            ids.append(f"{subject_id}_{filename}_{counter}_{secrets.token_hex(5)}")

    if not documents:
        return 0

    embeddings = embedder.encode(documents, normalize_embeddings=True).tolist()
    collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    return len(documents)


def retrieve(question, subject_id, k=5):
    collection = get_collection()
    embedder = get_embedder()
    embedding = embedder.encode([question], normalize_embeddings=True).tolist()[0]
    result = collection.query(
        query_embeddings=[embedding],
        n_results=k,
        where={"subject_id": str(subject_id)}
    )
    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    return list(zip(docs, metas))


def generate_answer(question, subject_id):
    client = get_gemini()
    if not client:
        return None, "GEMINI_API_KEY is not configured."

    retrieved = retrieve(question, subject_id)
    if not retrieved:
        return "I couldn't find relevant information in the uploaded study material.", []

    context_parts = []
    sources = []
    for text, meta in retrieved:
        context_parts.append(f"Source: {meta['filename']} | Page: {meta['page']}\n{text}")
        sources.append((meta["filename"], meta["page"]))

    sources = list(dict.fromkeys(sources))
    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""
You are StudyBuddy, an AI tutor for college students.

Answer the student's question using ONLY the provided study material.
Do not invent facts that are not supported by the material.
If the answer cannot be found in the material, clearly say:
"I couldn't find this information in the uploaded study material."

Explain concepts in simple student-friendly language.
Use examples only when they are supported by the notes or are clearly labeled as a general example.
Do not mention that you are using hidden context.

STUDY MATERIAL:
{context}

STUDENT QUESTION:
{question}
"""

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.2)
            )
            return response.text, sources
        except Exception as e:
            error_text = str(e)
            if "503" in error_text or "UNAVAILABLE" in error_text:
                if attempt < 2:
                    time.sleep(3)
                    continue
                return "Gemini is temporarily busy. Please wait a few seconds and try again.", sources
            if "401" in error_text or "403" in error_text:
                return "Gemini API key is invalid or does not have permission to use this model.", sources
            if "404" in error_text or "NOT_FOUND" in error_text:
                return "The selected Gemini model is currently unavailable.", sources
            return f"Unable to generate the answer right now: {e}", sources
    return "Unable to generate the answer.", sources


def generate_quiz(subject_id, count=5):
    client = get_gemini()
    if not client:
        return None, "GEMINI_API_KEY is not configured."

    retrieved = retrieve("important concepts, definitions and exam questions", subject_id, 10)
    if not retrieved:
        return None, "No notes found for this subject."

    context = "\n\n".join(text for text, _ in retrieved)
    prompt = f"""
Create {count} multiple-choice questions from the following college notes.

Return ONLY valid JSON in this format:
[
  {{
    "question": "...",
    "options": ["A", "B", "C", "D"],
    "answer": 0,
    "explanation": "..."
  }}
]

"answer" must be the zero-based index of the correct option.
Use only information supported by the notes.

NOTES:
{context}
"""

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.3)
            )
            raw = response.text.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1]
                raw = raw.rsplit("```", 1)[0]
            try:
                return json.loads(raw), None
            except Exception:
                return None, "The AI returned an invalid quiz format. Please try again."
        except Exception as e:
            error_text = str(e)
            if "503" in error_text or "UNAVAILABLE" in error_text:
                if attempt < 2:
                    time.sleep(3)
                    continue
                return None, "Gemini is temporarily busy. Please try the quiz again in a few seconds."
            if "401" in error_text or "403" in error_text:
                return None, "Gemini API key is invalid or does not have permission to use this model."
            if "404" in error_text or "NOT_FOUND" in error_text:
                return None, "The selected Gemini model is currently unavailable."
            return None, f"Unable to generate the quiz right now: {e}"
    return None, "Unable to generate the quiz."

# ---------------- LOGIN ----------------

def apply_app_theme():
    st.markdown("""
    <style>
    :root {
        --sb-ink: #17233f;
        --sb-muted: #68748e;
        --sb-primary: #6558d9;
        --sb-border: #e7e9f3;
        --sb-surface: #ffffff;
    }
    .stApp {
        background: linear-gradient(180deg, #f8f9ff 0%, #ffffff 360px);
        color: var(--sb-ink);
    }
    [data-testid="stMainBlockContainer"] {
        max-width: 1240px;
        padding-top: 2.25rem;
        padding-bottom: 3rem;
    }
    [data-testid="stSidebar"] {
        background: #f5f5ff;
        border-right: 1px solid var(--sb-border);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: var(--sb-muted);
    }
    h1, h2, h3, h4 {
        color: var(--sb-ink);
        letter-spacing: -0.025em;
    }
    h1 { font-weight: 750; }
    [data-testid="stCaptionContainer"], .stCaption {
        color: var(--sb-muted);
    }
    [data-testid="stMetric"] {
        background: var(--sb-surface);
        border: 1px solid var(--sb-border);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        box-shadow: 0 5px 18px rgba(37, 45, 86, 0.045);
    }
    [data-testid="stMetricLabel"] {
        color: var(--sb-muted);
        font-weight: 550;
    }
    [data-testid="stMetricValue"] {
        color: var(--sb-ink);
        font-weight: 750;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--sb-border);
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.88);
        box-shadow: 0 5px 18px rgba(37, 45, 86, 0.035);
    }
    [data-testid="stTabs"] [role="tab"] {
        color: var(--sb-muted);
        font-weight: 600;
    }
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        color: var(--sb-primary);
    }
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea,
    [data-testid="stSelectbox"] [data-baseweb="select"] > div,
    [data-testid="stNumberInput"] input {
        border-radius: 10px;
    }
    div[data-testid="stButton"] > button,
    div[data-testid="stFormSubmitButton"] > button,
    div[data-testid="stDownloadButton"] > button {
        min-height: 2.65rem;
        border-radius: 10px;
        font-weight: 600;
        transition: border-color 120ms ease, background-color 120ms ease;
    }
    div[data-testid="stButton"] > button:hover,
    div[data-testid="stDownloadButton"] > button:hover {
        border-color: var(--sb-primary);
        color: var(--sb-primary);
    }
    [data-testid="stExpander"] {
        border-color: var(--sb-border);
        border-radius: 12px;
        background: var(--sb-surface);
    }
    [data-testid="stChatMessage"] {
        border: 1px solid var(--sb-border);
        border-radius: 16px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.7rem;
    }
    @media (max-width: 640px) {
        [data-testid="stMainBlockContainer"] {
            padding: 1.1rem 1rem 2rem;
        }
        .stApp [data-testid="stHeadingWithActionElements"] h1 {
            font-size: 1.8rem !important;
            line-height: 1.2 !important;
        }
        .stApp [data-testid="stHeadingWithActionElements"] h2 {
            font-size: 1.5rem !important;
            line-height: 1.25 !important;
        }
        .stApp [data-testid="stHeadingWithActionElements"] h3 {
            font-size: 1.25rem !important;
            line-height: 1.3 !important;
        }
        [data-testid="stMetric"] {
            padding: 0.8rem;
        }
        [data-testid="stTabs"] [role="tab"] {
            padding-left: 0.5rem;
            padding-right: 0.5rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)


def login_page():
    intro, auth = st.columns([1.05, 1.2], gap="large", vertical_alignment="center")
    with intro:
        with st.container(border=True):
            st.markdown("## :material/school: StudyBuddy")
            st.markdown("### A calmer way to study")
            st.write(
                "Your course notes, an AI study partner, and progress tools "
                "together in one focused workspace."
            )
            st.divider()
            st.markdown(":material/menu_book: **Learn from your materials**")
            st.caption("Get grounded answers from your official course notes.")
            st.markdown(":material/quiz: **Practice with purpose**")
            st.caption("Build confidence with quizzes based on the topics you study.")
            st.markdown(":material/insights: **See your progress**")
            st.caption("Keep track of questions, quiz results, and study history.")

    if "auth_role" not in st.session_state:
        st.session_state.auth_role = "student"

    with auth:
        with st.container(border=True):
            st.markdown("### Welcome back")
            st.caption("Sign in to continue to your StudyBuddy workspace.")
            role = st.radio(
                "Account type",
                ["Student", "Admin"],
                horizontal=True,
                index=0 if st.session_state.auth_role == "student" else 1,
            )

    selected_role = "admin" if role == "Admin" else "student"
    st.session_state.auth_role = selected_role

    role_label = "Admin" if selected_role == "admin" else "Student"
    login_tab, register_tab = st.tabs(
        [f":material/login: {role_label} login", f":material/person_add: {role_label} registration"]
    )

    with login_tab:
        with st.form(f"{selected_role}_login"):
            email = st.text_input(f"{role_label} email", placeholder=f"{role_label.lower()}@example.com")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button(
                f"Sign in as {role_label}", width="stretch", type="primary"
            )

        if submitted:
            user = authenticate(email, password)
            if user and user["role"] == selected_role:
                st.session_state.user = user
                st.rerun()
            elif user:
                other_role = "Student" if selected_role == "admin" else "Admin"
                st.error(f"This account is registered as {other_role}. Please select {other_role}.")
            else:
                st.error(f"Invalid {role_label.lower()} email or password.")

    with register_tab:
        if selected_role == "admin":
            st.warning(
                "Admin registration is enabled for this demo. In a real deployment, "
                "administrator accounts should be created by an existing administrator."
            )
        with st.form(f"{selected_role}_register"):
            name = st.text_input("Full name")
            email = st.text_input(f"{role_label} email")
            password = st.text_input("Password", type="password")
            confirm = st.text_input("Confirm password", type="password")
            submitted = st.form_submit_button(
                f"Create {role_label.lower()} account", width="stretch", type="primary"
            )

        if submitted:
            if not name or not email or not password:
                st.error("Please fill all fields.")
            elif password != confirm:
                st.error("Passwords do not match.")
            elif len(password) < 6:
                st.error("Password must contain at least 6 characters.")
            else:
                ok, msg = create_user(name, email, password, role=selected_role)
                (st.success if ok else st.error)(msg)


def get_subjects():
    conn = db()
    rows = conn.execute("SELECT * FROM subjects ORDER BY name").fetchall()
    conn.close()
    return rows


def delete_user(user_id):
    conn = db()
    user = conn.execute("SELECT id, role FROM users WHERE id=?", (user_id,)).fetchone()
    if not user:
        conn.close()
        return False, "User not found."
    if user["role"] == "admin":
        conn.close()
        return False, "Admin accounts cannot be deleted."

    conn.execute("DELETE FROM chat_history WHERE user_id=?", (user_id,))
    conn.execute("DELETE FROM quiz_results WHERE user_id=?", (user_id,))
    conn.execute("DELETE FROM users WHERE id=? AND role='student'", (user_id,))
    conn.commit()
    conn.close()
    return True, "Student account and activity deleted."


def delete_subject(subject_id):
    conn = db()
    subject = conn.execute("SELECT id, name FROM subjects WHERE id=?", (subject_id,)).fetchone()
    if not subject:
        conn.close()
        return False, "Subject not found."

    conn.execute("DELETE FROM documents WHERE subject_id=?", (subject_id,))
    conn.execute("DELETE FROM subjects WHERE id=?", (subject_id,))
    conn.commit()
    conn.close()

    try:
        collection = get_collection()
        result = collection.get(where={"subject_id": str(subject_id)})
        ids = result.get("ids", [])
        if ids:
            collection.delete(ids=ids)
    except Exception:
        pass

    return True, "Subject deleted."


def delete_document(document_id):
    conn = db()
    row = conn.execute(
        "SELECT id, filename, subject_id, stored_path FROM documents WHERE id=?",
        (document_id,)
    ).fetchone()
    if not row:
        conn.close()
        return False, "Note not found."

    filename = row["filename"]
    subject_id = row["subject_id"]
    stored_path = row["stored_path"] if "stored_path" in row.keys() else None
    conn.execute("DELETE FROM documents WHERE id=?", (document_id,))
    conn.commit()
    conn.close()

    try:
        collection = get_collection()
        result = collection.get(
            where={"$and": [{"subject_id": str(subject_id)}, {"filename": filename}]}
        )
        ids = result.get("ids", [])
        if ids:
            collection.delete(ids=ids)
    except Exception:
        pass

    try:
        if stored_path and Path(stored_path).is_file():
            Path(stored_path).unlink(missing_ok=True)
        else:
            for candidate in UPLOAD_DIR.glob("*"):
                if candidate.name.endswith(filename):
                    candidate.unlink(missing_ok=True)
    except Exception:
        pass

    return True, "Note deleted."

# ---------------- ADMIN DASHBOARD ----------------

def admin_page(user):
    with st.container(border=True):
        st.title(":material/admin_panel_settings: Admin dashboard")
        st.caption("Manage your learning resources, students, and platform activity.")

    conn = db()
    students = conn.execute("SELECT COUNT(*) c FROM users WHERE role='student'").fetchone()["c"]
    subjects = conn.execute("SELECT COUNT(*) c FROM subjects").fetchone()["c"]
    notes = conn.execute("SELECT COUNT(*) c FROM documents").fetchone()["c"]
    questions = conn.execute("SELECT COUNT(*) c FROM chat_history").fetchone()["c"]
    quiz_attempts = conn.execute("SELECT COUNT(*) c FROM quiz_results").fetchone()["c"]
    conn.close()

    metric_row_one = st.columns(3)
    metric_row_one[0].metric("Students", students)
    metric_row_one[1].metric("Subjects", subjects)
    metric_row_one[2].metric("Study notes", notes)
    metric_row_two = st.columns(2)
    metric_row_two[0].metric("Questions asked", questions)
    metric_row_two[1].metric("Quiz attempts", quiz_attempts)

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs([
        ":material/menu_book: Subjects",
        ":material/description: Study notes",
        ":material/group: Students",
        ":material/insights: Activity"
    ])

    # Subjects
    with tab1:
        st.subheader("Manage subjects")
        st.caption("Create and organize the subjects students can study.")
        with st.form("subject"):
            name = st.text_input("Subject name", placeholder="Example: Computer Networks")
            desc = st.text_area("Description", placeholder="Enter a short description.")
            submitted = st.form_submit_button("Add Subject", type="primary")

        if submitted:
            if not name.strip():
                st.error("Enter a subject name.")
            else:
                conn = db()
                try:
                    conn.execute(
                        "INSERT INTO subjects(name,description,created_at) VALUES(?,?,?)",
                        (name.strip(), desc.strip(), now())
                    )
                    conn.commit()
                    st.success("Subject added successfully.")
                except sqlite3.IntegrityError:
                    st.error("Subject already exists.")
                finally:
                    conn.close()
                st.rerun()

        rows = get_subjects()
        if rows:
            st.markdown("### Your subjects")
            for row in rows:
                with st.container(border=True):
                    col1, col2, col3 = st.columns([2, 5, 1], vertical_alignment="center")
                    col1.markdown(f"**{row['name']}**")
                    col2.caption(row["description"] or "No description")
                    if col3.button("Delete", key=f"delete_subject_{row['id']}"):
                        st.session_state["delete_subject_id"] = row["id"]

            if "delete_subject_id" in st.session_state:
                subject_id = st.session_state["delete_subject_id"]
                subject_row = next((r for r in rows if r["id"] == subject_id), None)
                if subject_row:
                    st.warning(f"Delete subject '{subject_row['name']}' and its indexed notes?")
                    confirm_col, cancel_col = st.columns(2)
                    if confirm_col.button("Yes, Delete Subject", type="primary", key="confirm_delete_subject"):
                        ok, msg = delete_subject(subject_id)
                        (st.success if ok else st.error)(msg)
                        del st.session_state["delete_subject_id"]
                        st.rerun()
                    if cancel_col.button("Cancel", key="cancel_delete_subject"):
                        del st.session_state["delete_subject_id"]
                        st.rerun()
        else:
            st.info("No subjects have been created yet.")

    # Notes
    with tab2:
        st.subheader("Upload study notes")
        st.caption("Add PDF, DOCX, or TXT course materials to the subject library.")
        rows = get_subjects()
        if not rows:
            st.warning("Create a subject before uploading notes.")
        else:
            subject_map = {r["name"]: r["id"] for r in rows}
            subject_name = st.selectbox("Subject", list(subject_map))
            uploaded = st.file_uploader(
                "Upload PDF / DOCX / TXT notes",
                type=["pdf", "docx", "txt"],
                accept_multiple_files=True
            )

            if uploaded and st.button("Process & Index Notes", type="primary"):
                with st.spinner("Saving and indexing study notes..."):
                    for file in uploaded:
                        safe_name = f"{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{file.name}"
                        path = UPLOAD_DIR / safe_name
                        path.write_bytes(file.getbuffer())
                        try:
                            # Keep the uploaded file available for student downloads even if
                            # this PDF has no selectable text for RAG indexing.
                            chunks = index_document(path, file.name, subject_map[subject_name])

                            conn = db()
                            conn.execute(
                                "INSERT INTO documents(filename,subject_id,uploaded_by,uploaded_at,stored_path) VALUES(?,?,?,?,?)",
                                (file.name, subject_map[subject_name], user["id"], now(), str(path.resolve()))
                            )
                            conn.commit()
                            conn.close()

                            if chunks == 0:
                                st.warning(
                                    f"{file.name} was uploaded and is available to download, "
                                    "but no readable text was found for StudyBuddy Q&A. "
                                    "Scanned PDFs need OCR before their contents can be searched."
                                )
                            else:
                                st.success(f"{file.name}: uploaded and indexed {chunks} chunks.")
                        except Exception as e:
                            st.error(f"{file.name}: {e}")

            conn = db()
            docs = conn.execute(
                """
                SELECT d.id, d.filename, s.name subject, d.uploaded_at, u.name uploaded_by
                FROM documents d
                JOIN subjects s ON s.id=d.subject_id
                LEFT JOIN users u ON u.id=d.uploaded_by
                ORDER BY d.id DESC
                """
            ).fetchall()
            conn.close()

            if docs:
                st.markdown("### Uploaded notes")
                for doc in docs:
                    with st.container(border=True):
                        col1, col2, col3, col4, col5 = st.columns(
                            [3, 2, 2, 2, 1], vertical_alignment="center"
                        )
                        col1.markdown(f"**{doc['filename']}**")
                        col2.caption(doc["subject"])
                        col3.caption(doc["uploaded_at"])
                        col4.caption(doc["uploaded_by"] or "Unknown")
                        if col5.button("Delete", key=f"delete_doc_{doc['id']}"):
                            st.session_state["delete_doc_id"] = doc["id"]

                if "delete_doc_id" in st.session_state:
                    doc_id = st.session_state["delete_doc_id"]
                    selected_doc = next((d for d in docs if d["id"] == doc_id), None)
                    if selected_doc:
                        st.warning(f"Delete '{selected_doc['filename']}' from the StudyBuddy knowledge base?")
                        confirm_col, cancel_col = st.columns(2)
                        if confirm_col.button("Yes, Delete Note", type="primary", key="confirm_delete_doc"):
                            ok, msg = delete_document(doc_id)
                            (st.success if ok else st.error)(msg)
                            del st.session_state["delete_doc_id"]
                            st.rerun()
                        if cancel_col.button("Cancel", key="cancel_delete_doc"):
                            del st.session_state["delete_doc_id"]
                            st.rerun()
            else:
                st.info("No notes have been uploaded yet.")

    # Users
    with tab3:
        st.subheader("Manage students")
        st.caption("Review account activity and manage student access.")
        conn = db()
        rows = conn.execute(
            """
            SELECT
                u.id, u.name, u.email, u.role, u.created_at,
                COUNT(DISTINCT h.id) AS questions,
                COUNT(DISTINCT q.id) AS quiz_attempts,
                COALESCE(ROUND(AVG(CASE WHEN q.total > 0 THEN (q.score * 100.0 / q.total) END), 1), 0) AS avg_quiz_score
            FROM users u
            LEFT JOIN chat_history h ON h.user_id=u.id
            LEFT JOIN quiz_results q ON q.user_id=u.id
            WHERE u.role='student'
            GROUP BY u.id, u.name, u.email, u.role, u.created_at
            ORDER BY u.id DESC
            """
        ).fetchall()
        conn.close()

        if not rows:
            st.info("No student accounts found.")
        else:
            table_data = []
            for row in rows:
                table_data.append({
                    "ID": row["id"],
                    "Name": row["name"],
                    "Email": row["email"],
                    "Registered": row["created_at"],
                    "Questions": row["questions"],
                    "Quiz Attempts": row["quiz_attempts"],
                    "Avg Quiz Score": f"{row['avg_quiz_score']:.1f}%"
                })
            st.dataframe(table_data, use_container_width=True, hide_index=True)

            st.markdown("### Student actions")
            student_map = {f"{r['name']} — {r['email']}": r["id"] for r in rows}
            selected_student = st.selectbox("Select a student", list(student_map), key="admin_selected_student")
            selected_student_id = student_map[selected_student]

            action_col1, action_col2 = st.columns(2)
            if action_col1.button(
                ":material/insights: View student activity", width="stretch"
            ):
                st.session_state["activity_student_id"] = selected_student_id
                st.info("Open the 'Activity' tab to view this student's questions and quiz performance.")
            if action_col2.button(
                ":material/person_remove: Delete student", width="stretch"
            ):
                st.session_state["delete_user_id"] = selected_student_id

            if "delete_user_id" in st.session_state:
                delete_id = st.session_state["delete_user_id"]
                delete_row = next((r for r in rows if r["id"] == delete_id), None)
                if delete_row:
                    st.warning(
                        f"Delete student '{delete_row['name']}'? Their chat history and quiz history will also be deleted."
                    )
                    confirm_col, cancel_col = st.columns(2)
                    if confirm_col.button("Yes, Delete Student", type="primary", key="confirm_delete_user"):
                        ok, msg = delete_user(delete_id)
                        (st.success if ok else st.error)(msg)
                        del st.session_state["delete_user_id"]
                        st.rerun()
                    if cancel_col.button("Cancel", key="cancel_delete_user"):
                        del st.session_state["delete_user_id"]
                        st.rerun()

    # Student Activity
    with tab4:
        st.subheader("Student activity and performance")
        conn = db()
        student_rows = conn.execute(
            "SELECT id, name, email FROM users WHERE role='student' ORDER BY name"
        ).fetchall()
        conn.close()

        if not student_rows:
            st.info("No student activity is available yet.")
        else:
            student_map = {f"{r['name']} — {r['email']}": r["id"] for r in student_rows}
            default_student_id = st.session_state.get("activity_student_id")
            names = list(student_map)
            default_index = 0
            if default_student_id:
                for index, label in enumerate(names):
                    if student_map[label] == default_student_id:
                        default_index = index
                        break

            selected_label = st.selectbox(
                "Select student",
                names,
                index=default_index,
                key="activity_student_selector"
            )
            student_id = student_map[selected_label]

            conn = db()
            student = conn.execute(
                "SELECT id, name, email, created_at FROM users WHERE id=? AND role='student'",
                (student_id,)
            ).fetchone()
            question_count = conn.execute(
                "SELECT COUNT(*) c FROM chat_history WHERE user_id=?",
                (student_id,)
            ).fetchone()["c"]
            quiz_count = conn.execute(
                "SELECT COUNT(*) c FROM quiz_results WHERE user_id=?",
                (student_id,)
            ).fetchone()["c"]
            avg_score = conn.execute(
                """
                SELECT AVG(CASE WHEN total > 0 THEN score * 100.0 / total END) avg_score
                FROM quiz_results WHERE user_id=?
                """,
                (student_id,)
            ).fetchone()["avg_score"]
            question_rows = conn.execute(
                """
                SELECT h.question, h.answer, h.created_at, s.name subject
                FROM chat_history h
                LEFT JOIN subjects s ON s.id=h.subject_id
                WHERE h.user_id=? ORDER BY h.id DESC
                """,
                (student_id,)
            ).fetchall()
            quiz_rows = conn.execute(
                """
                SELECT q.score, q.total, q.created_at, s.name subject
                FROM quiz_results q
                LEFT JOIN subjects s ON s.id=q.subject_id
                WHERE q.user_id=? ORDER BY q.id DESC
                """,
                (student_id,)
            ).fetchall()
            conn.close()

            if student:
                st.markdown(f"### :material/person: {student['name']}")
                st.caption(f"Email: {student['email']} | Registered: {student['created_at']}")
                x, y, z, w = st.columns(4)
                x.metric("Questions Asked", question_count)
                y.metric("Quiz Attempts", quiz_count)
                z.metric("Average Quiz Score", f"{avg_score:.1f}%" if avg_score is not None else "—")
                w.metric(
                    "Subjects Used",
                    len(set(r["subject"] for r in question_rows if r["subject"]))
                )

            st.divider()
            st.markdown("### :material/chat: Questions asked")
            if question_rows:
                for row in question_rows:
                    subject = row["subject"] or "Unknown Subject"
                    with st.expander(f"{row['created_at']} — {subject} — {row['question']}"):
                        st.markdown("**Question**")
                        st.write(row["question"])
                        st.markdown("**StudyBuddy Answer**")
                        st.write(row["answer"])
            else:
                st.info("This student has not asked any questions yet.")

            st.markdown("### :material/quiz: Quiz performance")
            if quiz_rows:
                quiz_table = []
                for row in quiz_rows:
                    percentage = (row["score"] / row["total"]) * 100 if row["total"] else 0
                    quiz_table.append({
                        "Date": row["created_at"],
                        "Subject": row["subject"] or "Unknown Subject",
                        "Score": f"{row['score']}/{row['total']}",
                        "Percentage": f"{percentage:.1f}%"
                    })
                st.dataframe(quiz_table, use_container_width=True, hide_index=True)
            else:
                st.info("This student has not completed any quizzes yet.")

# ---------------- STUDENT DASHBOARD ----------------

def student_page(user):
    with st.container(border=True):
        st.title(f"Welcome, {user['name']}")
        st.caption("Your study space to learn, practice, and track your progress.")

    subjects = get_subjects()
    if not subjects:
        st.info("Your study space is ready. Subjects and study notes will appear here when available.")
        return

    subject_map = {r["name"]: r["id"] for r in subjects}
    with st.container(border=True):
        subject_name = st.selectbox("Choose a subject", list(subject_map))
        subject_description = next(
            (row["description"] for row in subjects if row["name"] == subject_name),
            ""
        )
        st.caption(
            subject_description
            or "Ask questions and practice using the study notes for this subject."
        )
    subject_id = subject_map[subject_name]

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        ":material/auto_awesome: Study assistant",
        ":material/download: Study notes",
        ":material/quiz: Practice quiz",
        ":material/insights: My progress",
        ":material/history: History"
    ])

    with tab1:
        with st.container(border=True):
            st.subheader("Ask StudyBuddy")
            st.caption(
                f"Answers are grounded in the uploaded notes for **{subject_name}**."
            )

        conn = db()
        conversation = conn.execute(
            """
            SELECT id, question, answer, created_at
            FROM chat_history
            WHERE user_id=? AND subject_id=?
            ORDER BY id DESC LIMIT 50
            """,
            (user["id"], subject_id)
        ).fetchall()
        conn.close()
        conversation = list(reversed(conversation))
        source_key = f"chat_sources_{user['id']}_{subject_id}"
        chat_sources = st.session_state.setdefault(source_key, {})

        if not conversation:
            st.info("Start a conversation by asking a question about your study notes.")
        else:
            for message in conversation:
                with st.chat_message("user", avatar=":material/person:"):
                    st.markdown(message["question"])
                with st.chat_message("assistant", avatar=":material/auto_awesome:"):
                    st.markdown(message["answer"])
                    sources = chat_sources.get(message["id"], [])
                    if sources:
                        with st.expander(f"Sources · {len(sources)}"):
                            for filename, page in sources:
                                st.caption(f":material/article: {filename} · Page {page}")

        suggestion_columns = st.columns(3)
        suggested_question = None
        if suggestion_columns[0].button(
            "Summarize key ideas", key=f"chat_suggest_summary_{subject_id}"
        ):
            suggested_question = f"Summarize the key ideas in {subject_name} from my notes."
        if suggestion_columns[1].button(
            "Explain a concept", key=f"chat_suggest_explain_{subject_id}"
        ):
            suggested_question = f"Explain an important concept from my {subject_name} notes."
        if suggestion_columns[2].button(
            "Review definitions", key=f"chat_suggest_definitions_{subject_id}"
        ):
            suggested_question = f"Review the important definitions in my {subject_name} notes."

        question = st.chat_input(
            "Ask a question about your study notes...",
            key=f"chat_input_{subject_id}"
        )
        question = suggested_question or question

        if question:
            with st.spinner("Searching your notes and preparing an answer..."):
                answer, sources = generate_answer(question, subject_id)

            error_prefixes = (
                "GEMINI_API_KEY",
                "Unable to generate",
                "Gemini is temporarily",
                "The selected Gemini",
                "Gemini API key is invalid",
            )
            if answer is None:
                st.error(sources if isinstance(sources, str) else "Unable to generate an answer.")
            elif isinstance(answer, str) and answer.startswith(error_prefixes):
                st.error(answer)
            else:
                with st.chat_message("user", avatar=":material/person:"):
                    st.markdown(question)
                with st.chat_message("assistant", avatar=":material/auto_awesome:"):
                    st.markdown(answer)
                    if sources:
                        with st.expander(f"Sources · {len(sources)}"):
                            for filename, page in sources:
                                st.caption(f":material/article: {filename} · Page {page}")

                conn = db()
                cursor = conn.execute(
                    "INSERT INTO chat_history(user_id,subject_id,question,answer,created_at) VALUES(?,?,?,?,?)",
                    (user["id"], subject_id, question, answer, now())
                )
                conn.commit()
                chat_sources[cursor.lastrowid] = sources
                conn.close()

    with tab2:
        st.subheader("Your study notes")
        st.caption(f"Browse and download the official materials for **{subject_name}**.")

        conn = db()
        note_rows = conn.execute(
            "SELECT id, filename, uploaded_at, stored_path FROM documents WHERE subject_id=? ORDER BY id DESC",
            (subject_id,)
        ).fetchall()
        conn.close()

        if not note_rows:
            st.info("No notes are available for this subject yet. Please check back later.")
        else:
            st.caption(f"{len(note_rows)} file(s) available")
            for note in note_rows:
                file_name = note["filename"]
                # Prefer the exact path saved with this database record.
                file_candidates = []
                saved_path = note["stored_path"] if "stored_path" in note.keys() else None
                if saved_path:
                    candidate = Path(saved_path)
                    if candidate.is_file():
                        file_candidates = [candidate]
                    else:
                        # Fallback when the project folder has been moved.
                        local_candidate = UPLOAD_DIR / candidate.name
                        if local_candidate.is_file():
                            file_candidates = [local_candidate]

                # Backward compatibility for older rows that predate stored_path.
                if not file_candidates:
                    file_candidates = list(UPLOAD_DIR.glob(f"*_{file_name}"))
                if not file_candidates:
                    file_candidates = [f for f in UPLOAD_DIR.glob("*") if f.name.endswith(file_name)]
                file_candidates = [f for f in file_candidates if f.is_file()]

                with st.container(border=True):
                    note_col, date_col, action_col = st.columns([4, 2, 2], vertical_alignment="center")
                    suffix = Path(file_name).suffix.lower()
                    icon = (
                        "picture_as_pdf" if suffix == ".pdf"
                        else ("description" if suffix == ".docx" else "article")
                    )
                    note_col.markdown(f":material/{icon}: **{file_name}**")
                    note_col.caption(f"{suffix.lstrip('.').upper() or 'FILE'} document")
                    date_col.caption("Uploaded")
                    date_col.write(note["uploaded_at"])

                    if file_candidates:
                        # If filenames repeat, use the most recently uploaded matching file.
                        file_path = max(file_candidates, key=lambda f: f.stat().st_mtime)
                        mime = {
                            ".pdf": "application/pdf",
                            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            ".txt": "text/plain",
                        }.get(suffix, "application/octet-stream")
                        action_col.download_button(
                            ":material/download: Download",
                            data=file_path.read_bytes(),
                            file_name=file_name,
                            mime=mime,
                            key=f"download_note_{note['id']}",
                            use_container_width=True
                        )
                    else:
                        action_col.warning("File unavailable")
                        st.caption("The note record exists, but its uploaded file could not be found on disk.")

    with tab3:
        st.subheader("Practice with an AI quiz")
        st.caption("Questions are generated from the study notes for this subject.")
        count = st.slider("Number of questions", 3, 10, 5)
        quiz_key = f"quiz_{subject_id}"

        if quiz_key not in st.session_state:
            st.session_state[quiz_key] = None

        if st.button(":material/refresh: Generate quiz", type="primary"):
            with st.spinner("Generating quiz from your notes..."):
                quiz, error = generate_quiz(subject_id, count)
            if error:
                st.error(error)
            else:
                st.session_state[quiz_key] = quiz

        current_quiz = st.session_state.get(quiz_key)
        if current_quiz:
            answers = {}
            with st.form(f"quiz_form_{subject_id}"):
                for i, q in enumerate(current_quiz):
                    with st.container(border=True):
                        st.markdown(f"**Question {i+1}**")
                        st.write(q["question"])
                        answers[i] = st.radio(
                            "Choose an answer",
                            q["options"],
                            key=f"quiz_{subject_id}_{i}",
                            index=None
                        )
                submitted = st.form_submit_button(
                    ":material/check_circle: Submit quiz", type="primary"
                )

            if submitted:
                score = sum(
                    1 for i, q in enumerate(current_quiz)
                    if answers[i] == q["options"][q["answer"]]
                )

                score_col, total_col = st.columns(2)
                score_col.metric("Your score", f"{score}/{len(current_quiz)}")
                total_col.metric(
                    "Result",
                    f"{(score / len(current_quiz) * 100):.0f}%"
                )

                conn = db()
                conn.execute(
                    "INSERT INTO quiz_results(user_id,subject_id,score,total,created_at) VALUES(?,?,?,?,?)",
                    (user["id"], subject_id, score, len(current_quiz), now())
                )
                conn.commit()
                conn.close()

                for i, q in enumerate(current_quiz):
                    if answers[i] != q["options"][q["answer"]]:
                        st.info(
                            f"Q{i+1}: Correct answer: {q['options'][q['answer']]} — {q.get('explanation','')}"
                        )

    with tab4:
        st.subheader("Your learning progress")
        st.caption("A snapshot of your questions and quiz results.")
        conn = db()
        results = conn.execute(
            """
            SELECT q.score, q.total, q.created_at, s.name subject
            FROM quiz_results q
            LEFT JOIN subjects s ON s.id=q.subject_id
            WHERE q.user_id=? ORDER BY q.id DESC
            """,
            (user["id"],)
        ).fetchall()
        total_questions = conn.execute(
            "SELECT COUNT(*) c FROM chat_history WHERE user_id=?",
            (user["id"],)
        ).fetchone()["c"]
        conn.close()

        x, y = st.columns(2)
        x.metric("Questions asked", total_questions)
        if results:
            valid_results = [r for r in results if r["total"] > 0]
            avg = sum((r["score"] / r["total"]) * 100 for r in valid_results) / len(valid_results) if valid_results else 0
            y.metric("Average quiz score", f"{avg:.1f}%")
            st.markdown("### Recent quiz results")
            st.dataframe([dict(r) for r in results], use_container_width=True, hide_index=True)
        else:
            y.metric("Average quiz score", "—")
            st.info("Complete a quiz to see your progress.")

    with tab5:
        st.subheader("Your study history")
        st.caption("Review recent questions and answers from your study sessions.")
        conn = db()
        rows = conn.execute(
            """
            SELECT h.question, h.answer, h.created_at, s.name subject
            FROM chat_history h
            LEFT JOIN subjects s ON s.id=h.subject_id
            WHERE h.user_id=? ORDER BY h.id DESC LIMIT 30
            """,
            (user["id"],)
        ).fetchall()
        conn.close()

        if not rows:
            st.info("No chat history yet.")
        for r in rows:
            with st.expander(f"{r['created_at']} · {r['question']}"):
                st.caption(f"Subject: {r['subject'] or 'Unknown subject'}")
                st.markdown("**Your question**")
                st.write(r["question"])
                st.markdown("**StudyBuddy**")
                st.write(r["answer"])

# ---------------- MAIN ----------------

def main():
    init_db()
    apply_app_theme()

    if "user" not in st.session_state:
        login_page()
        return

    user = st.session_state.user

    with st.sidebar:
        st.title(":material/school: StudyBuddy")
        st.write(f"**{user['name']}**")
        st.caption(f"{user['role'].title()} workspace")
        st.divider()

        if st.button(":material/logout: Log out", width="stretch"):
            del st.session_state["user"]
            st.rerun()

    if user["role"] == "admin":
        admin_page(user)
    else:
        student_page(user)


if __name__ == "__main__":
    main()
