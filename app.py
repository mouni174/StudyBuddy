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

    existing = cur.execute(
        "SELECT id FROM users WHERE email=?",
        ("admin@studybuddy.local",)
    ).fetchone()

    if not existing:
        cur.execute(
            """
            INSERT INTO users(name,email,password_hash,role,created_at)
            VALUES(?,?,?,?,?)
            """,
            ("Administrator", "admin@studybuddy.local", hash_password("Admin@123"), "admin", now())
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

def login_page():
    st.markdown("""
    <style>
    .login-title { text-align:center; font-size:40px; font-weight:800; margin-top:20px; margin-bottom:4px; }
    .login-subtitle { text-align:center; color:#6b7280; margin-bottom:28px; }
    .role-card { border:1px solid #e5e7eb; border-radius:14px; padding:22px; text-align:center; min-height:145px; background:#fff; box-shadow:0 2px 10px rgba(0,0,0,.04); margin-bottom:22px; }
    .role-title { font-size:24px; font-weight:700; margin-bottom:6px; }
    .role-text { color:#6b7280; font-size:15px; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="login-title">🎓 StudyBuddy</div>', unsafe_allow_html=True)
    st.markdown('<div class="login-subtitle">RAG Tutor for College Notes</div>', unsafe_allow_html=True)

    if "auth_role" not in st.session_state:
        st.session_state.auth_role = "student"

    role = st.radio(
        "Select account type",
        ["👨‍💼 Admin", "🎓 Student"],
        horizontal=True,
        label_visibility="collapsed",
        index=0 if st.session_state.auth_role == "admin" else 1,
    )

    selected_role = "admin" if role.startswith("👨‍💼") else "student"
    st.session_state.auth_role = selected_role

    if selected_role == "admin":
        st.markdown("""
        <div class="role-card">
            <div class="role-title">👨‍💼 Administrator</div>
            <div class="role-text">Manage students, subjects, official notes and platform activity.</div>
        </div>
        """, unsafe_allow_html=True)

        login_tab, register_tab = st.tabs(["🔐 Admin Login", "📝 Admin Registration"])

        with login_tab:
            with st.form("admin_login"):
                email = st.text_input("Admin Email", placeholder="admin@example.com")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Login as Admin", use_container_width=True, type="primary")

            if submitted:
                user = authenticate(email, password)
                if user and user["role"] == "admin":
                    st.session_state.user = user
                    st.rerun()
                elif user:
                    st.error("This account is registered as a Student. Please select Student.")
                else:
                    st.error("Invalid admin email or password.")

            st.info("Demo Admin: admin@studybuddy.local / Admin@123")

        with register_tab:
            st.warning("For this college-project demo, admin registration is enabled. In a real deployment, administrator accounts should be created by an existing administrator.")
            with st.form("admin_register"):
                name = st.text_input("Administrator Name")
                email = st.text_input("Admin Email")
                password = st.text_input("Password", type="password")
                confirm = st.text_input("Confirm Password", type="password")
                submitted = st.form_submit_button("Create Admin Account", use_container_width=True, type="primary")

            if submitted:
                if not name or not email or not password:
                    st.error("Please fill all fields.")
                elif password != confirm:
                    st.error("Passwords do not match.")
                elif len(password) < 6:
                    st.error("Password must contain at least 6 characters.")
                else:
                    ok, msg = create_user(name, email, password, role="admin")
                    (st.success if ok else st.error)(msg)

    else:
        st.markdown("""
        <div class="role-card">
            <div class="role-title">🎓 Student</div>
            <div class="role-text">Learn from college notes, ask questions, take quizzes and track progress.</div>
        </div>
        """, unsafe_allow_html=True)

        login_tab, register_tab = st.tabs(["🔐 Student Login", "📝 Student Registration"])

        with login_tab:
            with st.form("student_login"):
                email = st.text_input("Student Email", placeholder="student@example.com")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Login as Student", use_container_width=True, type="primary")

            if submitted:
                user = authenticate(email, password)
                if user and user["role"] == "student":
                    st.session_state.user = user
                    st.rerun()
                elif user:
                    st.error("This account is registered as an Admin. Please select Admin.")
                else:
                    st.error("Invalid student email or password.")

        with register_tab:
            with st.form("student_register"):
                name = st.text_input("Full Name")
                email = st.text_input("Student Email")
                password = st.text_input("Password", type="password")
                confirm = st.text_input("Confirm Password", type="password")
                submitted = st.form_submit_button("Create Student Account", use_container_width=True, type="primary")

            if submitted:
                if not name or not email or not password:
                    st.error("Please fill all fields.")
                elif password != confirm:
                    st.error("Passwords do not match.")
                elif len(password) < 6:
                    st.error("Password must contain at least 6 characters.")
                else:
                    ok, msg = create_user(name, email, password, role="student")
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
    st.title("👨‍💼 Admin Dashboard")
    st.caption("Manage StudyBuddy content, students and platform activity.")

    conn = db()
    students = conn.execute("SELECT COUNT(*) c FROM users WHERE role='student'").fetchone()["c"]
    subjects = conn.execute("SELECT COUNT(*) c FROM subjects").fetchone()["c"]
    notes = conn.execute("SELECT COUNT(*) c FROM documents").fetchone()["c"]
    questions = conn.execute("SELECT COUNT(*) c FROM chat_history").fetchone()["c"]
    quiz_attempts = conn.execute("SELECT COUNT(*) c FROM quiz_results").fetchone()["c"]
    conn.close()

    a, b, c, d, e = st.columns(5)
    a.metric("Students", students)
    b.metric("Subjects", subjects)
    c.metric("Notes", notes)
    d.metric("Questions", questions)
    e.metric("Quiz Attempts", quiz_attempts)

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📚 Subjects", "📄 Notes", "👥 Users", "📊 Student Activity"
    ])

    # Subjects
    with tab1:
        st.subheader("Manage Subjects")
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
            st.markdown("### Existing Subjects")
            for row in rows:
                col1, col2, col3 = st.columns([2, 5, 1])
                col1.write(f"**{row['name']}**")
                col2.write(row["description"] or "No description")
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
        st.subheader("Upload Official Study Notes")
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
                st.markdown("### Uploaded Notes")
                for doc in docs:
                    col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 2, 1])
                    col1.write(f"📄 **{doc['filename']}**")
                    col2.write(doc["subject"])
                    col3.write(doc["uploaded_at"])
                    col4.write(doc["uploaded_by"] or "Unknown")
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
        st.subheader("Manage Students")
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

            st.markdown("### Student Actions")
            student_map = {f"{r['name']} — {r['email']}": r["id"] for r in rows}
            selected_student = st.selectbox("Select a student", list(student_map), key="admin_selected_student")
            selected_student_id = student_map[selected_student]

            action_col1, action_col2 = st.columns(2)
            if action_col1.button("📊 View Student Activity", use_container_width=True):
                st.session_state["activity_student_id"] = selected_student_id
                st.info("Open the 'Student Activity' tab to view this student's questions and quiz performance.")
            if action_col2.button("🗑️ Delete Student", use_container_width=True):
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
        st.subheader("📊 Student Activity & Performance")
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
                st.markdown(f"### 👤 {student['name']}")
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
            st.markdown("### 💬 Questions Asked")
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

            st.markdown("### 📝 Quiz Performance")
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
    # Polished, responsive styling for the student workspace.
    st.markdown(
        """
        <style>
        .stApp { background: linear-gradient(180deg, #f8faff 0%, #ffffff 42%); }
        .sb-hero {
            padding: 1.35rem 1.55rem; border-radius: 18px; margin-bottom: 1rem;
            background: linear-gradient(120deg, #33256f 0%, #5b4acb 58%, #8274ee 100%);
            color: white; box-shadow: 0 8px 24px rgba(60, 48, 130, .15);
        }
        .sb-hero h1 { color: white !important; margin: 0 0 .3rem 0; font-size: 2rem; }
        .sb-hero p { color: #eeecff !important; margin: 0; font-size: 1rem; }
        div[data-testid="stMetric"] {
            background: white; padding: 15px 17px; border: 1px solid #e8eaf4;
            border-radius: 14px; box-shadow: 0 3px 12px rgba(32, 40, 80, .04);
        }
        div.stButton > button, div.stDownloadButton > button {
            border-radius: 10px; font-weight: 600; transition: all .15s ease;
        }
        div.stButton > button:hover, div.stDownloadButton > button:hover {
            border-color: #6657d9; color: #5142bd; transform: translateY(-1px);
        }
        div[data-testid="stTabs"] button { font-weight: 600; }
        div[data-testid="stExpander"] { border-radius: 12px; border-color: #e4e7f1; }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="sb-hero"><h1>🎓 Welcome, {user["name"]}</h1>'
        '<p>Your personal study space — learn, practice, and track your progress.</p></div>',
        unsafe_allow_html=True
    )

    subjects = get_subjects()
    if not subjects:
        st.warning("No subjects or notes have been added by the administrator yet.")
        return

    subject_map = {r["name"]: r["id"] for r in subjects}
    subject_name = st.selectbox("Choose a subject", list(subject_map))
    subject_id = subject_map[subject_name]

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🤖 Ask StudyBuddy", "📥 Download Notes", "📝 Quiz", "📊 My Progress", "💬 History"
    ])

    with tab1:
        question = st.text_area(
            "Ask a question from your notes",
            placeholder="Example: Explain normalization with a simple example.",
            height=120
        )

        if st.button("Ask StudyBuddy", type="primary"):
            if not question.strip():
                st.warning("Enter a question first.")
            else:
                with st.spinner("Searching your notes and generating an answer..."):
                    answer, sources = generate_answer(question, subject_id)

                if answer is None:
                    st.error("Unable to generate an answer.")
                elif isinstance(answer, str) and (
                    answer.startswith("GEMINI_API_KEY")
                    or answer.startswith("Unable to generate")
                    or answer.startswith("Gemini is temporarily")
                    or answer.startswith("The selected Gemini")
                ):
                    st.error(answer)
                else:
                    st.markdown("### Answer")
                    st.write(answer)
                    if sources:
                        st.markdown("### 📚 Sources")
                        for filename, page in sources:
                            st.write(f"- {filename} — Page {page}")

                    conn = db()
                    conn.execute(
                        "INSERT INTO chat_history(user_id,subject_id,question,answer,created_at) VALUES(?,?,?,?,?)",
                        (user["id"], subject_id, question, answer, now())
                    )
                    conn.commit()
                    conn.close()

    with tab2:
        st.subheader("📥 Download Study Notes")
        st.caption(f"Browse and download the official notes uploaded for **{subject_name}**.")

        conn = db()
        note_rows = conn.execute(
            "SELECT id, filename, uploaded_at, stored_path FROM documents WHERE subject_id=? ORDER BY id DESC",
            (subject_id,)
        ).fetchall()
        conn.close()

        if not note_rows:
            st.info("No notes are available for this subject yet. Please check back later.")
        else:
            st.success(f"{len(note_rows)} note file(s) available")
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
                    icon = "📕" if suffix == ".pdf" else ("📝" if suffix == ".docx" else "📄")
                    note_col.markdown(f"**{icon} {file_name}**")
                    note_col.caption(f"Format: {suffix.lstrip('.').upper() or 'FILE'}")
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
                            "⬇️ Download",
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
        st.subheader("AI Quiz")
        count = st.slider("Number of questions", 3, 10, 5)
        quiz_key = f"quiz_{subject_id}"

        if quiz_key not in st.session_state:
            st.session_state[quiz_key] = None

        if st.button("Generate Quiz", type="primary"):
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
                    st.markdown(f"**Q{i+1}. {q['question']}**")
                    answers[i] = st.radio(
                        "Choose an answer",
                        q["options"],
                        key=f"quiz_{subject_id}_{i}",
                        index=None
                    )
                submitted = st.form_submit_button("Submit Quiz")

            if submitted:
                score = sum(
                    1 for i, q in enumerate(current_quiz)
                    if answers[i] == q["options"][q["answer"]]
                )

                st.success(f"Score: {score}/{len(current_quiz)}")

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
        x.metric("Questions Asked", total_questions)
        if results:
            valid_results = [r for r in results if r["total"] > 0]
            avg = sum((r["score"] / r["total"]) * 100 for r in valid_results) / len(valid_results) if valid_results else 0
            y.metric("Average Quiz Score", f"{avg:.1f}%")
            st.subheader("Quiz History")
            st.dataframe([dict(r) for r in results], use_container_width=True, hide_index=True)
        else:
            y.metric("Average Quiz Score", "—")
            st.info("Complete a quiz to see your progress.")

    with tab5:
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
            with st.expander(f"{r['created_at']} — {r['question']}"):
                st.caption(f"Subject: {r['subject']}")
                st.write(r["answer"])

# ---------------- MAIN ----------------

def main():
    init_db()

    if "user" not in st.session_state:
        login_page()
        return

    user = st.session_state.user

    with st.sidebar:
        st.title("🎓 StudyBuddy")
        st.write(f"**{user['name']}**")
        st.caption(user["role"].upper())
        st.divider()

        if st.button("Logout", use_container_width=True):
            del st.session_state["user"]
            st.rerun()

    if user["role"] == "admin":
        admin_page(user)
    else:
        student_page(user)


if __name__ == "__main__":
    main()
