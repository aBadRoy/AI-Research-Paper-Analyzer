"""
AI Research Paper Analyzer
~~~~~~~~~~~~~~~~~~~~~~~~~~
A Flask web application that analyzes research paper PDFs using Google's Gemini API.
Generates summaries, keywords, viva questions, MCQs, and research insights.
Features RAG-based chat with vector search and real-time progress tracking.

:copyright: (c) 2026 by AI Research Paper Analyzer
:license: MIT, see LICENSE for more details.
"""

import json
import logging
import os
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import chromadb
import google.generativeai as genai
import pdfplumber
from chromadb import EmbeddingFunction
from dotenv import load_dotenv
from flask import (
    Flask,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from flask_bcrypt import Bcrypt
from flask_login import (
    LoginManager,
    UserMixin,
    current_user,
    login_required,
    login_user,
    logout_user,
)
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from werkzeug.utils import secure_filename

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class Config:
    SECRET_KEY: str = os.getenv("SECRET_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    DATABASE_PATH: str = os.getenv("DATABASE_PATH", "database/app.db")
    UPLOAD_FOLDER: str = "uploads"
    REPORTS_FOLDER: str = "reports"
    CHROMA_PERSIST_DIR: str = "chroma_db"
    MAX_CONTENT_LENGTH: int = 16 * 1024 * 1024
    ALLOWED_EXTENSIONS: frozenset = frozenset({"pdf"})
    GEMINI_MODEL: str = "models/gemini-2.5-flash"
    GEMINI_EMBEDDING_MODEL: str = "models/gemini-embedding-001"
    MAX_TEXT_LENGTH: int = 30000
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200


if not Config.SECRET_KEY:
    logger.warning("SECRET_KEY is not set. Using a fallback key for development.")
    Config.SECRET_KEY = "dev-fallback-key-change-in-production"

if not Config.GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY is not set. Please set it in the .env file.")

genai.configure(api_key=Config.GEMINI_API_KEY)

app = Flask(__name__)
app.secret_key = Config.SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = Config.MAX_CONTENT_LENGTH
app.config["UPLOAD_FOLDER"] = Config.UPLOAD_FOLDER
app.config["REPORTS_FOLDER"] = Config.REPORTS_FOLDER

bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message_category = "info"

# ---------------------------------------------------------------------------
# Progress tracking (in-memory, keyed by file_id)
# ---------------------------------------------------------------------------
analysis_progress: dict[int, dict] = {}
progress_lock = threading.Lock()


def update_progress(file_id: int, progress: float, step: str, **kwargs) -> None:
    with progress_lock:
        analysis_progress[file_id] = {
            "progress": progress,
            "step": step,
            "status": kwargs.get("status", "processing"),
        }
        if "result_id" in kwargs:
            analysis_progress[file_id]["result_id"] = kwargs["result_id"]
        if "error" in kwargs:
            analysis_progress[file_id]["error"] = kwargs["error"]


# ---------------------------------------------------------------------------
# Gemini embedding function for ChromaDB
# ---------------------------------------------------------------------------
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input):
        result = genai.embed_content(
            model=Config.GEMINI_EMBEDDING_MODEL,
            content=input,
            task_type="retrieval_document",
        )
        emb = result["embedding"]
        if emb and isinstance(emb[0], (int, float)):
            return [emb]
        return emb


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
@dataclass
class User(UserMixin):
    id: int
    username: str
    email: str
    password_hash: str


@contextmanager
def get_db() -> Iterator[sqlite3.Connection]:
    Path(Config.DATABASE_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(Config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                username    TEXT    UNIQUE NOT NULL,
                email       TEXT    UNIQUE NOT NULL,
                password_hash TEXT  NOT NULL
            );

            CREATE TABLE IF NOT EXISTS uploaded_files (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     INTEGER NOT NULL,
                filename    TEXT    NOT NULL,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS analysis_results (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id           INTEGER NOT NULL,
                summary           TEXT,
                keywords          TEXT,
                viva_questions    TEXT,
                mcqs              TEXT,
                research_insights TEXT,
                vector_store_id   TEXT,
                generated_date    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (file_id) REFERENCES uploaded_files(id) ON DELETE CASCADE
            );
        """)
        try:
            conn.execute("ALTER TABLE analysis_results ADD COLUMN vector_store_id TEXT")
        except sqlite3.OperationalError:
            pass
    logger.info("Database initialized successfully.")


init_db()


@login_manager.user_loader
def load_user(user_id: str) -> Optional[User]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE id = ?", (int(user_id),)
        ).fetchone()
    if row:
        return User(row["id"], row["username"], row["email"], row["password_hash"])
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in Config.ALLOWED_EXTENSIONS
    )


def extract_text_from_pdf(pdf_path: str) -> str:
    text_parts: list[str] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
                else:
                    logger.warning("No text found on page %d of %s", page_number, pdf_path)
    except Exception as exc:
        logger.exception("PDF extraction failed for %s", pdf_path)
        raise RuntimeError(f"Failed to extract text from PDF: {exc}") from exc

    full_text = "\n".join(text_parts)
    if not full_text.strip():
        raise RuntimeError("No text content could be extracted from the PDF.")

    logger.info("Successfully extracted %d characters from PDF.", len(full_text))
    return full_text


def call_gemini(prompt: str, text: str) -> str:
    try:
        model = genai.GenerativeModel(Config.GEMINI_MODEL)
        full_prompt = f"{prompt}\n\nResearch Paper Content:\n{text[:Config.MAX_TEXT_LENGTH]}"
        response = model.generate_content(full_prompt)
        logger.info("Gemini API call succeeded (%d chars input).", len(full_prompt))
        return response.text
    except Exception as exc:
        logger.exception("Gemini API call failed.")
        raise RuntimeError(f"AI analysis failed: {exc}") from exc


def generate_summary(text: str) -> str:
    prompt = (
        "You are an expert research analyst. Summarize the uploaded research paper "
        "in a structured format with the following sections:\n"
        "- Introduction\n- Objective\n- Methodology\n- Results\n- Conclusion\n\n"
        "Provide clear, concise points under each heading."
    )
    return call_gemini(prompt, text)


def generate_keywords(text: str) -> str:
    prompt = (
        "Extract the top 20 most important keywords and key phrases from this "
        "research paper. Return them as a clean comma-separated list without numbering."
    )
    return call_gemini(prompt, text)


def generate_viva_questions(text: str) -> str:
    prompt = (
        "Generate 20 comprehensive viva (oral exam) questions based on this research "
        "paper. The questions should test deep understanding of the paper's concepts, "
        "methodology, and findings. Number them from 1 to 20."
    )
    return call_gemini(prompt, text)


def generate_mcqs(text: str) -> str:
    prompt = (
        "Generate 20 multiple-choice questions based on this research paper. "
        "Each question should have exactly 4 options (A, B, C, D) and one correct answer.\n\n"
        "Format each question exactly as:\n"
        "Question: <question text>\n"
        "A) <option A>\nB) <option B>\nC) <option C>\nD) <option D>\n"
        "Correct Answer: <letter>\n"
    )
    return call_gemini(prompt, text)


def generate_insights(text: str) -> str:
    prompt = (
        "You are an expert research reviewer. Analyze the research paper and identify:\n"
        "- Research Problem: What specific problem does this paper address?\n"
        "- Research Gap: What gap in existing literature does this fill?\n"
        "- Methodology: What research methodology was employed?\n"
        "- Limitations: What are the limitations of this study?\n"
        "- Future Scope: What future research directions are suggested?\n\n"
        "Provide detailed analysis under each heading."
    )
    return call_gemini(prompt, text)


def generate_pdf_report(
    filename: str,
    paper_name: str,
    summary: str,
    keywords: str,
    viva: str,
    mcqs: str,
    insights: str,
) -> str:
    pdf_path = os.path.join(Config.REPORTS_FOLDER, filename)
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("AI Research Paper Analyzer — Analysis Report", styles["Title"]))
    story.append(Spacer(1, 14))
    story.append(Paragraph(f"<b>Paper:</b> {paper_name}", styles["Normal"]))
    story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    story.append(Spacer(1, 20))

    sections = [
        ("Summary", summary),
        ("Keywords", keywords),
        ("Viva Questions", viva),
        ("Multiple Choice Questions", mcqs),
        ("Research Insights", insights),
    ]

    for heading, content in sections:
        story.append(Paragraph(f"<b>{heading}</b>", styles["Heading2"]))
        story.append(Spacer(1, 6))
        for paragraph in content.split("\n"):
            if paragraph.strip():
                story.append(Paragraph(paragraph, styles["Normal"]))
        story.append(Spacer(1, 16))

    doc.build(story)
    logger.info("PDF report generated: %s", pdf_path)
    return str(pdf_path)


# ---------------------------------------------------------------------------
# Vector store (RAG)
# ---------------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = None, chunk_overlap: int = None) -> list[str]:
    chunk_size = chunk_size or Config.CHUNK_SIZE
    chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        if end < text_len:
            pos = text.rfind("\n\n", start, end)
            if pos != -1 and pos > start:
                end = pos
            else:
                pos = text.rfind(". ", start, end)
                if pos != -1 and pos > start:
                    end = pos + 1
        chunk = text[start:end].strip()
        if chunk and len(chunk) > 50:
            chunks.append(chunk)
        start = end - chunk_overlap if end < text_len else text_len

    return chunks


def create_vector_store(chunks: list[str], collection_name: str) -> str:
    client = chromadb.PersistentClient(path=Config.CHROMA_PERSIST_DIR)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(
        name=collection_name,
        embedding_function=GeminiEmbeddingFunction(),
    )
    batch_size = 20
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        ids = [f"chunk_{i + j}" for j in range(len(batch))]
        collection.add(documents=batch, ids=ids)
    logger.info("Vector store created: %s (%d chunks)", collection_name, len(chunks))
    return collection_name


def query_vector_store(collection_name: str, question: str, k: int = 5) -> list[str]:
    client = chromadb.PersistentClient(path=Config.CHROMA_PERSIST_DIR)
    try:
        collection = client.get_collection(
            name=collection_name,
            embedding_function=GeminiEmbeddingFunction(),
        )
    except Exception:
        logger.warning("Vector store '%s' not found.", collection_name)
        return []
    results = collection.query(query_texts=[question], n_results=k)
    return results["documents"][0] if results.get("documents") else []


def answer_question(question: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(context_chunks)
    prompt = (
        "You are a research paper assistant. Answer the user's question based ONLY on the "
        "provided research paper context. If the answer cannot be found in the context, "
        "say so politely.\n\n"
        f"Context from the research paper:\n{context}\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    try:
        model = genai.GenerativeModel(Config.GEMINI_MODEL)
        response = model.generate_content(prompt)
        return response.text
    except Exception as exc:
        logger.exception("Gemini chat call failed.")
        raise RuntimeError(f"Failed to answer question: {exc}") from exc


# ---------------------------------------------------------------------------
# Background analysis runner
# ---------------------------------------------------------------------------
def run_analysis(file_id: int, user_id: int) -> None:
    try:
        update_progress(file_id, 2, "Starting analysis...")

        with get_db() as conn:
            file_record = conn.execute(
                "SELECT * FROM uploaded_files WHERE id = ? AND user_id = ?",
                (file_id, user_id),
            ).fetchone()

        if not file_record:
            update_progress(file_id, 0, "error", status="error", error="File not found.")
            return

        filepath = os.path.join(Config.UPLOAD_FOLDER, file_record["filename"])

        update_progress(file_id, 8, "Extracting text from PDF...")
        text = extract_text_from_pdf(filepath)

        update_progress(file_id, 15, "Chunking text into segments...")
        chunks = chunk_text(text)
        logger.info("Text chunked into %d segments.", len(chunks))

        update_progress(file_id, 20, "Building vector search index...")
        collection_name = f"paper_{file_id}"
        create_vector_store(chunks, collection_name)

        update_progress(file_id, 35, "Generating summary...")
        summary = generate_summary(text)

        update_progress(file_id, 48, "Extracting keywords...")
        keywords = generate_keywords(text)

        update_progress(file_id, 58, "Generating viva questions...")
        viva = generate_viva_questions(text)

        update_progress(file_id, 70, "Generating multiple-choice questions...")
        mcqs = generate_mcqs(text)

        update_progress(file_id, 82, "Analyzing research insights...")
        insights = generate_insights(text)

        update_progress(file_id, 92, "Saving results...")

        with get_db() as conn:
            cursor = conn.execute(
                """INSERT INTO analysis_results
                   (file_id, summary, keywords, viva_questions, mcqs, research_insights, vector_store_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (file_id, summary, keywords, viva, mcqs, insights, collection_name),
            )
            result_id = cursor.lastrowid

        update_progress(file_id, 100, "Complete!", status="done", result_id=result_id)
        logger.info("Analysis complete for file_id %d (result_id %d)", file_id, result_id)

    except Exception as exc:
        logger.exception("Analysis failed for file_id %d", file_id)
        update_progress(file_id, 0, "error", status="error", error=str(exc))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username: str = request.form.get("username", "").strip()
        email: str = request.form.get("email", "").strip()
        password: str = request.form.get("password", "")
        confirm: str = request.form.get("confirm_password", "")

        if not username or not email or not password:
            flash("All fields are required.", "danger")
            return render_template("register.html")

        if password != confirm:
            flash("Passwords do not match.", "danger")
            return render_template("register.html")

        if len(password) < 6:
            flash("Password must be at least 6 characters.", "danger")
            return render_template("register.html")

        with get_db() as conn:
            existing = conn.execute(
                "SELECT id FROM users WHERE username = ? OR email = ?",
                (username, email),
            ).fetchone()
            if existing:
                flash("Username or email already exists.", "danger")
                return render_template("register.html")

            pw_hash: str = bcrypt.generate_password_hash(password).decode("utf-8")
            conn.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                (username, email, pw_hash),
            )

        logger.info("New user registered: %s (%s)", username, email)
        flash("Registration successful! Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username: str = request.form.get("username", "").strip()
        password: str = request.form.get("password", "")

        if not username or not password:
            flash("Please enter both username and password.", "danger")
            return render_template("login.html")

        with get_db() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE username = ?", (username,)
            ).fetchone()

        if row and bcrypt.check_password_hash(row["password_hash"], password):
            user = User(row["id"], row["username"], row["email"], row["password_hash"])
            login_user(user)
            logger.info("User logged in: %s", username)
            flash(f"Welcome back, {username}!", "success")
            return redirect(url_for("dashboard"))

        flash("Invalid username or password.", "danger")

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out successfully.", "info")
    return redirect(url_for("index"))


@app.route("/dashboard")
@login_required
def dashboard():
    with get_db() as conn:
        files = conn.execute(
            "SELECT * FROM uploaded_files WHERE user_id = ? ORDER BY upload_date DESC",
            (current_user.id,),
        ).fetchall()

        analyses: list[dict] = []
        for f in files:
            result = conn.execute(
                "SELECT * FROM analysis_results WHERE file_id = ? ORDER BY generated_date DESC LIMIT 1",
                (f["id"],),
            ).fetchone()
            analyses.append({"file": f, "result": result})

    return render_template("dashboard.html", analyses=analyses)


@app.route("/upload", methods=["POST"])
@login_required
def upload_file():
    if "file" not in request.files:
        flash("No file selected. Please choose a PDF to upload.", "danger")
        return redirect(url_for("dashboard"))

    file = request.files["file"]

    if not file.filename or file.filename == "":
        flash("No file selected. Please choose a PDF to upload.", "danger")
        return redirect(url_for("dashboard"))

    if not allowed_file(file.filename):
        flash("Invalid file type. Only PDF files are allowed.", "danger")
        return redirect(url_for("dashboard"))

    original_filename: str = secure_filename(file.filename)
    unique_filename: str = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{original_filename}"
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    filepath: str = os.path.join(Config.UPLOAD_FOLDER, unique_filename)
    file.save(filepath)

    logger.info("PDF uploaded: %s (saved as %s)", original_filename, unique_filename)

    with get_db() as conn:
        cursor = conn.execute(
            "INSERT INTO uploaded_files (user_id, filename) VALUES (?, ?)",
            (current_user.id, unique_filename),
        )
        file_id = cursor.lastrowid

    if file_id is None:
        flash("Failed to save file record. Please try again.", "danger")
        return redirect(url_for("dashboard"))

    return redirect(url_for("processing", file_id=file_id))


@app.route("/processing/<int:file_id>")
@login_required
def processing(file_id):
    with get_db() as conn:
        file_record = conn.execute(
            "SELECT * FROM uploaded_files WHERE id = ? AND user_id = ?",
            (file_id, current_user.id),
        ).fetchone()

        if not file_record:
            flash("File not found.", "danger")
            return redirect(url_for("dashboard"))

        result = conn.execute(
            "SELECT id FROM analysis_results WHERE file_id = ? LIMIT 1",
            (file_id,),
        ).fetchone()
        if result:
            return redirect(url_for("result", result_id=result["id"]))

    with progress_lock:
        already_running = analysis_progress.get(file_id, {}).get("status") == "processing"

    if not already_running:
        thread = threading.Thread(target=run_analysis, args=(file_id, current_user.id), daemon=True)
        thread.start()

    return render_template("processing.html", file_id=file_id)


@app.route("/api/progress/<int:file_id>")
@login_required
def api_progress(file_id):
    with progress_lock:
        data = analysis_progress.get(file_id, {"progress": 0, "step": "Queued...", "status": "processing"})
        return jsonify(data)


@app.route("/api/chat/<int:result_id>", methods=["POST"])
@login_required
def chat_with_paper(result_id):
    data = request.get_json()
    question = (data.get("message", "") if data else "").strip()
    if not question:
        return jsonify({"reply": "Please enter a question."})

    with get_db() as conn:
        analysis = conn.execute(
            """SELECT ar.*, uf.filename AS paper_name
            FROM analysis_results ar
            JOIN uploaded_files uf ON ar.file_id = uf.id
            WHERE ar.id = ? AND uf.user_id = ?""",
            (result_id, current_user.id),
        ).fetchone()

    if not analysis:
        return jsonify({"reply": "Analysis result not found."}), 404

    collection_name = analysis["vector_store_id"] or f"paper_{analysis['file_id']}"

    try:
        relevant_chunks = query_vector_store(collection_name, question)
        if not relevant_chunks:
            return jsonify({
                "reply": "I could not find relevant information in the paper to answer your question. "
                         "Try asking about the paper's content, methodology, or findings."
            })
        answer = answer_question(question, relevant_chunks)
        return jsonify({"reply": answer})
    except Exception as exc:
        logger.exception("Chat failed for result_id %d", result_id)
        return jsonify({"reply": f"Sorry, an error occurred: {exc}"}), 500


@app.route("/result/<int:result_id>")
@login_required
def result(result_id):
    with get_db() as conn:
        analysis = conn.execute(
            """SELECT ar.*, uf.filename AS paper_name
            FROM analysis_results ar
            JOIN uploaded_files uf ON ar.file_id = uf.id
            WHERE ar.id = ? AND uf.user_id = ?""",
            (result_id, current_user.id),
        ).fetchone()

    if not analysis:
        flash("Analysis result not found or access denied.", "danger")
        return redirect(url_for("dashboard"))

    return render_template(
        "result.html",
        analysis=analysis,
        paper_name=analysis["paper_name"],
        summary=analysis["summary"],
        keywords=analysis["keywords"],
        viva=analysis["viva_questions"],
        mcqs=analysis["mcqs"],
        insights=analysis["research_insights"],
        result_id=result_id,
        has_vector_store=bool(analysis["vector_store_id"]),
    )


@app.route("/download_report/<int:result_id>")
@login_required
def download_report(result_id):
    with get_db() as conn:
        analysis = conn.execute(
            """SELECT ar.*, uf.filename AS paper_name
            FROM analysis_results ar
            JOIN uploaded_files uf ON ar.file_id = uf.id
            WHERE ar.id = ? AND uf.user_id = ?""",
            (result_id, current_user.id),
        ).fetchone()

    if not analysis:
        flash("Analysis result not found or access denied.", "danger")
        return redirect(url_for("dashboard"))

    os.makedirs(Config.REPORTS_FOLDER, exist_ok=True)
    pdf_filename: str = f"report_{result_id}.pdf"

    generate_pdf_report(
        pdf_filename,
        analysis["paper_name"],
        analysis["summary"],
        analysis["keywords"],
        analysis["viva_questions"],
        analysis["mcqs"],
        analysis["research_insights"],
    )

    return send_from_directory(
        Config.REPORTS_FOLDER, pdf_filename, as_attachment=True, download_name="analysis_report.pdf"
    )


@app.errorhandler(404)
def not_found(error):
    return render_template("404.html"), 404


@app.errorhandler(413)
def too_large(error):
    flash("File size exceeds the 16 MB limit.", "danger")
    return redirect(url_for("dashboard"))


@app.errorhandler(500)
def server_error(error):
    logger.exception("Internal server error")
    return render_template("500.html"), 500


if __name__ == "__main__":
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(Config.REPORTS_FOLDER, exist_ok=True)
    os.makedirs(Config.CHROMA_PERSIST_DIR, exist_ok=True)
    logger.info("Starting AI Research Paper Analyzer on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True, threaded=True)
