<div align="center">
  <br>
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Flask-3.0-black?style=for-the-badge&logo=flask&logoColor=white" alt="Flask">
  <img src="https://img.shields.io/badge/ChromaDB-5C5CFF?style=for-the-badge&logo=chroma&logoColor=white" alt="ChromaDB">
  <img src="https://img.shields.io/badge/Gemini%20AI-4285F4?style=for-the-badge&logo=google&logoColor=white" alt="Gemini AI">
  <img src="https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white" alt="SQLite">
  <br>
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="License">
  <img src="https://img.shields.io/badge/status-production-green?style=flat-square" alt="Status">

  <br><br>

  <h1>AI Research Paper Analyzer</h1>
  <p>
    <strong>Upload a research paper &middot; Get instant AI analysis &middot; Chat with your paper</strong>
  </p>
  <p>
    A full-stack web application powered by Google Gemini AI with RAG-based chat, real-time progress tracking, and vector search.
  </p>
  <br>
</div>

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [RAG Chat System](#-rag-chat-system)
- [API Reference](#api-reference)
- [Security](#security)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

**AI Research Paper Analyzer** is a production-ready web application that helps students and researchers understand academic papers faster. Upload a PDF, and the system leverages **Google Gemini AI** with **Retrieval-Augmented Generation (RAG)** to deliver:

- Structured summaries with key insights
- Interactive chat — ask questions about the paper
- Keyword extraction and research analysis
- Practice questions (viva & MCQ generation)
- Professional PDF report downloads

---

## Features

### AI-Powered Analysis
| Feature | Description |
|---------|-------------|
| **Summary** | Structured breakdown: Introduction, Objective, Methodology, Results, Conclusion |
| **Keywords** | Top 20 important terms extracted from the paper |
| **Viva Questions** | 20 oral exam-style questions testing deep understanding |
| **MCQs** | 20 multiple-choice questions with correct answers |
| **Research Insights** | Problem, Gap, Methodology, Limitations, Future Scope |

### RAG Chat System
- **Conversational Q&A** — Ask follow-up questions about any paper
- **Vector Search** — ChromaDB stores chunk embeddings for semantic retrieval
- **Context-Aware Answers** — Only relevant sections are sent to Gemini for accurate responses
- **Real-time Streaming UI** — Typing indicators and smooth message delivery

### Real-Time Progress Tracking
- **Live Progress Bar** — Animated progress with step descriptions
- **Background Processing** — Analysis runs in a separate thread, non-blocking
- **Step-by-step Updates** — See each phase: extracting → chunking → embedding → analyzing

### User Experience
- Secure registration and login with bcrypt hashing
- Drag-and-drop PDF upload with validation
- Analysis history with one-click revisit
- Professional PDF report download (ReportLab)
- Responsive Bootstrap 5 design
- Custom error pages (404, 413, 500)

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3.8+, Flask 3.0 |
| **Frontend** | HTML5, CSS3, Bootstrap 5.3 |
| **Database** | SQLite 3 (WAL mode) |
| **Vector Store** | ChromaDB (persistent, with Gemini embeddings) |
| **AI Engine** | Google Gemini API (`gemini-2.5-flash`, `gemini-embedding-001`) |
| **PDF Parsing** | pdfplumber |
| **PDF Generation** | ReportLab |
| **Authentication** | Flask-Bcrypt, Flask-Login |
| **Security** | Werkzeug, python-dotenv |

---

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│   Browser   │────▶│   Flask App  │────▶│   SQLite DB │     │   ChromaDB   │
│  (Bootstrap) │     │   (Python)   │     │             │     │  (Vectors)   │
└─────────────┘     └──────┬───────┘     └─────────────┘     └──────────────┘
                           │                                         ▲
                    ┌──────▼───────┐                                  │
                    │  Google Gemini│──────────────────────────────────┘
                    │     API      │  Embeddings + Generation
                    └──────────────┘
```

1. User uploads a PDF through the browser
2. Flask saves the file and spawns a background analysis thread
3. The processing page polls a progress endpoint for real-time updates
4. **pdfplumber** extracts text, which is chunked and embedded via Gemini
5. Embeddings are stored in **ChromaDB** for semantic search
6. Text is sent to **Gemini API** for summary, keywords, questions, and insights
7. Results are stored in **SQLite** and displayed in the dashboard
8. Users can **chat** with the paper — questions are answered using RAG retrieval
9. A professional **PDF report** can be downloaded at any time

---

## Project Structure

```
AI-RESEARCH-PAPER-ANALYZER/
│
├── app.py                      # Main Flask application
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
├── .gitignore                  # Git ignore rules
├── LICENSE                     # MIT License
├── README.md                   # Project documentation
│
├── database/
│   └── .gitkeep                # SQLite database directory
│
├── uploads/
│   └── .gitkeep                # Uploaded PDFs directory
│
├── reports/
│   └── .gitkeep                # Generated PDF reports directory
│
├── chroma_db/                  # ChromaDB persistent storage (gitignored)
│
├── static/
│   └── style.css               # Custom styles including chat UI
│
├── templates/
│   ├── base.html               # Base template with navigation
│   ├── index.html              # Landing page
│   ├── login.html              # Login form
│   ├── register.html           # Registration form
│   ├── dashboard.html          # User dashboard with upload & history
│   ├── processing.html         # Real-time progress bar page
│   ├── result.html             # Analysis results + chat interface
│   ├── 404.html                # Not found error page
│   └── 500.html                # Server error page
│
└── screenshots/                # Application screenshots
    ├── home.png
    ├── login.png
    ├── register.png
    ├── dashboard.png
    └── results.png
```

---

## Installation

### Prerequisites

- **Python 3.8+** — [Download Python](https://www.python.org/downloads/)
- **pip** — Python package manager (comes with Python)
- **Google Gemini API Key** — [Get one free](https://aistudio.google.com/app/apikey)

### Step 1: Clone the Repository

```bash
git clone https://github.com/aBadRoy/AI-Research-Paper-Analyzer.git
cd AI-Research-Paper-Analyzer
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your values:

```ini
GEMINI_API_KEY=your_actual_gemini_api_key_here
SECRET_KEY=your_random_secret_key_here
```

> **Tip:** Generate a secure SECRET_KEY with:
> ```bash
> python -c "import secrets; print(secrets.token_hex(32))"
> ```

### Step 5: Run the Application

```bash
python app.py
```

Open your browser and navigate to: **http://127.0.0.1:5000**

---

## Usage

### 1. Register an Account
- Navigate to `/register`
- Enter a username, email, and password (minimum 6 characters)

### 2. Log In
- Use your credentials to log in at `/login`

### 3. Upload a Research Paper
- From the dashboard, click "Choose File" and select a PDF
- Click "Upload & Analyze"
- **Watch the real-time progress bar** showing each analysis step

### 4. View Results
- The results page displays five analysis sections in organized cards:
  - **Summary** — Structured breakdown of the paper
  - **Keywords** — Top 20 extracted terms
  - **Viva Questions** — Oral exam practice questions
  - **Multiple Choice Questions** — Test your understanding
  - **Research Insights** — Deep analysis of contributions and limitations

### 5. Chat with Your Paper
- Expand the **"Chat with this Paper"** section at the bottom
- Ask questions like:
  - "What methodology was used in this paper?"
  - "What are the key findings?"
  - "Explain the experiment design"
  - "What are the limitations mentioned?"
- Answers are generated using **RAG** — only the most relevant sections are retrieved

### 6. Download Report
- Click "Download PDF Report" to get a complete analysis document

### 7. Review History
- All past analyses are listed in the dashboard table
- Click "View" to revisit any previous result

---

## 🤖 RAG Chat System

The chat feature uses **Retrieval-Augmented Generation** to provide accurate, context-aware answers:

### How It Works

1. **Text Chunking** — Extracted PDF text is split into overlapping chunks (~1000 chars each) at natural boundaries
2. **Embedding** — Each chunk is embedded using Google's `gemini-embedding-001` model (3072 dimensions)
3. **Vector Storage** — Embeddings are stored in a persistent **ChromaDB** collection
4. **Semantic Search** — When a question is asked, it's embedded and matched against stored chunks
5. **Context Assembly** — The top-5 most relevant chunks are combined as context
6. **Answer Generation** — Gemini generates an answer using only the retrieved context

### Benefits

- ✅ **No hallucination** — Answers are grounded in the actual paper content
- ✅ **Fast retrieval** — Semantic search finds relevant sections instantly
- ✅ **Persistent storage** — ChromaDB saves vectors to disk between sessions
- ✅ **Scalable** — Handles papers of any length efficiently

---

## API Reference

### Web Routes

| Route | Method | Auth | Description |
|-------|--------|------|-------------|
| `/` | GET | No | Landing page |
| `/register` | GET, POST | No | User registration |
| `/login` | GET, POST | No | User login |
| `/logout` | GET | Yes | Log out |
| `/dashboard` | GET | Yes | User dashboard |
| `/upload` | POST | Yes | Upload PDF |
| `/processing/<file_id>` | GET | Yes | Real-time progress view |
| `/result/<id>` | GET | Yes | View analysis results |
| `/download_report/<id>` | GET | Yes | Download PDF report |

### API Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/progress/<file_id>` | GET | Yes | Poll analysis progress |
| `/api/chat/<result_id>` | POST | Yes | Chat with paper (RAG) |

### Chat API Format

**Request:**
```json
{
  "message": "What methodology was used in this paper?"
}
```

**Response:**
```json
{
  "reply": "The paper employs a transformer-based architecture..."
}
```

### Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 302 | Redirect (e.g., after login) |
| 404 | Page not found |
| 413 | File too large (>16 MB) |
| 500 | Internal server error |

---

## Security

| Measure | Implementation |
|---------|---------------|
| **Password Storage** | bcrypt hashing (via Flask-Bcrypt) |
| **Session Management** | Flask-Login with secure cookies |
| **File Validation** | Extension whitelist (PDF only) |
| **File Size Limit** | 16 MB maximum upload |
| **Filename Sanitization** | Werkzeug `secure_filename` |
| **SQL Injection** | Parameterized queries (not f-strings) |
| **CSRF Protection** | Flask's built-in protection |
| **Environment Variables** | API keys never hard-coded |
| **Error Handling** | Custom error pages (404, 413, 500) |

---

## Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

Please ensure your code follows the existing style and includes appropriate documentation.

---

## License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more information.

---

<div align="center">
  <br>
  <p>
    <sub>Built with ❤️ for researchers and students everywhere</sub>
  </p>
  <p>
    <a href="#">Back to Top</a>
  </p>
</div>
