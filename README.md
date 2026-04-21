# ✦ ResumeX — AI Resume Analyzer & Job Portal

> A production-grade SaaS platform that analyzes resumes using a **Hybrid AI Engine** combining Semantic Similarity, Skill Matching, and Experience Analysis.
## 🎯 What is ResumeX?

ResumeX helps job seekers understand **how well their resume matches a job description** — using real AI, not just keyword counting.

**Unlike basic keyword matchers**, ResumeX uses:
- 🧠 **Sentence Transformers** (`all-MiniLM-L6-v2`) for semantic understanding
- 🤖 **Google Gemini 1.5 Flash** for structured coaching feedback
- 📊 **Hybrid scoring** across 3 dimensions with a realistic 50–85% range

---

## ✨ Features

| Feature | Description |
|---|---|
| ⚡ **AI Resume Analyzer** | Upload PDF → get Match Score, ATS Score, skill gap, AI suggestions |
| 📊 **Score Breakdown** | Semantic (40%) + Skill Match (30%) + Experience (30%) |
| 🔍 **Multi-Resume Compare** | Upload 5 resumes × 3 JDs → comparison table |
| 💼 **Job Board** | Browse, search, filter, save, and post jobs |
| 📌 **Application Tracker** | Track jobs: Applied → Interview → Selected |
| 🔒 **Auth System** | JWT + bcrypt secure accounts |
| 📋 **Dashboard** | Stats, history, saved jobs all in one place |

---

## 🧠 How the AI Scoring Works

```
Final Score = (0.40 × Semantic Score) + (0.30 × Skill Score) + (0.30 × Experience Score)
```

| Component | Weight | Method |
|---|---|---|
| Semantic Similarity | 40% | Sentence Transformers cosine similarity |
| Skill Matching | 30% | Normalized skill intersection from 80+ skill DB |
| Experience Matching | 30% | Years, seniority level, section quality heuristic |
| ATS Score | Separate | 6-factor engine: sections, keywords, verbs, numbers, length, contact |

---

## 🛠️ Tech Stack

```
Backend  → FastAPI (Python)
AI       → sentence-transformers (all-MiniLM-L6-v2) + Google Gemini 1.5 Flash
Fallback → scikit-learn TF-IDF
Database → SQLite + SQLAlchemy ORM
Auth     → JWT (python-jose) + bcrypt
PDF      → pdfminer.six
Frontend → Vanilla HTML + CSS + JavaScript (no framework)
Server   → Uvicorn (ASGI)
```

---

## 📁 Project Structure

```
ResumeX_2/
├── backend/
│   ├── main.py          # FastAPI app — all 20+ routes
│   ├── analyzer.py      # Hybrid AI engine (semantic + skill + experience + ATS)
│   ├── auth.py          # JWT creation, bcrypt hashing
│   ├── database.py      # SQLAlchemy setup
│   ├── models.py        # 5 ORM models: User, Job, Analysis, JobHistory, SavedJob
│   └── requirements.txt
│
├── frontend/
│   ├── index.html       # Landing page
│   ├── analyzer.html    # Resume analyzer (single + multi-compare)
│   ├── dashboard.html   # Stats + application tracker
│   ├── jobs.html        # Job board
│   ├── login.html       # Auth page
│   ├── profile.html     # Profile + skills
│   ├── css/
│   │   └── app.css      # Full design system
│   └── js/
│       └── app.js       # Shared utilities (auth, fetch, toasts)
│
├── .gitignore
└── README.md
```

---

## 🚀 Quick Setup

### Prerequisites
- Python 3.10+
- Internet connection (first run downloads ~90MB AI model)
- Optional: [Free Gemini API key](https://aistudio.google.com/app/apikey)

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/ResumeX.git
cd ResumeX

# 2. Create virtual environment
python -m venv venv

# 3. Activate it
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 4. Install dependencies
cd backend
pip install -r requirements.txt

# 5. (Optional) Set Gemini API key for AI feedback
set GEMINI_API_KEY=your_key_here      # Windows CMD
$env:GEMINI_API_KEY="your_key_here"   # PowerShell
export GEMINI_API_KEY=your_key_here   # Mac/Linux

# 6. Run
uvicorn main:app --reload
```

Open **http://localhost:8000**

> **First run note:** The AI model (`all-MiniLM-L6-v2`, ~90MB) downloads automatically on first startup. Takes 1–3 minutes. Subsequent starts are instant.

---

## 🔑 Getting a Free Gemini API Key

1. Go to [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Sign in with Google
3. Click **"Create API Key"**
4. Copy and set: `set GEMINI_API_KEY=your_key_here`
5. Restart server

**Without key:** Uses local hybrid scoring (still accurate, 50–85% range)  
**With key:** Adds Gemini AI coaching: strengths, improvements, ATS tips, "Why this score"

---

## 📡 API Endpoints

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| POST | `/api/register` | ❌ | Register user |
| POST | `/api/login` | ❌ | Login, get JWT |
| GET | `/api/me` | ✅ | Current user stats |
| POST | `/api/analyze` | ✅ | Analyze PDF resume |
| POST | `/api/analyze-multiple` | ✅ | Multi-resume compare |
| GET | `/api/my-analyses` | ✅ | Analysis history |
| GET | `/api/jobs` | ❌ | Job listings |
| POST | `/api/jobs` | ✅ | Post a job |
| POST | `/api/apply-job` | ✅ | Track application |
| GET | `/api/engine-info` | ❌ | Check AI engine status |

### Example Response — `/api/analyze`

```json
{
  "match_score": 74.2,
  "semantic_score": 79.1,
  "skill_score": 66.7,
  "experience_score": 71.0,
  "ats_score": 82.5,
  "resume_skills": ["python", "fastapi", "docker", "sql"],
  "missing_skills": ["aws", "kubernetes"],
  "matched_skills": ["python", "fastapi", "docker"],
  "summary": "Strong Python developer with solid FastAPI expertise...",
  "strengths": ["3 years Python experience", "FastAPI + REST expertise"],
  "improvements": ["Learn AWS or GCP", "Add Kubernetes"],
  "ats_tips": ["Include AWS certification", "Quantify API metrics"],
  "why_this_score": "Semantic content contributed 79% (40% weight)...",
  "ai_powered": true,
  "semantic_engine": "sentence-transformers"
}
```

---

## 🗄️ Database Schema

```
users          → id, username, email, hashed_password, saved_skills, created_at
jobs           → id, title, company, description, location, job_type, salary, skills
analyses       → id, user_id, resume_filename, match_score, ats_score, semantic_score,
                 skill_score, experience_score, resume_skills, job_skills, missing_skills,
                 ai_summary, strengths, improvements, ats_tips, ai_powered
job_history    → id, user_id, company, role, match_score, status, notes, applied_at
saved_jobs     → id, user_id, job_id, created_at
```

---

## ⚙️ Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | Optional | Google Gemini API key for AI feedback |
| `HF_HUB_DISABLE_SYMLINKS_WARNING` | Optional | Set to `1` to suppress Windows warning |

---

## 🐛 Common Issues

| Error | Fix |
|---|---|
| `Could not import module "main"` | Run from `backend/` folder: `cd backend && uvicorn main:app --reload` |
| CSS not loading / plain HTML | Hard refresh: `Ctrl+Shift+R` |
| Model download stuck | Don't press Ctrl+C — wait 1–3 min for 90MB download |
| Scores very low (5–10%) | You have old `analyzer.py` — replace with hybrid version |
| Port 8000 in use | `uvicorn main:app --reload --port 8001` |

---

## 🔮 Roadmap

- [ ] Real job search API (JSearch / Adzuna integration)
- [ ] Resume PDF export with AI improvements applied
- [ ] AI cover letter generator
- [ ] Resume bullet point rewriter
- [ ] PostgreSQL support for production deployment
- [ ] Email notifications for new matching jobs
- [ ] Docker containerization

---

## 👩‍💻 Built By

**Palak Khare** — CS Undergraduate @ LNCTS Indore  
📧 palakkhare1902@gmail.com  
🔗 [GitHub](https://github.com/palakkhare02) | [LinkedIn](https://linkedin.com/in/palakkhare02) | [Portfolio](https://palakkhare02.github.io)

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

<div align="center">
<strong>⭐ Star this repo if it helped you!</strong>
</div>
