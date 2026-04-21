from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional
import json, os

from database import engine, get_db
import models, auth
from analyzer import extract_text_from_pdf, analyze_resume, analyze_multiple, extract_skills, compute_ats_score

models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="ResumeX — AI Resume Analyzer", version="2.0.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

app.mount("/static", StaticFiles(directory="../frontend"), name="static")
templates = Jinja2Templates(directory="../frontend")

# ── Page Routes ───────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def home(r: Request): return templates.TemplateResponse(r, "index.html")

@app.get("/login", response_class=HTMLResponse)
def login_pg(r: Request): return templates.TemplateResponse(r, "login.html")

@app.get("/dashboard", response_class=HTMLResponse)
def dash_pg(r: Request): return templates.TemplateResponse(r, "dashboard.html")

@app.get("/jobs", response_class=HTMLResponse)
def jobs_pg(r: Request): return templates.TemplateResponse(r, "jobs.html")

@app.get("/analyzer", response_class=HTMLResponse)
def analyzer_pg(r: Request): return templates.TemplateResponse(r, "analyzer.html")

@app.get("/profile", response_class=HTMLResponse)
def profile_pg(r: Request): return templates.TemplateResponse(r, "profile.html")

# ── Auth Routes ───────────────────────────────────────────────────────────────

@app.post("/api/register")
def register(username: str = Form(...), email: str = Form(...), password: str = Form(...),
             db: Session = Depends(get_db)):
    if len(password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters")
    if db.query(models.User).filter(models.User.username == username).first():
        raise HTTPException(400, "Username already taken")
    if db.query(models.User).filter(models.User.email == email).first():
        raise HTTPException(400, "Email already registered")
    u = models.User(username=username, email=email, hashed_password=auth.hash_password(password))
    db.add(u); db.commit(); db.refresh(u)
    token = auth.create_access_token({"sub": u.username})
    return {"access_token": token, "token_type": "bearer", "username": u.username, "email": u.email}

@app.post("/api/login")
def login(fd: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    u = db.query(models.User).filter(models.User.username == fd.username).first()
    if not u or not auth.verify_password(fd.password, u.hashed_password):
        raise HTTPException(401, "Invalid username or password")
    token = auth.create_access_token({"sub": u.username})
    return {"access_token": token, "token_type": "bearer", "username": u.username, "email": u.email}

@app.get("/api/me")
def me(cu: models.User = Depends(auth.get_current_user)):
    last = cu.analyses[-1] if cu.analyses else None
    skills = json.loads(last.resume_skills) if last else []
    saved_skills = json.loads(cu.saved_skills) if cu.saved_skills else []
    if saved_skills:
        skills = saved_skills
    best = max((a.match_score for a in cu.analyses), default=0)
    avg = sum(a.match_score for a in cu.analyses) / len(cu.analyses) if cu.analyses else 0
    return {
        "id": cu.id, "username": cu.username, "email": cu.email,
        "skills": skills, "saved_skills": saved_skills,
        "total_analyses": len(cu.analyses),
        "best_score": round(best, 1),
        "avg_score": round(avg, 1),
        "joined": cu.created_at.isoformat(),
    }

@app.post("/api/profile/skills")
def save_skills(skills: str = Form(...), cu: models.User = Depends(auth.get_current_user),
                db: Session = Depends(get_db)):
    skill_list = [s.strip() for s in skills.split(",") if s.strip()]
    cu.saved_skills = json.dumps(skill_list)
    db.commit()
    return {"ok": True, "skills": skill_list}

# ── Analysis Routes ───────────────────────────────────────────────────────────

@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    job_desc: str = Form(default=""),
    db: Session = Depends(get_db),
    cu: models.User = Depends(auth.get_current_user)
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")
    
    file_bytes = await file.read()
    if len(file_bytes) > 10 * 1024 * 1024:
        raise HTTPException(400, "File size exceeds 10MB limit")
    
    resume_text = extract_text_from_pdf(file_bytes)
    if not resume_text.strip():
        raise HTTPException(400, "Could not extract text from PDF. Please ensure it's a text-based PDF.")
    
    result = await analyze_resume(resume_text, job_desc)
    
    rec = models.Analysis(
        user_id=cu.id,
        resume_filename=file.filename,
        resume_text=resume_text[:5000],
        match_score=result["match_score"],
        ats_score=result.get("ats_score", 0),
        resume_skills=json.dumps(result.get("resume_skills", [])),
        job_skills=json.dumps(result.get("job_skills", [])),
        missing_skills=json.dumps(result.get("missing_skills", [])),
        ai_summary=result.get("summary", ""),
        strengths=json.dumps(result.get("strengths", [])),
        improvements=json.dumps(result.get("improvements", [])),
        ats_tips=json.dumps(result.get("ats_tips", [])),
        experience_match=result.get("experience_match", ""),
        education_match=result.get("education_match", ""),
        ai_powered=result.get("ai_powered", False),
        job_desc_used=job_desc[:500],
    )
    db.add(rec); db.commit(); db.refresh(rec)
    return {**result, "analysis_id": rec.id, "message": "Analysis complete ✅"}

@app.post("/api/analyze-multiple")
async def analyze_multiple_route(
    files: List[UploadFile] = File(...),
    job_descs: str = Form(default="[]"),
    db: Session = Depends(get_db),
    cu: models.User = Depends(auth.get_current_user)
):
    if len(files) > 5:
        raise HTTPException(400, "Maximum 5 resumes at a time")
    
    jd_list = json.loads(job_descs)
    if not jd_list:
        raise HTTPException(400, "Provide at least one job description")
    
    resumes = []
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            continue
        text = extract_text_from_pdf(await f.read())
        if text.strip():
            resumes.append((f.filename, text))
    
    if not resumes:
        raise HTTPException(400, "No valid PDF resumes found")
    
    jd_pairs = [(jd.get("name", f"JD {i+1}"), jd.get("text", "")) for i, jd in enumerate(jd_list)]
    results = await analyze_multiple(resumes, jd_pairs)
    return {"results": results, "resume_count": len(resumes), "jd_count": len(jd_pairs)}

@app.get("/api/my-analyses")
def my_analyses(cu: models.User = Depends(auth.get_current_user), db: Session = Depends(get_db)):
    rows = db.query(models.Analysis).filter(
        models.Analysis.user_id == cu.id
    ).order_by(models.Analysis.created_at.desc()).all()
    return [{
        "id": a.id,
        "resume_filename": a.resume_filename,
        "match_score": a.match_score,
        "ats_score": a.ats_score or 0,
        "resume_skills": json.loads(a.resume_skills),
        "job_skills": json.loads(a.job_skills),
        "missing_skills": json.loads(a.missing_skills),
        "ai_summary": a.ai_summary or "",
        "strengths": json.loads(a.strengths) if a.strengths else [],
        "improvements": json.loads(a.improvements) if a.improvements else [],
        "ats_tips": json.loads(a.ats_tips) if a.ats_tips else [],
        "experience_match": a.experience_match or "",
        "education_match": a.education_match or "",
        "ai_powered": a.ai_powered,
        "created_at": a.created_at.isoformat(),
    } for a in rows]

@app.delete("/api/analyses/{analysis_id}")
def delete_analysis(analysis_id: int, cu: models.User = Depends(auth.get_current_user),
                    db: Session = Depends(get_db)):
    rec = db.query(models.Analysis).filter(
        models.Analysis.id == analysis_id, models.Analysis.user_id == cu.id
    ).first()
    if not rec:
        raise HTTPException(404, "Analysis not found")
    db.delete(rec); db.commit()
    return {"ok": True}

# ── Job Tracker Routes ────────────────────────────────────────────────────────

@app.post("/api/apply-job")
def apply_job(company: str = Form(...), role: str = Form(...),
              match_score: float = Form(default=0), status: str = Form(default="Applied"),
              notes: str = Form(default=""),
              cu: models.User = Depends(auth.get_current_user), db: Session = Depends(get_db)):
    rec = models.JobHistory(user_id=cu.id, company=company, role=role,
                            match_score=match_score, status=status, notes=notes)
    db.add(rec); db.commit(); db.refresh(rec)
    return {"id": rec.id, "company": rec.company, "role": rec.role, "status": rec.status}

@app.get("/api/job-history")
def job_history(cu: models.User = Depends(auth.get_current_user), db: Session = Depends(get_db)):
    rows = db.query(models.JobHistory).filter(
        models.JobHistory.user_id == cu.id
    ).order_by(models.JobHistory.applied_at.desc()).all()
    return [{"id": a.id, "company": a.company, "role": a.role,
             "match_score": a.match_score, "status": a.status,
             "notes": a.notes or "", "applied_at": a.applied_at.isoformat()} for a in rows]

@app.put("/api/update-status/{app_id}")
def update_status(app_id: int, status: str = Form(...),
                  cu: models.User = Depends(auth.get_current_user), db: Session = Depends(get_db)):
    rec = db.query(models.JobHistory).filter(
        models.JobHistory.id == app_id, models.JobHistory.user_id == cu.id
    ).first()
    if not rec:
        raise HTTPException(404, "Application not found")
    rec.status = status; db.commit()
    return {"ok": True, "status": status}

@app.delete("/api/job-history/{app_id}")
def delete_app(app_id: int, cu: models.User = Depends(auth.get_current_user),
               db: Session = Depends(get_db)):
    rec = db.query(models.JobHistory).filter(
        models.JobHistory.id == app_id, models.JobHistory.user_id == cu.id
    ).first()
    if not rec:
        raise HTTPException(404, "Not found")
    db.delete(rec); db.commit()
    return {"ok": True}

# ── Job Listings Routes ───────────────────────────────────────────────────────

@app.get("/api/jobs")
def list_jobs(q: Optional[str] = None, job_type: Optional[str] = None,
              db: Session = Depends(get_db)):
    query = db.query(models.Job)
    if q:
        query = query.filter(
            models.Job.title.contains(q) | models.Job.company.contains(q) |
            models.Job.description.contains(q)
        )
    if job_type:
        query = query.filter(models.Job.job_type == job_type)
    jobs = query.order_by(models.Job.created_at.desc()).all()
    return [{"id": j.id, "title": j.title, "company": j.company,
             "location": j.location or "Remote", "job_type": j.job_type or "Full-time",
             "salary": j.salary or "", "description": j.description,
             "skills": json.loads(j.skills) if j.skills else [],
             "created_at": j.created_at.isoformat()} for j in jobs]

@app.post("/api/jobs")
def create_job(title: str = Form(...), company: str = Form(...), description: str = Form(...),
               location: str = Form(default="Remote"), job_type: str = Form(default="Full-time"),
               salary: str = Form(default=""),
               cu: models.User = Depends(auth.get_current_user), db: Session = Depends(get_db)):
    j = models.Job(title=title, company=company, description=description, location=location,
                   job_type=job_type, salary=salary, posted_by=cu.id,
                   skills=json.dumps(extract_skills(description)))
    db.add(j); db.commit(); db.refresh(j)
    return {"id": j.id, "title": j.title, "message": "Job posted successfully"}

@app.post("/api/jobs/{job_id}/save")
def save_job(job_id: int, cu: models.User = Depends(auth.get_current_user),
             db: Session = Depends(get_db)):
    ex = db.query(models.SavedJob).filter(
        models.SavedJob.user_id == cu.id, models.SavedJob.job_id == job_id
    ).first()
    if ex:
        db.delete(ex); db.commit(); return {"saved": False}
    db.add(models.SavedJob(user_id=cu.id, job_id=job_id)); db.commit()
    return {"saved": True}

@app.get("/api/saved-jobs")
def saved_jobs(cu: models.User = Depends(auth.get_current_user), db: Session = Depends(get_db)):
    saved = db.query(models.SavedJob).filter(models.SavedJob.user_id == cu.id).all()
    ids = [s.job_id for s in saved]
    jobs = db.query(models.Job).filter(models.Job.id.in_(ids)).all()
    return [{"id": j.id, "title": j.title, "company": j.company,
             "location": j.location or "Remote", "job_type": j.job_type or "Full-time"} for j in jobs]

# ── Seed sample jobs on startup ───────────────────────────────────────────────

@app.on_event("startup")
def seed_jobs():
    db = next(get_db())
    if db.query(models.Job).count() == 0:
        sample_jobs = [
            {"title": "Senior Python Developer", "company": "TechCorp India", "location": "Bangalore",
             "job_type": "Full-time", "salary": "₹18-28 LPA",
             "description": "We are looking for a Senior Python Developer with expertise in FastAPI, Django, and microservices architecture. You will design scalable REST APIs, work with PostgreSQL and Redis, and deploy on AWS. Required skills: Python, FastAPI, Django, PostgreSQL, Redis, Docker, AWS, Git, Agile. Experience with machine learning pipelines is a plus."},
            {"title": "Full Stack Engineer", "company": "StartupXYZ", "location": "Remote",
             "job_type": "Full-time", "salary": "₹12-20 LPA",
             "description": "Join our growing team as a Full Stack Engineer. You'll work with React, Node.js, and Python to build our SaaS platform. Must know: React, TypeScript, Node.js, Express, MongoDB, REST API, Git, CI/CD. Experience with Next.js and GraphQL is a bonus."},
            {"title": "Data Scientist", "company": "Analytics Pro", "location": "Hyderabad",
             "job_type": "Full-time", "salary": "₹15-25 LPA",
             "description": "We need a Data Scientist to build ML models and data pipelines. Required: Python, pandas, numpy, scikit-learn, TensorFlow or PyTorch, SQL, Tableau or Power BI. Strong background in statistics, regression, classification, and NLP preferred."},
            {"title": "DevOps Engineer", "company": "CloudSolutions", "location": "Pune",
             "job_type": "Full-time", "salary": "₹14-22 LPA",
             "description": "Build and maintain our cloud infrastructure. Skills needed: Docker, Kubernetes, AWS or Azure or GCP, Terraform, Ansible, CI/CD pipelines, Linux, Git, monitoring tools. Knowledge of Python scripting and microservices architecture is required."},
            {"title": "React Frontend Developer", "company": "DesignFirst Co.", "location": "Mumbai",
             "job_type": "Full-time", "salary": "₹10-18 LPA",
             "description": "Create beautiful user interfaces with React. Requirements: React, TypeScript, HTML, CSS, JavaScript, REST API integration, Git, Figma. Experience with Next.js, GraphQL, and performance optimization is valued."},
            {"title": "ML Engineer Intern", "company": "AI Startup", "location": "Remote",
             "job_type": "Internship", "salary": "₹25,000/month",
             "description": "6-month internship for ML enthusiasts. Skills: Python, machine learning, deep learning, pandas, numpy, scikit-learn, basic NLP. Exposure to PyTorch or TensorFlow preferred. Good problem solving and communication skills required."},
            {"title": "Backend Developer (Go)", "company": "FinTech Solutions", "location": "Gurgaon",
             "job_type": "Full-time", "salary": "₹20-30 LPA",
             "description": "Build high-performance financial APIs using Go. Required: Go, REST API, microservices, PostgreSQL, Redis, Docker, Kubernetes, AWS. Experience with payment gateways and security best practices is a plus."},
            {"title": "UI/UX Designer", "company": "Creative Agency", "location": "Bangalore",
             "job_type": "Full-time", "salary": "₹8-14 LPA",
             "description": "Design stunning user experiences for web and mobile apps. Skills: Figma, Photoshop, Illustrator, wireframing, prototyping, UX research, HTML, CSS basics. Portfolio demonstrating product design work required."},
        ]
        for job_data in sample_jobs:
            j = models.Job(**job_data, skills=json.dumps(extract_skills(job_data["description"])))
            db.add(j)
        db.commit()
    db.close()

# ── Add this endpoint to main.py (after the existing routes) ──────────────────

@app.get("/api/engine-info")
def engine_info():
    """Returns what semantic engine is loaded."""
    from analyzer import SEMANTIC_ENGINE
    return {
        "semantic_engine": SEMANTIC_ENGINE,
        "gemini_enabled": bool(os.getenv("GEMINI_API_KEY", "")),
        "scoring": {
            "semantic_weight": "40%",
            "skill_weight": "30%",
            "experience_weight": "30%"
        }
    }    