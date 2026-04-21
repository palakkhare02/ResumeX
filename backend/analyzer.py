"""
analyzer.py — Production AI Resume Analyzer
============================================
Hybrid scoring engine:
  - Semantic Similarity  → 40% (sentence-transformers / TF-IDF fallback)
  - Skill Matching       → 30% (normalized skill intersection)
  - Experience Matching  → 30% (heuristic section analysis)

AI layer: Google Gemini 1.5 Flash (structured JSON prompt)
Fallback:  Full local scoring when Gemini unavailable
"""

import os, json, re, tempfile, math
from typing import Tuple

from pdfminer.high_level import extract_text as pdf_extract
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

# ── Optional: sentence-transformers (install: pip install sentence-transformers)
try:
    from sentence_transformers import SentenceTransformer, util as st_util
    _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    SEMANTIC_ENGINE = "sentence-transformers"
except Exception:
    _ST_MODEL = None
    SEMANTIC_ENGINE = "tfidf"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ── Expanded Skills Database ──────────────────────────────────────────────────
SKILLS_DB = [
    # Languages
    "python","java","c++","c#","javascript","typescript","go","golang","rust",
    "ruby","php","kotlin","swift","r","scala","matlab","dart","perl","bash",
    # Web
    "html","css","react","angular","vue","svelte","next.js","nuxt","gatsby",
    "tailwind","bootstrap","sass","webpack","vite",
    # Backend
    "django","flask","fastapi","node","express","spring","spring boot",
    "hibernate","laravel","rails","asp.net","nestjs",
    # API / Arch
    "rest","graphql","grpc","microservices","api","websocket","oauth","jwt",
    # ML / AI / Data
    "machine learning","deep learning","nlp","computer vision","llm",
    "data analysis","data science","statistics","pandas","numpy",
    "scikit-learn","tensorflow","pytorch","keras","opencv","huggingface",
    "langchain","regression","classification","clustering","neural network",
    "transformers","bert","gpt","stable diffusion","reinforcement learning",
    # Data Engineering
    "sql","mysql","postgresql","mongodb","redis","sqlite","cassandra",
    "elasticsearch","neo4j","hadoop","spark","kafka","airflow","dbt",
    "tableau","power bi","looker","excel","bigquery","snowflake",
    # DevOps / Cloud
    "docker","kubernetes","aws","azure","gcp","terraform","ansible",
    "ci/cd","jenkins","github actions","linux","nginx","helm","prometheus",
    "grafana","serverless","lambda","firebase","supabase",
    # Tools / Practices
    "git","agile","scrum","jira","figma","photoshop","illustrator",
    "postman","swagger","junit","pytest","jest","cypress","selenium",
    "maven","gradle","npm","pip","poetry",
    # Soft Skills (weighted lower)
    "communication","leadership","problem solving","teamwork","mentoring",
]

# Synonyms — map variations to canonical skill names
SKILL_SYNONYMS = {
    "golang": "go",
    "node.js": "node",
    "nodejs": "node",
    "reactjs": "react",
    "react.js": "react",
    "vuejs": "vue",
    "vue.js": "vue",
    "next": "next.js",
    "k8s": "kubernetes",
    "ml": "machine learning",
    "ai": "machine learning",
    "dl": "deep learning",
    "tf": "tensorflow",
    "pg": "postgresql",
    "postgres": "postgresql",
    "mongo": "mongodb",
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "cpp": "c++",
    "csharp": "c#",
}

ACTION_VERBS = [
    "developed","built","led","managed","improved","created","designed",
    "implemented","achieved","increased","reduced","delivered","launched",
    "architected","automated","optimized","deployed","migrated","integrated",
    "trained","mentored","collaborated","scaled","refactored","analyzed",
    "engineered","established","coordinated","streamlined","accelerated",
]

SECTION_KEYWORDS = {
    "experience":  ["experience","work history","employment","professional background","career"],
    "education":   ["education","academic","degree","university","college","school","qualification"],
    "skills":      ["skills","technical skills","competencies","technologies","expertise","tools"],
    "projects":    ["projects","portfolio","open source","github","personal projects"],
    "summary":     ["summary","objective","profile","about me","overview","professional summary"],
    "achievements":["achievements","accomplishments","awards","certifications","honors"],
}


# ── 1. PDF Text Extraction ────────────────────────────────────────────────────

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract plain text from PDF bytes. Returns empty string on failure."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        text = pdf_extract(tmp_path) or ""
        return text.strip()
    except Exception as e:
        print(f"[PDF] Extraction error: {e}")
        return ""
    finally:
        os.unlink(tmp_path)


# ── 2. Skill Extraction ───────────────────────────────────────────────────────

def _normalize_skill(skill: str) -> str:
    """Lowercase and resolve synonyms."""
    s = skill.lower().strip()
    return SKILL_SYNONYMS.get(s, s)

def extract_skills(text: str) -> list[str]:
    """Extract and normalize skills found in text."""
    text_lower = text.lower()
    found = set()
    for skill in SKILLS_DB:
        # Use word-boundary matching for short skills to avoid false positives
        if len(skill) <= 3:
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text_lower):
                found.add(_normalize_skill(skill))
        else:
            if skill in text_lower:
                found.add(_normalize_skill(skill))
    return sorted(list(found))

def compute_skill_score(resume_text: str, jd_text: str) -> Tuple[float, list, list, list]:
    """
    Compare skills between resume and JD.
    Returns: (score 0-100, resume_skills, jd_skills, missing_skills)
    """
    resume_skills = set(extract_skills(resume_text))
    jd_skills     = set(extract_skills(jd_text))

    if not jd_skills:
        # No identifiable skills in JD → neutral score
        return 60.0, sorted(resume_skills), [], []

    matched  = resume_skills & jd_skills
    missing  = jd_skills - resume_skills

    # Weighted: matched / jd_skills, capped and scaled
    raw_score = len(matched) / len(jd_skills) if jd_skills else 0
    # Bonus: having extra skills beyond JD requirements
    bonus = min(len(resume_skills - jd_skills) * 2, 10)
    score = min(raw_score * 90 + bonus, 100)

    return round(score, 1), sorted(resume_skills), sorted(jd_skills), sorted(missing)


# ── 3. Semantic Similarity ────────────────────────────────────────────────────

def semantic_match(resume_text: str, jd_text: str) -> float:
    """
    Compute semantic similarity (0-100).
    Uses sentence-transformers if available, falls back to enhanced TF-IDF.
    """
    if not resume_text.strip() or not jd_text.strip():
        return 0.0

    if _ST_MODEL is not None:
        try:
            emb_r = _ST_MODEL.encode(resume_text[:4000], convert_to_tensor=True)
            emb_j = _ST_MODEL.encode(jd_text[:2000],    convert_to_tensor=True)
            sim = float(st_util.cos_sim(emb_r, emb_j)[0][0])
            # Sentence transformers returns -1 to 1; rescale to 0-100
            score = (sim + 1) / 2 * 100
            return round(min(max(score, 0), 100), 1)
        except Exception as e:
            print(f"[Semantic] ST error: {e}, falling back to TF-IDF")

    # Enhanced TF-IDF fallback with chunk averaging
    try:
        # Split into chunks for better coverage
        resume_chunks = _chunk_text(resume_text, 500)
        jd_chunks     = _chunk_text(jd_text, 500)
        all_docs = resume_chunks + jd_chunks
        tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)
        matrix = tfidf.fit_transform(all_docs)
        r_vecs = matrix[:len(resume_chunks)]
        j_vecs = matrix[len(resume_chunks):]
        sims = sk_cosine(r_vecs, j_vecs)
        raw = float(sims.max())
        # TF-IDF cosine tends to be low; scale up to realistic range
        score = raw * 100
        # Apply a sigmoid-like boost to push scores into realistic range
        score = _boost_score(score, low=5, high=85, midpoint=30)
        return round(score, 1)
    except Exception as e:
        print(f"[Semantic] TF-IDF error: {e}")
        return 50.0

def _chunk_text(text: str, size: int) -> list[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), size):
        chunk = " ".join(words[i:i+size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks or [text]

def _boost_score(raw: float, low: float = 10, high: float = 90,
                 midpoint: float = 35) -> float:
    """
    Sigmoid-style boost: maps raw TF-IDF score into a more realistic range.
    Scores near 0 stay near `low`, scores near 100 approach `high`.
    """
    if raw <= 0:
        return low
    # Logistic stretch
    k = 0.12
    shifted = raw - midpoint
    sig = 1 / (1 + math.exp(-k * shifted))
    return low + (high - low) * sig


# ── 4. Experience Matching ────────────────────────────────────────────────────

def compute_experience_score(resume_text: str, jd_text: str) -> Tuple[float, str]:
    """
    Heuristic experience matching (0-100).
    Checks years of experience, seniority level alignment, and section quality.
    Returns: (score, label)
    """
    resume_lower = resume_text.lower()
    jd_lower     = jd_text.lower()

    score = 50.0  # neutral base

    # ── Years of experience extraction ──
    resume_years = _extract_years(resume_lower)
    jd_years     = _extract_years(jd_lower)

    if resume_years > 0 and jd_years > 0:
        if resume_years >= jd_years:
            score += 20 + min((resume_years - jd_years) * 3, 10)
        else:
            gap = jd_years - resume_years
            score -= min(gap * 8, 25)

    # ── Seniority level matching ──
    jd_level     = _detect_level(jd_lower)
    resume_level = _detect_level(resume_lower)
    level_order  = ["intern","junior","mid","senior","lead","principal","staff","director"]

    if jd_level in level_order and resume_level in level_order:
        jd_idx = level_order.index(jd_level)
        rv_idx = level_order.index(resume_level)
        diff   = rv_idx - jd_idx
        if diff >= 0:
            score += min(diff * 5, 15)
        else:
            score -= min(abs(diff) * 7, 20)

    # ── Section quality ──
    sections_found = sum(
        1 for kws in SECTION_KEYWORDS.values()
        if any(kw in resume_lower for kw in kws)
    )
    score += sections_found * 2

    # ── Action verbs ──
    verbs_found = sum(1 for v in ACTION_VERBS if v in resume_lower)
    score += min(verbs_found * 1.5, 12)

    score = round(min(max(score, 10), 100), 1)
    label = "Good" if score >= 60 else "Partial" if score >= 35 else "Low"
    return score, label

def _extract_years(text: str) -> float:
    """Extract maximum years of experience mentioned."""
    patterns = [
        r'(\d+)\+?\s*years?\s*of\s*experience',
        r'(\d+)\+?\s*yrs?\s*of\s*experience',
        r'experience\s*of\s*(\d+)\+?\s*years?',
        r'(\d+)\+?\s*years?\s*(?:professional|industry|work)',
    ]
    found = []
    for p in patterns:
        for m in re.finditer(p, text):
            try:
                found.append(float(m.group(1)))
            except ValueError:
                pass
    return max(found) if found else 0.0

def _detect_level(text: str) -> str:
    """Detect seniority level from text."""
    levels = {
        "intern":    ["intern","internship","trainee","fresher","entry level"],
        "junior":    ["junior","associate","entry","jr.","jr ","0-2 years","1-2 years"],
        "mid":       ["mid-level","mid level","intermediate","2-4 years","3-5 years"],
        "senior":    ["senior","sr.","sr ","5+ years","6+ years","7+ years"],
        "lead":      ["lead","team lead","tech lead","principal engineer"],
        "principal": ["principal","staff engineer","architect"],
        "director":  ["director","vp","vice president","head of"],
    }
    for level, keywords in levels.items():
        if any(kw in text for kw in keywords):
            return level
    return "mid"  # default assumption


# ── 5. Advanced ATS Score Engine ──────────────────────────────────────────────

def compute_ats_score(resume_text: str, jd_text: str = "") -> float:
    """
    Weighted ATS scoring (0-100):
      Section Coverage      25 pts
      Keyword Density       25 pts
      Action Verbs          15 pts
      Quantified Results    15 pts
      Resume Length/Format  10 pts
      Contact Info          10 pts
    """
    text  = resume_text.lower()
    score = 0.0

    # 1. Section Coverage (25 pts)
    section_score = 0
    for section, keywords in SECTION_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            section_score += 25 / len(SECTION_KEYWORDS)
    score += section_score

    # 2. Keyword Density vs JD (25 pts)
    if jd_text.strip():
        jd_skills = set(extract_skills(jd_text))
        res_skills = set(extract_skills(resume_text))
        if jd_skills:
            coverage = len(jd_skills & res_skills) / len(jd_skills)
            score += coverage * 25
    else:
        # No JD: base score from skill count
        skill_count = len(extract_skills(resume_text))
        score += min(skill_count * 2, 25)

    # 3. Action Verbs (15 pts)
    verbs_found = sum(1 for v in ACTION_VERBS if v in text)
    score += min(verbs_found / len(ACTION_VERBS) * 15 * 4, 15)

    # 4. Quantified Achievements (15 pts)
    numbers     = re.findall(r'\b\d+(?:\.\d+)?%?\b', resume_text)
    metrics     = [n for n in numbers if len(n) >= 2]
    score      += min(len(metrics) * 1.5, 15)

    # 5. Resume Length (10 pts)
    word_count = len(resume_text.split())
    if word_count >= 400:
        score += 10
    elif word_count >= 200:
        score += 6
    elif word_count >= 100:
        score += 3

    # 6. Contact Info (10 pts)
    contact_pts = 0
    if re.search(r'[\w.+-]+@[\w-]+\.\w+', resume_text):  contact_pts += 4   # email
    if re.search(r'[\+\(]?[\d\s\-\(\)]{9,15}', resume_text): contact_pts += 3  # phone
    if re.search(r'linkedin\.com|github\.com', text):     contact_pts += 3   # links
    score += contact_pts

    return round(min(max(score, 10), 100), 1)


# ── 6. Hybrid Score Combiner ──────────────────────────────────────────────────

def compute_hybrid_score(semantic: float, skill: float, experience: float) -> float:
    """
    Weighted combination:
      Semantic    → 40%
      Skill       → 30%
      Experience  → 30%
    """
    final = (0.40 * semantic) + (0.30 * skill) + (0.30 * experience)
    return round(min(max(final, 0), 100), 1)


# ── 7. Gemini AI Feedback ─────────────────────────────────────────────────────

async def generate_ai_feedback(
    resume_text: str,
    jd_text: str,
    hybrid_score: float,
    resume_skills: list,
    jd_skills: list,
    missing_skills: list,
) -> dict | None:
    """
    Call Gemini with a structured prompt. Returns parsed JSON or None.
    Passes pre-computed scores so Gemini focuses on qualitative feedback.
    """
    if not GEMINI_API_KEY:
        return None

    try:
        import httpx

        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
        )

        prompt = f"""You are a Senior ATS Expert and Career Coach with 15+ years of experience.
Analyze this resume against the job description and provide ACCURATE, DETAILED feedback.

=== RESUME (first 3000 chars) ===
{resume_text[:3000]}

=== JOB DESCRIPTION (first 2000 chars) ===
{jd_text[:2000]}

=== PRE-COMPUTED CONTEXT ===
Hybrid Match Score: {hybrid_score}%
Candidate Skills Found: {", ".join(resume_skills[:20])}
Job Required Skills: {", ".join(jd_skills[:20])}
Missing Skills: {", ".join(missing_skills[:15])}

=== YOUR TASK ===
Based on the above, provide a professional evaluation. Be SPECIFIC and ACTIONABLE.
The match score is already computed — focus on WHY and WHAT TO DO.

Respond ONLY with valid JSON. No markdown, no explanation outside JSON.

{{
  "summary": "<3 sentence professional assessment explaining the match quality, key strengths, and main gaps>",
  "strengths": [
    "<specific strength 1 with evidence from resume>",
    "<specific strength 2>",
    "<specific strength 3>",
    "<specific strength 4>"
  ],
  "improvements": [
    "<specific actionable improvement 1>",
    "<specific actionable improvement 2>",
    "<specific actionable improvement 3>",
    "<specific actionable improvement 4>"
  ],
  "missing_skills": {missing_skills[:10]},
  "ats_tips": [
    "<concrete ATS optimization tip 1>",
    "<concrete ATS optimization tip 2>",
    "<concrete ATS optimization tip 3>"
  ],
  "experience_match": "<Good|Partial|Low>",
  "education_match": "<Good|Partial|Low>",
  "why_this_score": "<2 sentence explanation of why the candidate received this score>"
}}"""

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.15,
                    "maxOutputTokens": 1500,
                    "topP": 0.8,
                }
            })

        if resp.status_code != 200:
            print(f"[Gemini] HTTP {resp.status_code}: {resp.text[:300]}")
            return None

        raw  = resp.json()
        text = raw["candidates"][0]["content"]["parts"][0]["text"]
        text = re.sub(r"```json|```", "", text).strip()

        # Safe parse
        parsed = json.loads(text)
        return parsed

    except json.JSONDecodeError as e:
        print(f"[Gemini] JSON parse error: {e}")
        return None
    except Exception as e:
        print(f"[Gemini] Error: {e}")
        return None


# ── 8. Local Feedback Generator (No-API Fallback) ────────────────────────────

def generate_local_feedback(
    resume_text: str,
    jd_text: str,
    semantic_score: float,
    skill_score: float,
    experience_score: float,
    hybrid_score: float,
    resume_skills: list,
    jd_skills: list,
    missing_skills: list,
    experience_label: str,
) -> dict:
    """
    Generate structured feedback without any API call.
    Produces realistic, context-aware output.
    """
    # Build strengths from what's found
    strengths = []
    if len(resume_skills) >= 8:
        strengths.append(f"Strong technical profile with {len(resume_skills)} relevant skills identified")
    elif resume_skills:
        strengths.append(f"Has {len(resume_skills)} relevant skills including {', '.join(resume_skills[:3])}")

    matched = set(resume_skills) & set(jd_skills)
    if matched:
        strengths.append(f"Directly matches {len(matched)} of {len(jd_skills)} required skills: {', '.join(sorted(matched)[:4])}")

    verbs_found = [v for v in ACTION_VERBS if v in resume_text.lower()]
    if verbs_found:
        strengths.append(f"Uses strong action verbs: {', '.join(verbs_found[:4])}")

    numbers = re.findall(r'\b\d+(?:\.\d+)?%\b', resume_text)
    if numbers:
        strengths.append(f"Includes {len(numbers)} quantified achievement(s) with metrics")

    sections_found = [
        s for s, kws in SECTION_KEYWORDS.items()
        if any(kw in resume_text.lower() for kw in kws)
    ]
    if "projects" in sections_found:
        strengths.append("Has a dedicated Projects section which strengthens the application")

    if not strengths:
        strengths = ["Resume submitted for analysis", "Review improvements below to strengthen application"]

    # Build improvements
    improvements = []
    if missing_skills:
        improvements.append(f"Learn and add these missing skills: {', '.join(missing_skills[:5])}")
    if "projects" not in sections_found:
        improvements.append("Add a Projects section showcasing relevant work with technologies used")
    if not numbers:
        improvements.append("Quantify achievements with numbers (e.g., 'Reduced load time by 40%', 'Led team of 5')")
    if len(resume_text.split()) < 300:
        improvements.append("Expand resume content — aim for 400-600 words with detailed experience descriptions")
    if "summary" not in sections_found:
        improvements.append("Add a professional Summary/Objective section tailored to this specific role")
    improvements.append("Mirror exact keywords from the job description to pass automated ATS filters")

    # ATS Tips
    ats_tips = [
        f"Include these exact keywords from the JD: {', '.join(list(missing_skills)[:4]) if missing_skills else 'match all JD terms'}",
        "Use standard section headers: 'Work Experience', 'Education', 'Skills', 'Projects'",
        "Avoid tables, columns, and graphics — ATS systems parse plain text better",
        "Add your LinkedIn profile URL and GitHub portfolio link",
        "Use bullet points starting with action verbs for each experience item",
    ]

    # Summary
    level = "strong" if hybrid_score >= 70 else "moderate" if hybrid_score >= 45 else "low"
    summary = (
        f"This resume shows a {level} match ({hybrid_score:.0f}%) for the target role. "
        f"The candidate has {len(resume_skills)} relevant skills with {len(matched)} directly matching job requirements. "
        f"{'Focus on adding missing skills and quantifying achievements to improve competitiveness.' if hybrid_score < 70 else 'The profile is competitive — refine with exact JD keywords for best ATS performance.'}"
    )

    return {
        "summary":          summary,
        "strengths":        strengths[:5],
        "improvements":     improvements[:5],
        "ats_tips":         ats_tips[:4],
        "experience_match": experience_label,
        "education_match":  "Partial",
        "why_this_score":   (
            f"Semantic content similarity contributed {semantic_score:.0f}% (40% weight), "
            f"skill matching {skill_score:.0f}% (30% weight), and experience alignment "
            f"{experience_score:.0f}% (30% weight) to produce the final score."
        ),
    }


# ── 9. Main Analyze Function ──────────────────────────────────────────────────

async def analyze_resume(resume_text: str, job_desc: str) -> dict:
    """
    Full hybrid analysis pipeline.
    Returns comprehensive scoring breakdown + AI feedback.
    """
    # ── Step 1: Compute all three scores ──
    semantic_score = semantic_match(resume_text, job_desc)
    skill_score, resume_skills, jd_skills, missing_skills = compute_skill_score(resume_text, job_desc)
    experience_score, experience_label = compute_experience_score(resume_text, job_desc)
    ats_score = compute_ats_score(resume_text, job_desc)
    hybrid_score = compute_hybrid_score(semantic_score, skill_score, experience_score)

    # ── Step 2: Try Gemini AI feedback ──
    ai_feedback = None
    if job_desc.strip():  # Only call Gemini when JD is provided
        ai_feedback = await generate_ai_feedback(
            resume_text, job_desc, hybrid_score,
            resume_skills, jd_skills, missing_skills
        )

    # ── Step 3: Build final response ──
    if ai_feedback:
        # Merge AI feedback with computed scores
        feedback = ai_feedback
        ai_powered = True
        # Prefer AI's missing_skills if it's a list, else use computed
        if not isinstance(feedback.get("missing_skills"), list):
            feedback["missing_skills"] = missing_skills
    else:
        # Full local fallback
        feedback = generate_local_feedback(
            resume_text, job_desc,
            semantic_score, skill_score, experience_score, hybrid_score,
            resume_skills, jd_skills, missing_skills, experience_label,
        )
        ai_powered = False

    return {
        # ── Core Scores ──
        "match_score":      hybrid_score,
        "semantic_score":   semantic_score,
        "skill_score":      skill_score,
        "experience_score": experience_score,
        "ats_score":        ats_score,

        # ── Skills ──
        "resume_skills":    resume_skills,
        "job_skills":       jd_skills,
        "missing_skills":   feedback.get("missing_skills", missing_skills),
        "matched_skills":   sorted(list(set(resume_skills) & set(jd_skills))),

        # ── Feedback ──
        "summary":          feedback.get("summary", ""),
        "strengths":        feedback.get("strengths", []),
        "improvements":     feedback.get("improvements", []),
        "ats_tips":         feedback.get("ats_tips", []),
        "why_this_score":   feedback.get("why_this_score", ""),
        "experience_match": feedback.get("experience_match", experience_label),
        "education_match":  feedback.get("education_match", "Partial"),

        # ── Meta ──
        "ai_powered":       ai_powered,
        "semantic_engine":  SEMANTIC_ENGINE,
    }


# ── 10. Multi-Resume Analysis ─────────────────────────────────────────────────

async def analyze_multiple(resumes: list, job_descs: list) -> list:
    """
    Analyze multiple resumes against multiple job descriptions.
    resumes:   [(filename, text), ...]
    job_descs: [(name, text), ...]
    Returns comparison matrix.
    """
    results = []
    for filename, resume_text in resumes:
        row = {"resume": filename, "scores": []}
        for jd_name, jd_text in job_descs:
            semantic = semantic_match(resume_text, jd_text)
            skill_sc, r_sk, j_sk, missing = compute_skill_score(resume_text, jd_text)
            exp_sc, exp_label = compute_experience_score(resume_text, jd_text)
            ats = compute_ats_score(resume_text, jd_text)
            hybrid = compute_hybrid_score(semantic, skill_sc, exp_sc)
            row["scores"].append({
                "job":              jd_name,
                "match_score":      hybrid,
                "semantic_score":   semantic,
                "skill_score":      skill_sc,
                "experience_score": exp_sc,
                "ats_score":        ats,
                "matched_skills":   sorted(set(r_sk) & set(j_sk)),
                "missing_skills":   missing[:5],
                "experience_match": exp_label,
            })
        results.append(row)
    return results