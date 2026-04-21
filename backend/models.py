from sqlalchemy import Column, Integer, String, Float, Text, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base

class User(Base):
    __tablename__ = "users"
    id              = Column(Integer, primary_key=True, index=True)
    username        = Column(String, unique=True, index=True)
    email           = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    saved_skills    = Column(Text, default="[]")
    created_at      = Column(DateTime, default=datetime.utcnow)
    analyses        = relationship("Analysis", back_populates="user", cascade="all, delete-orphan")
    job_history     = relationship("JobHistory", back_populates="user", cascade="all, delete-orphan")
    saved_jobs      = relationship("SavedJob", back_populates="user", cascade="all, delete-orphan")

class Job(Base):
    __tablename__ = "jobs"
    id          = Column(Integer, primary_key=True, index=True)
    title       = Column(String)
    company     = Column(String)
    description = Column(Text)
    location    = Column(String, default="Remote")
    job_type    = Column(String, default="Full-time")
    salary      = Column(String, default="")
    skills      = Column(Text, default="[]")
    posted_by   = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at  = Column(DateTime, default=datetime.utcnow)

class Analysis(Base):
    __tablename__ = "analyses"
    id              = Column(Integer, primary_key=True, index=True)
    user_id         = Column(Integer, ForeignKey("users.id"))
    resume_filename = Column(String)
    resume_text     = Column(Text, default="")
    match_score     = Column(Float, default=0)
    ats_score       = Column(Float, default=0)
    resume_skills   = Column(Text, default="[]")
    job_skills      = Column(Text, default="[]")
    missing_skills  = Column(Text, default="[]")
    ai_summary      = Column(Text, default="")
    strengths       = Column(Text, default="[]")
    improvements    = Column(Text, default="[]")
    ats_tips        = Column(Text, default="[]")
    experience_match = Column(String, default="")
    education_match  = Column(String, default="")
    ai_powered      = Column(Boolean, default=False)
    job_desc_used   = Column(Text, default="")
    created_at      = Column(DateTime, default=datetime.utcnow)
    user            = relationship("User", back_populates="analyses")

class JobHistory(Base):
    __tablename__ = "job_history"
    id          = Column(Integer, primary_key=True, index=True)
    user_id     = Column(Integer, ForeignKey("users.id"))
    company     = Column(String)
    role        = Column(String)
    match_score = Column(Float, default=0)
    status      = Column(String, default="Applied")
    notes       = Column(Text, default="")
    applied_at  = Column(DateTime, default=datetime.utcnow)
    user        = relationship("User", back_populates="job_history")

class SavedJob(Base):
    __tablename__ = "saved_jobs"
    id         = Column(Integer, primary_key=True, index=True)
    user_id    = Column(Integer, ForeignKey("users.id"))
    job_id     = Column(Integer, ForeignKey("jobs.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    user       = relationship("User", back_populates="saved_jobs")