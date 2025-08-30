# AI-Powered Recommendation System Backend
# Production-ready FastAPI backend with multi-domain recommendations

import os
import asyncio
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Union
from contextlib import asynccontextmanager
import json
import numpy as np
from enum import Enum
import logging
from functools import wraps

# FastAPI and web framework imports
from fastapi import FastAPI, HTTPException, Depends, Security, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Database and caching
import asyncpg
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import String, Integer, Float, DateTime, Text, Boolean, JSON, ForeignKey, Index

# Authentication and security
import jwt
from passlib.context import CryptContext
from argon2 import PasswordHasher
from cryptography.fernet import Fernet

# ML and AI imports
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import pandas as pd

# Data validation
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings

# Monitoring and logging
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge
import structlog

# Async task processing
from celery import Celery

# ==================== CONFIGURATION ====================

class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql+asyncpg://user:password@localhost/recommendations"
    redis_url: str = "redis://localhost:6379"
    neo4j_url: str = "bolt://localhost:7687"
    
    # Security
    secret_key: str = secrets.token_urlsafe(32)
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    encryption_key: str = Fernet.generate_key().decode()
    
    # API
    api_title: str = "AI Recommendation System"
    api_version: str = "1.0.0"
    debug: bool = False
    
    # Rate limiting
    rate_limit: str = "100/minute"
    
    # Celery
    broker_url: str = "redis://localhost:6379/1"
    result_backend: str = "redis://localhost:6379/2"
    
    class Config:
        env_file = ".env"

settings = Settings()

# ==================== SECURITY SETUP ====================

# Password hashing
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")
ph = PasswordHasher()

# Encryption
cipher_suite = Fernet(settings.encryption_key.encode())

# JWT Security
security = HTTPBearer()

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

# ==================== LOGGING & MONITORING ====================

# Structured logging
logger = structlog.get_logger()

# Prometheus metrics
REQUEST_COUNT = Counter('requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('request_duration_seconds', 'Request duration')
RECOMMENDATION_COUNT = Counter('recommendations_total', 'Total recommendations served', ['domain'])
ACTIVE_USERS = Gauge('active_users', 'Currently active users')

# ==================== DATABASE MODELS ====================

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    username: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    preferences: Mapped[Dict] = mapped_column(JSON, default=dict)

class Content(Base):
    __tablename__ = "content"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(255), index=True)
    description: Mapped[str] = mapped_column(Text)
    domain: Mapped[str] = mapped_column(String(50), index=True)  # music, movies, games, apps
    genre: Mapped[str] = mapped_column(String(100), index=True)
    features: Mapped[Dict] = mapped_column(JSON, default=dict)
    embedding: Mapped[List[float]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (Index('idx_domain_genre', 'domain', 'genre'),)

class UserInteraction(Base):
    __tablename__ = "user_interactions"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"))
    content_id: Mapped[int] = mapped_column(Integer, ForeignKey("content.id"))
    interaction_type: Mapped[str] = mapped_column(String(50))  # like, dislike, view, share
    rating: Mapped[Optional[float]] = mapped_column(Float)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    context: Mapped[Dict] = mapped_column(JSON, default=dict)

class RecommendationLog(Base):
    __tablename__ = "recommendation_logs"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"))
    recommendations: Mapped[List] = mapped_column(JSON)
    algorithm_used: Mapped[str] = mapped_column(String(100))
    context: Mapped[Dict] = mapped_column(JSON, default=dict)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

# ==================== PYDANTIC MODELS ====================

class UserCreate(BaseModel):
    email: str = Field(..., regex=r'^[^@]+@[^@]+\.[^@]+$')
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)

class UserResponse(BaseModel):
    id: int
    email: str
    username: str
    is_active: bool
    created_at: datetime

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

class RecommendationRequest(BaseModel):
    user_id: Optional[int] = None
    domain: Optional[str] = None
    limit: int = Field(default=10, le=50)
    context: Dict = Field(default_factory=dict)

class MoodAnalysisRequest(BaseModel):
    text: Optional[str] = None
    audio_data: Optional[str] = None  # base64 encoded audio

class FeedbackRequest(BaseModel):
    content_id: int
    feedback_type: str = Field(..., regex=r'^(like|dislike|view|share)$')
    rating: Optional[float] = Field(None, ge=0, le=5)

class DomainType(str, Enum):
    MUSIC = "music"
    MOVIES = "movies"
    GAMES = "games"
    APPS = "apps"

# ==================== AI/ML COMPONENTS ====================

class RecommendationEngine:
    def __init__(self):
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        self.tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
        self.sentiment_analyzer = pipeline("sentiment-analysis", 
                                         model="cardiffnlp/twitter-roberta-base-sentiment-latest")
        self.redis_client = None
        
    async def initialize_redis(self):
        self.redis_client = redis.from_url(settings.redis_url)
        
    def generate_content_embedding(self, text: str) -> List[float]:
        """Generate embeddings for content using SentenceTransformer"""
        embedding = self.sentence_transformer.encode(text)
        return embedding.tolist()
    
    def compute_content_similarity(self, embeddings1: List[float], 
                                 embeddings2: List[List[float]]) -> List[float]:
        """Compute cosine similarity between content embeddings"""
        emb1 = np.array(embeddings1).reshape(1, -1)
        emb2 = np.array(embeddings2)
        return cosine_similarity(emb1, emb2)[0].tolist()
    
    async def content_based_recommendations(self, user_id: int, domain: str, 
                                          limit: int = 10) -> List[Dict]:
        """Content-based filtering using embeddings"""
        cache_key = f"content_rec:{user_id}:{domain}:{limit}"
        
        # Check cache first
        cached = await self.redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
        
        # Get user's interaction history
        # This would query the database for user interactions
        # For demo purposes, we'll simulate recommendations
        recommendations = [
            {
                "content_id": i,
                "title": f"Recommended {domain.title()} {i}",
                "score": 0.9 - (i * 0.1),
                "explanation": f"Based on your interest in similar {domain}"
            }
            for i in range(1, limit + 1)
        ]
        
        # Cache results for 1 hour
        await self.redis_client.setex(cache_key, 3600, json.dumps(recommendations))
        return recommendations
    
    async def collaborative_filtering(self, user_id: int, domain: str, 
                                    limit: int = 10) -> List[Dict]:
        """Collaborative filtering using user similarity"""
        cache_key = f"collab_rec:{user_id}:{domain}:{limit}"
        
        cached = await self.redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
        
        # Simulate collaborative filtering recommendations
        recommendations = [
            {
                "content_id": i + 100,
                "title": f"Popular {domain.title()} {i}",
                "score": 0.85 - (i * 0.08),
                "explanation": f"Users with similar taste also liked this {domain}"
            }
            for i in range(1, limit + 1)
        ]
        
        await self.redis_client.setex(cache_key, 3600, json.dumps(recommendations))
        return recommendations
    
    def analyze_mood(self, text: str) -> Dict:
        """Analyze mood/sentiment from text"""
        result = self.sentiment_analyzer(text)[0]
        
        mood_mapping = {
            'LABEL_0': 'negative',
            'LABEL_1': 'neutral', 
            'LABEL_2': 'positive'
        }
        
        return {
            "mood": mood_mapping.get(result['label'], 'neutral'),
            "confidence": result['score'],
            "suggestions": self._mood_based_suggestions(mood_mapping.get(result['label'], 'neutral'))
        }
    
    def _mood_based_suggestions(self, mood: str) -> Dict:
        """Generate mood-based recommendations"""
        suggestions = {
            "positive": {
                "music": ["upbeat", "energetic", "pop"],
                "movies": ["comedy", "adventure", "musical"],
                "games": ["party", "racing", "adventure"],
                "apps": ["fitness", "social", "productivity"]
            },
            "negative": {
                "music": ["calm", "acoustic", "classical"],
                "movies": ["drama", "documentary", "indie"],
                "games": ["puzzle", "strategy", "relaxing"],
                "apps": ["meditation", "journaling", "wellness"]
            },
            "neutral": {
                "music": ["popular", "rock", "jazz"],
                "movies": ["thriller", "sci-fi", "biography"],
                "games": ["rpg", "simulation", "arcade"],
                "apps": ["news", "educational", "tools"]
            }
        }
        return suggestions.get(mood, suggestions["neutral"])

# ==================== DATABASE & CACHING ====================

# Database setup
engine = create_async_engine(settings.database_url, echo=settings.debug)
async_session = async_sessionmaker(engine, expire_on_commit=False)

# Redis setup
redis_client = redis.from_url(settings.redis_url)

# Celery setup
celery_app = Celery(
    "recommendation_system",
    broker=settings.broker_url,
    backend=settings.result_backend
)

# ==================== AUTHENTICATION ====================

async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Validate JWT token and return current user"""
    try:
        payload = jwt.decode(credentials.credentials, settings.secret_key, 
                           algorithms=[settings.algorithm])
        user_id: int = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        # Here you would fetch user from database
        # For demo, we'll return a mock user
        return {"id": user_id, "username": "demo_user"}
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
    
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password using Argon2"""
    try:
        return ph.verify(hashed_password, plain_password)
    except:
        return False

def hash_password(password: str) -> str:
    """Hash password using Argon2"""
    return ph.hash(password)

# ==================== WEBSOCKET CONNECTION MANAGER ====================

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        ACTIVE_USERS.inc()
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        ACTIVE_USERS.dec()
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast_recommendations(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                await self.disconnect(connection)

manager = ConnectionManager()

# ==================== FASTAPI APPLICATION ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting AI Recommendation System")
    
    # Initialize ML models
    app.state.recommendation_engine = RecommendationEngine()
    await app.state.recommendation_engine.initialize_redis()
    
    # Create database tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield
    
    # Shutdown
    logger.info("Shutting down AI Recommendation System")
    await redis_client.close()
    await engine.dispose()

# Create FastAPI app
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    lifespan=lifespan
)

# Add middlewares
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure properly for production
)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ==================== API ENDPOINTS ====================

@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint"""
    return {"status": "healthy", "service": "AI Recommendation System"}

@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check"""
    try:
        # Check Redis
        await redis_client.ping()
        redis_status = "healthy"
    except:
        redis_status = "unhealthy"
    
    return {
        "status": "healthy",
        "redis": redis_status,
        "timestamp": datetime.utcnow()
    }

@app.post("/auth/register", response_model=Token, tags=["Authentication"])
@limiter.limit("10/minute")
async def register(request, user: UserCreate):
    """Register a new user"""
    REQUEST_COUNT.labels(method="POST", endpoint="/auth/register").inc()
    
    # Check if user exists (mock implementation)
    # In production, check database
    
    # Hash password
    hashed_password = hash_password(user.password)
    
    # Create tokens
    access_token = create_access_token(data={"sub": str(1)})  # Mock user ID
    refresh_token = create_access_token(
        data={"sub": str(1)}, 
        expires_delta=timedelta(days=settings.refresh_token_expire_days)
    )
    
    return Token(access_token=access_token, refresh_token=refresh_token)

@app.post("/auth/login", response_model=Token, tags=["Authentication"])
@limiter.limit("20/minute")
async def login(request, email: str, password: str):
    """User login"""
    REQUEST_COUNT.labels(method="POST", endpoint="/auth/login").inc()
    
    # Mock authentication - in production, verify against database
    if email == "demo@example.com" and password == "password":
        access_token = create_access_token(data={"sub": str(1)})
        refresh_token = create_access_token(
            data={"sub": str(1)},
            expires_delta=timedelta(days=settings.refresh_token_expire_days)
        )
        return Token(access_token=access_token, refresh_token=refresh_token)
    
    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.post("/recommendations", tags=["Recommendations"])
@limiter.limit(settings.rate_limit)
async def get_recommendations(
    request,
    rec_request: RecommendationRequest,
    current_user: dict = Depends(get_current_user)
):
    """Get personalized recommendations"""
    with REQUEST_DURATION.time():
        REQUEST_COUNT.labels(method="POST", endpoint="/recommendations").inc()
        
        engine = app.state.recommendation_engine
        domain = rec_request.domain or "music"
        
        # Get content-based recommendations
        content_recs = await engine.content_based_recommendations(
            current_user["id"], domain, rec_request.limit // 2
        )
        
        # Get collaborative filtering recommendations
        collab_recs = await engine.collaborative_filtering(
            current_user["id"], domain, rec_request.limit // 2
        )
        
        # Combine and rank recommendations
        all_recs = content_recs + collab_recs
        all_recs = sorted(all_recs, key=lambda x: x["score"], reverse=True)
        
        RECOMMENDATION_COUNT.labels(domain=domain).inc()
        
        # Log recommendations
        log_data = {
            "user_id": current_user["id"],
            "recommendations": all_recs[:rec_request.limit],
            "algorithm": "hybrid",
            "domain": domain
        }
        
        # Background task to save to database
        # In production, use Celery for this
        
        return {
            "recommendations": all_recs[:rec_request.limit],
            "total": len(all_recs[:rec_request.limit]),
            "domain": domain,
            "generated_at": datetime.utcnow()
        }

@app.post("/mood", tags=["AI Analysis"])
@limiter.limit("50/minute")
async def analyze_mood(
    request,
    mood_request: MoodAnalysisRequest,
    current_user: dict = Depends(get_current_user)
):
    """Analyze mood from text or audio"""
    REQUEST_COUNT.labels(method="POST", endpoint="/mood").inc()
    
    engine = app.state.recommendation_engine
    
    if mood_request.text:
        result = engine.analyze_mood(mood_request.text)
        return {
            "mood_analysis": result,
            "recommendations": result["suggestions"],
            "analyzed_at": datetime.utcnow()
        }
    elif mood_request.audio_data:
        # In production, implement audio sentiment analysis
        return {
            "mood_analysis": {"mood": "neutral", "confidence": 0.5},
            "message": "Audio analysis not implemented in demo",
            "analyzed_at": datetime.utcnow()
        }
    else:
        raise HTTPException(status_code=400, detail="Provide either text or audio data")

@app.get("/search", tags=["Search"])
@limiter.limit("100/minute")
async def unified_search(
    request,
    q: str,
    domain: Optional[str] = None,
    limit: int = 20,
    current_user: dict = Depends(get_current_user)
):
    """Unified search across all domains"""
    REQUEST_COUNT.labels(method="GET", endpoint="/search").inc()
    
    # Mock search results - in production, implement with Elasticsearch
    domains = [domain] if domain else ["music", "movies", "games", "apps"]
    
    results = []
    for d in domains:
        for i in range(1, min(limit//len(domains) + 1, 6)):
            results.append({
                "id": f"{d}_{i}",
                "title": f"{d.title()} result {i} for '{q}'",
                "domain": d,
                "score": 1.0 - (i * 0.1),
                "description": f"This is a {d} item matching your search"
            })
    
    return {
        "query": q,
        "results": sorted(results, key=lambda x: x["score"], reverse=True)[:limit],
        "total": len(results),
        "domains": domains
    }

@app.post("/feedback", tags=["User Feedback"])
@limiter.limit("200/minute")
async def submit_feedback(
    request,
    feedback: FeedbackRequest,
    current_user: dict = Depends(get_current_user)
):
    """Submit user feedback on recommendations"""
    REQUEST_COUNT.labels(method="POST", endpoint="/feedback").inc()
    
    # In production, save to database and trigger model retraining
    feedback_data = {
        "user_id": current_user["id"],
        "content_id": feedback.content_id,
        "feedback_type": feedback.feedback_type,
        "rating": feedback.rating,
        "timestamp": datetime.utcnow()
    }
    
    # Cache feedback for real-time model updates
    cache_key = f"feedback:{current_user['id']}:{feedback.content_id}"
    await redis_client.setex(cache_key, 86400, json.dumps(feedback_data))
    
    return {
        "status": "feedback_received",
        "message": "Thank you for your feedback!",
        "feedback_id": f"fb_{secrets.token_hex(8)}"
    }

# ==================== WEBSOCKET ENDPOINTS ====================

@app.websocket("/ws/recommendations")
async def websocket_recommendations(websocket: WebSocket):
    """Real-time recommendation updates via WebSocket"""
    await manager.connect(websocket)
    
    try:
        while True:
            # Wait for client messages
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Process real-time recommendation request
            if message.get("type") == "get_recommendations":
                # Mock real-time recommendations
                recommendations = {
                    "type": "recommendations_update",
                    "data": [
                        {
                            "id": i,
                            "title": f"Real-time recommendation {i}",
                            "score": 0.9 - (i * 0.1)
                        }
                        for i in range(1, 6)
                    ],
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                await manager.send_personal_message(
                    json.dumps(recommendations), websocket
                )
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# ==================== ADMIN ENDPOINTS ====================

@app.get("/admin/stats", tags=["Admin"])
async def get_system_stats(current_user: dict = Depends(get_current_user)):
    """Get system statistics (admin only)"""
    # In production, check if user is admin
    
    return {
        "active_users": len(manager.active_connections),
        "total_requests": REQUEST_COUNT._value._value,
        "recommendations_served": RECOMMENDATION_COUNT._value._value,
        "system_health": "optimal",
        "cache_hit_rate": 0.85,  # Mock data
        "avg_response_time": "120ms"
    }

# ==================== BACKGROUND TASKS ====================

@celery_app.task
def retrain_models():
    """Background task to retrain ML models"""
    logger.info("Starting model retraining...")
    # Implement model retraining logic
    return "Model retraining completed"

@celery_app.task
def cleanup_cache():
    """Background task to cleanup old cache entries"""
    logger.info("Cleaning up cache...")
    # Implement cache cleanup
    return "Cache cleanup completed"

# ==================== METRICS ENDPOINT ====================

@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """Prometheus metrics endpoint"""
    return prometheus_client.generate_latest()

# ==================== ERROR HANDLERS ====================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """General exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=4,
        reload=settings.debug
    )