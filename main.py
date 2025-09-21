import os
import hashlib
import time
import re
import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional, List
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
import resend

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, HttpUrl
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean, Index
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/website_monitor")
RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")
FROM_EMAIL = os.getenv("FROM_EMAIL", "Website Monitor <noreply@yourdomain.com>")
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
SECRET_KEY = os.getenv("SECRET_KEY", "change-this-in-production")

# Handle Railway's PostgreSQL URL format
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
engine = create_engine(DATABASE_URL, pool_size=10, max_overflow=20)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database Models
class Subscription(Base):
    __tablename__ = "subscriptions"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), nullable=False, index=True)
    url = Column(Text, nullable=False)
    content_hash = Column(String(64), nullable=True)
    last_checked = Column(DateTime, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    is_active = Column(Boolean, default=True, index=True)
    sensitivity = Column(String(20), default="medium")
    unsubscribe_token = Column(String(64), unique=True, index=True)
    consecutive_failures = Column(Integer, default=0)
    last_error = Column(Text, nullable=True)

class CheckHistory(Base):
    __tablename__ = "check_history"
    
    id = Column(Integer, primary_key=True, index=True)
    subscription_id = Column(Integer, nullable=False)
    checked_at = Column(DateTime, server_default=func.now())
    change_detected = Column(Boolean, default=False)
    old_hash = Column(String(64), nullable=True)
    new_hash = Column(String(64), nullable=True)
    error_message = Column(Text, nullable=True)

# Create tables
Base.metadata.create_all(bind=engine)

# Initialize Resend
resend.api_key = RESEND_API_KEY

# FastAPI app
app = FastAPI(title="Website Change Monitor", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pydantic models
class SubscriptionCreate(BaseModel):
    email: EmailStr
    url: HttpUrl
    sensitivity: Optional[str] = "medium"

class SubscriptionResponse(BaseModel):
    id: int
    email: str
    url: str
    is_active: bool
    created_at: datetime
    last_checked: Optional[datetime]

# Helper functions
def validate_url(url: str) -> bool:
    """Validate URL for security"""
    try:
        parsed = urlparse(str(url))
        if not parsed.scheme or not parsed.netloc:
            return False
        if parsed.scheme not in ['http', 'https']:
            return False
        
        hostname = parsed.hostname
        if not hostname:
            return False
            
        # Block internal networks
        blocked_patterns = [
            r'^localhost$', r'^127\.', r'^10\.', r'^172\.(1[6-9]|2[0-9]|3[0-1])\.',
            r'^192\.168\.', r'^169\.254\.', r'^::1$', r'^fe80:', r'^fc00:', r'^fd00:'
        ]
        
        for pattern in blocked_patterns:
            if re.match(pattern, hostname, re.IGNORECASE):
                return False
        return True
    except:
        return False

def clean_content(html: str, sensitivity: str = "medium") -> str:
    """Clean HTML content for comparison"""
    soup = BeautifulSoup(html, 'html.parser')
    
    # Remove script and style elements
    for element in soup.find_all(['script', 'style', 'noscript']):
        element.decompose()
    
    # Remove dynamic elements based on sensitivity
    dynamic_selectors = [
        '.timestamp', '.date', '.time', '.ad', '.advertisement', 
        '#comments', '.comments', '.social-share', '.cookie-notice'
    ]
    
    if sensitivity in ["medium", "high"]:
        dynamic_selectors.extend(['.sidebar', '.related-posts', '.trending'])
    
    if sensitivity == "high":
        dynamic_selectors.extend(['.navigation', '.nav', '.footer'])
    
    for selector in dynamic_selectors:
        try:
            for element in soup.select(selector):
                element.decompose()
        except:
            continue
    
    # Get clean text and normalize
    text = soup.get_text(separator=' ', strip=True)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\d+ (views?|comments?|shares?)', '', text, flags=re.IGNORECASE)
    return text.strip()

def generate_hash(content: str) -> str:
    """Generate SHA-256 hash of content"""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def fetch_content(url: str) -> tuple[Optional[str], Optional[str]]:
    """Fetch content from URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; WebsiteMonitor/1.0)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
        }
        response = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
        response.raise_for_status()
        
        content_type = response.headers.get('content-type', '').lower()
        if 'text/html' not in content_type:
            return None, f"Unsupported content type: {content_type}"
        
        return response.text, None
    except requests.exceptions.Timeout:
        return None, "Request timeout"
    except requests.exceptions.ConnectionError:
        return None, "Connection error"
    except requests.exceptions.HTTPError as e:
        return None, f"HTTP error: {e.response.status_code}"
    except Exception as e:
        return None, f"Error: {str(e)}"

def send_email(to_email: str, subject: str, html_content: str, text_content: str = None):
    """Send email via Resend"""
    try:
        resend.Emails.send({
            "from": FROM_EMAIL,
            "to": [to_email],
            "subject": subject,
            "html": html_content,
            "text": text_content or "Please view this email in HTML format."
        })
        logger.info(f"Email sent to {to_email}")
        return True
    except Exception as e:
        logger.error(f"Failed to send email to {to_email}: {e}")
        return False

def send_confirmation_email(email: str, url: str, unsubscribe_token: str):
    """Send subscription confirmation email"""
    unsubscribe_url = f"{BASE_URL}/unsubscribe/{unsubscribe_token}"
    
    html_content = f"""
    <html>
    <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; text-align: center;">
            <h1>🔍 Website Monitor</h1>
            <p>Your subscription is active!</p>
        </div>
        <div style="padding: 20px;">
            <h2>Welcome!</h2>
            <p>You're now monitoring: <strong>{url}</strong></p>
            <p>We'll email you when we detect changes to this website.</p>
            <p><a href="{BASE_URL}/dashboard?email={email}" style="background: #667eea; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">View Dashboard</a></p>
        </div>
        <div style="padding: 20px; border-top: 1px solid #ddd; color: #666; font-size: 12px; text-align: center;">
            <p><a href="{unsubscribe_url}">Unsubscribe</a></p>
        </div>
    </body>
    </html>
    """
    
    send_email(email, "🔍 Website Monitor - Subscription Confirmed", html_content)

def send_change_notification(email: str, url: str, unsubscribe_token: str):
    """Send change notification email"""
    unsubscribe_url = f"{BASE_URL}/unsubscribe/{unsubscribe_token}"
    
    html_content = f"""
    <html>
    <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 20px; text-align: center;">
            <h1>🚨 Change Detected!</h1>
            <p>A website you're monitoring has been updated</p>
        </div>
        <div style="padding: 20px;">
            <h2>Website Updated:</h2>
            <p><strong>{url}</strong></p>
            <p>We've detected changes to this website content.</p>
            <p><a href="{url}" style="background: #f5576c; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">Visit Website</a></p>
        </div>
        <div style="padding: 20px; border-top: 1px solid #ddd; color: #666; font-size: 12px; text-align: center;">
            <p><a href="{unsubscribe_url}">Unsubscribe</a></p>
        </div>
    </body>
    </html>
    """
    
    send_email(email, f"🚨 Change Detected - {url}", html_content)

def check_subscription(subscription_id: int):
    """Check a single subscription for changes"""
    db = SessionLocal()
    try:
        subscription = db.query(Subscription).filter(
            Subscription.id == subscription_id,
            Subscription.is_active == True
        ).first()
        
        if not subscription:
            return
        
        logger.info(f"Checking {subscription.url}")
        
        # Fetch current content
        html_content, error_message = fetch_content(subscription.url)
        
        if html_content:
            # Clean and hash content
            clean_text = clean_content(html_content, subscription.sensitivity)
            new_hash = generate_hash(clean_text)
            
            # Check for changes
            change_detected = False
            if subscription.content_hash and subscription.content_hash != new_hash:
                change_detected = True
                logger.info(f"Change detected for {subscription.url}")
                
                # Send notification
                send_change_notification(
                    subscription.email,
                    subscription.url,
                    subscription.unsubscribe_token
                )
            
            # Update subscription
            subscription.content_hash = new_hash
            subscription.last_checked = datetime.utcnow()
            subscription.consecutive_failures = 0
            subscription.last_error = None
            
            # Log check
            check_record = CheckHistory(
                subscription_id=subscription.id,
                change_detected=change_detected,
                old_hash=subscription.content_hash,
                new_hash=new_hash
            )
            db.add(check_record)
            
        else:
            # Handle error
            subscription.last_checked = datetime.utcnow()
            subscription.consecutive_failures += 1
            subscription.last_error = error_message
            
            check_record = CheckHistory(
                subscription_id=subscription.id,
                change_detected=False,
                error_message=error_message
            )
            db.add(check_record)
            
            logger.error(f"Failed to check {subscription.url}: {error_message}")
        
        db.commit()
        
    except Exception as e:
        logger.error(f"Error checking subscription {subscription_id}: {e}")
    finally:
        db.close()

def check_all_subscriptions():
    """Check all active subscriptions"""
    db = SessionLocal()
    try:
        # Get subscriptions that need checking (not checked in last hour)
        cutoff_time = datetime.utcnow() - timedelta(hours=1)
        subscriptions = db.query(Subscription).filter(
            Subscription.is_active == True,
            (Subscription.last_checked.is_(None) | (Subscription.last_checked < cutoff_time))
        ).limit(100).all()
        
        logger.info(f"Checking {len(subscriptions)} subscriptions")
        
        for subscription in subscriptions:
            check_subscription(subscription.id)
            time.sleep(2)  # Be polite to websites
            
    except Exception as e:
        logger.error(f"Error in check_all_subscriptions: {e}")
    finally:
        db.close()

# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, email: Optional[str] = None, db: Session = Depends(get_db)):
    subscriptions = []
    if email:
        subscriptions = db.query(Subscription).filter(
            Subscription.email == email,
            Subscription.is_active == True
        ).all()
    
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "subscriptions": subscriptions,
        "email": email
    })

@app.post("/subscribe", response_model=SubscriptionResponse)
async def create_subscription(
    subscription: SubscriptionCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    # Validate URL
    if not validate_url(str(subscription.url)):
        raise HTTPException(status_code=400, detail="Invalid or blocked URL")
    
    # Check if subscription already exists
    existing = db.query(Subscription).filter(
        Subscription.email == subscription.email,
        Subscription.url == str(subscription.url),
        Subscription.is_active == True
    ).first()
    
    if existing:
        raise HTTPException(status_code=400, detail="Subscription already exists")
    
    # Create subscription
    db_subscription = Subscription(
        email=subscription.email,
        url=str(subscription.url),
        sensitivity=subscription.sensitivity,
        unsubscribe_token=str(uuid.uuid4())
    )
    
    db.add(db_subscription)
    db.commit()
    db.refresh(db_subscription)
    
    # Send confirmation email and do initial check
    background_tasks.add_task(send_confirmation_email, subscription.email, str(subscription.url), db_subscription.unsubscribe_token)
    background_tasks.add_task(check_subscription, db_subscription.id)
    
    return SubscriptionResponse(
        id=db_subscription.id,
        email=db_subscription.email,
        url=db_subscription.url,
        is_active=db_subscription.is_active,
        created_at=db_subscription.created_at,
        last_checked=db_subscription.last_checked
    )

@app.get("/subscriptions/{email}")
async def get_subscriptions(email: str, db: Session = Depends(get_db)):
    subscriptions = db.query(Subscription).filter(
        Subscription.email == email,
        Subscription.is_active == True
    ).all()
    
    return [SubscriptionResponse(
        id=sub.id,
        email=sub.email,
        url=sub.url,
        is_active=sub.is_active,
        created_at=sub.created_at,
        last_checked=sub.last_checked
    ) for sub in subscriptions]

@app.get("/unsubscribe/{token}")
async def unsubscribe(token: str, db: Session = Depends(get_db)):
    subscription = db.query(Subscription).filter(
        Subscription.unsubscribe_token == token
    ).first()
    
    if not subscription:
        raise HTTPException(status_code=404, detail="Invalid unsubscribe token")
    
    subscription.is_active = False
    db.commit()
    
    return {"message": f"Successfully unsubscribed {subscription.email} from {subscription.url}"}

@app.delete("/subscriptions/{subscription_id}")
async def delete_subscription(subscription_id: int, email: str, db: Session = Depends(get_db)):
    subscription = db.query(Subscription).filter(
        Subscription.id == subscription_id,
        Subscription.email == email,
        Subscription.is_active == True
    ).first()
    
    if not subscription:
        raise HTTPException(status_code=404, detail="Subscription not found")
    
    subscription.is_active = False
    db.commit()
    
    return {"message": "Subscription deleted"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}

@app.get("/stats")
async def get_stats(db: Session = Depends(get_db)):
    total_subscriptions = db.query(Subscription).filter(Subscription.is_active == True).count()
    total_checks = db.query(CheckHistory).count()
    recent_changes = db.query(CheckHistory).filter(
        CheckHistory.change_detected == True,
        CheckHistory.checked_at >= datetime.utcnow() - timedelta(days=7)
    ).count()
    
    return {
        "total_active_subscriptions": total_subscriptions,
        "total_checks_performed": total_checks,
        "changes_detected_last_7_days": recent_changes
    }

# Background scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(
    func=check_all_subscriptions,
    trigger=IntervalTrigger(minutes=30),  # Check every 30 minutes
    id='check_subscriptions',
    name='Check website subscriptions'
)

@app.on_event("startup")
async def startup_event():
    if RESEND_API_KEY:
        scheduler.start()
        logger.info("Scheduler started")
    else:
        logger.warning("RESEND_API_KEY not set, scheduler not started")

@app.on_event("shutdown")
async def shutdown_event():
    scheduler.shutdown()
    logger.info("Scheduler stopped")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
