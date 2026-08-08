from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets
import os
from dotenv import load_dotenv
load_dotenv()
from src.api import enroll, verify

app = FastAPI(title="Periocular Verification System")

security = HTTPBasic()

ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "changeme")


def verify_admin(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, ADMIN_USERNAME)
    correct_password = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


app.include_router(enroll.router, prefix="/enroll", tags=["enrollment"], dependencies=[Depends(verify_admin)])
app.include_router(verify.router, prefix="/verify", tags=["verification"])

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/")
def root(request: Request):
    return templates.TemplateResponse(request, "index.html", {})


@app.get("/enroll-page")
def enroll_page(request: Request, username: str = Depends(verify_admin)):
    return templates.TemplateResponse(request, "enroll.html", {})