from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from src.api import enroll, verify

app = FastAPI(title="Periocular Verification System")

app.include_router(enroll.router, prefix="/enroll", tags=["enrollment"])
app.include_router(verify.router, prefix="/verify", tags=["verification"])

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/")
def root(request: Request):
    return templates.TemplateResponse(request, "index.html", {})


@app.get("/enroll-page")
def enroll_page(request: Request):
    return templates.TemplateResponse(request, "enroll.html", {})