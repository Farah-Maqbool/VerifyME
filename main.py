from fastapi import FastAPI
from src.api import enroll, verify

app = FastAPI(title="Periocular Verification System")

app.include_router(enroll.router, prefix="/enroll", tags=["enrollment"])
app.include_router(verify.router, prefix="/verify", tags=["verification"])


@app.get("/")
def root():
    return {"status": "API is running"}