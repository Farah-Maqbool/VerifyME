from fastapi import FastAPI

app = FastAPI(title="VerifyME")

@app.get("/")
def root():
    return {"status": "API is running"}