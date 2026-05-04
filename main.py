from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag import ingest_pdf, ask_question
from dotenv import load_dotenv
import os

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError("GROQ_API_KEY not found. Check your .env file.")

app = FastAPI(title="Doc Q&A API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:4173",
        # Add your deployed Vercel URL here when deploying:
        # "https://your-app.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Question(BaseModel):
    question: str


@app.get("/")
def root():
    return {"status": "running", "message": "Doc Q&A API is live"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    contents = await file.read()

    if len(contents) > 20 * 1024 * 1024:  # 20MB limit
        raise HTTPException(status_code=400, detail="File too large. Max size is 20MB.")

    message = ingest_pdf(contents, file.filename)
    return {"message": message, "filename": file.filename}


@app.post("/ask")
async def ask(q: Question):
    if not q.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    return ask_question(q.question)
