from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import tempfile
import os

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = None
current_filename = None


def ingest_pdf(file_bytes: bytes, filename: str) -> str:
    global vectorstore, current_filename

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(file_bytes)
        tmp_path = f.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(chunks, embeddings)
    current_filename = filename
    os.unlink(tmp_path)
    return f"Ingested {len(chunks)} chunks from '{filename}'"


def ask_question(question: str) -> dict:
    if not vectorstore:
        return {
            "answer": "No document loaded. Please upload a PDF first.",
            "sources": []
        }

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True
    )

    result = qa.invoke({"query": question})

    sources = list(set([
        f"Page {int(d.metadata.get('page', 0)) + 1}"
        for d in result["source_documents"]
    ]))
    sources.sort(key=lambda x: int(x.split(" ")[1]))

    return {
        "answer": result["result"],
        "sources": sources,
        "document": current_filename
    }