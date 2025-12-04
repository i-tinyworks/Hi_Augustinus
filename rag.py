import os
from openai import OpenAI
from supabase import create_client
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://api.cerebras.ai/v1",
    api_key=os.getenv("CEREBRAS_API_KEY")
)

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# -------------------------------
# 임베딩 생성
# -------------------------------
def embed_text(text: str):
    res = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return res.data[0].embedding

# -------------------------------
# PDF 텍스트 추출
# -------------------------------
def extract_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# -------------------------------
# Supabase 저장
# -------------------------------
def save_document(text: str):
    vector = embed_text(text)
    supabase.table("documents").insert({
        "content": text,
        "embedding": vector
    }).execute()

# -------------------------------
# Vector Search
# -------------------------------
def search(query: str, k=5):
    query_emb = embed_text(query)

    resp = supabase.rpc(
        "match_documents",
        {
            "query_embedding": query_emb,
            "match_count": k
        }
    ).execute()

    return [d["content"] for d in resp.data]
