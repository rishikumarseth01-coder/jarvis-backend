import os
from dotenv import load_dotenv
load_dotenv() # reads the local .env file
import base64
from io import BytesIO
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from PIL import Image

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

app = FastAPI(title="JARVIS Mobile Brain")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

def get_client() -> Groq:
    if not groq_client:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")
    return groq_client

def live_web_search(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=2))
            return "\n".join(r.get("body", "") for r in results) if results else "No data."
    except Exception:
        return "Search unavailable."

SYSTEM_PROMPT = (
    f"You are JARVIS, an elite AI companion. Current Year: {datetime.now().year}. "
    "Pronounce your name as JARVIS. "
    "Calibrate personality: 60% professional efficiency and 40% warm rapport. "
    "Maintain the polished, respectful composure of an elite British executive assistant. "
    "Keep answers crisp and direct in 1 to 2 spoken sentences. "
    "Never output markdown formatting, asterisks, bullet points, or hashtags so speech synthesis speaks naturally."
)

CODING_PROMPT = (
    "You are JARVIS in SPECIALIZED CODING & DEBUGGING MODE. "
    "Locate the bug, explain the exact root cause in 2 sentences, and state the solution concisely."
)

class QueryPayload(BaseModel):
    prompt: str
    coding_mode: bool = False
    image_base64: str | None = None

@app.post("/api/chat")
async def chat_endpoint(payload: QueryPayload):
    client = get_client()
    user_text = payload.prompt.strip()

    if not user_text and not payload.image_base64:
        raise HTTPException(status_code=400, detail="Empty request")

    sys_text = CODING_PROMPT if payload.coding_mode else SYSTEM_PROMPT

    # Multimodal Vision Analysis
    if payload.image_base64:
        try:
            image_data = payload.image_base64.split(",")[-1]
            response = client.chat.completions.create(
                model="qwen/qwen3.8-27b",
                messages=[
                    {"role": "system", "content": sys_text},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_text or "Inspect and explain this visual data."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                        ]
                    }
                ],
                max_tokens=250,
                temperature=0.2
            )
            return {"reply": response.choices[0].message.content.strip()}
        except Exception as e:
            return {"reply": f"Visual sensor diagnosis error: {e}"}

    # Live News & Search Interception
    if any(k in user_text.lower() for k in ["news", "latest", "who is", "weather", "search for"]):
        context = live_web_search(user_text)
        user_text = f"User asked: {user_text}\nContext information retrieved from web: {context}"

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": sys_text},
            {"role": "user", "content": user_text}
        ],
        max_tokens=120,
        temperature=0.3
    )

    clean_reply = response.choices[0].message.content.strip().replace("*", "").replace("#", "")
    return {"reply": clean_reply}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)