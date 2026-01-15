import os
import httpx
from datetime import datetime
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load .env
load_dotenv()

app = FastAPI(title="F1 R&D AI Assistant")

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration from Environment
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = os.getenv("OPENROUTER_URL")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL")
HTTP_REFERER = os.getenv("SITE_URL", "http://localhost:3000")
X_TITLE = os.getenv("SITE_NAME", "F1 Engineering Studio")

# In-memory history (Session Management)
conversations: Dict[str, List[Dict]] = {}

SYSTEM_PROMPT = """You are a senior F1 (Formula 1) R&D Engineer.
Provide technical, actionable advice on aerodynamics, chassis stability, 
suspension kinematics, and race strategy. Keep responses concise and engineering-focused."""

# --- SCHEMAS ---
class ChatRequest(BaseModel):
    session_id: str = Field(..., example="SESSION_001")
    message: str = Field(..., example="How does ride height affect downforce?")
    model: Optional[str] = Field(default=None)

class ChatResponse(BaseModel):
    session_id: str
    response: str
    timestamp: str

# --- CORE LOGIC ---
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="API Key configuration missing.")

    try:
        # 1. Setup Session
        if request.session_id not in conversations:
            conversations[request.session_id] = []
        
        conversations[request.session_id].append({"role": "user", "content": request.message})
        
        # 2. Model Selection Logic
        selected_model = request.model or DEFAULT_MODEL
        
        # 3. API Call
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": HTTP_REFERER,
                    "X-Title": X_TITLE,
                },
                json={
                    "model": selected_model,
                    "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + conversations[request.session_id],
                    "max_tokens": 500,
                    "temperature": 0.7,
                }
            )

        # 4. Handle Response
        if response.status_code != 200:
            error_data = response.json()
            error_msg = error_data.get("error", {}).get("message", "Unknown AI Core Error")
            raise HTTPException(status_code=response.status_code, detail=f"AI Core Error: {error_msg}")

        data = response.json()
        ai_text = data["choices"][0]["message"]["content"]
        
        # Save to history
        conversations[request.session_id].append({"role": "assistant", "content": ai_text})
        
        return ChatResponse(
            session_id=request.session_id,
            response=ai_text,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        if isinstance(e, HTTPException): raise e
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {"status": "ONLINE", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)