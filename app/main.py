from fastapi import FastAPI
from pydantic import BaseModel
from app.agent import run_agent
from app.config import APP_HOST, APP_PORT

app = FastAPI(title="Private AI Agent API")

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default_session"

class ChatResponse(BaseModel):
    response: str

@app.get("/")
def read_root():
    return {"status": "online", "message": "Private AI Agent is running"}

@app.post("/ask", response_model=ChatResponse)
def ask(req: ChatRequest):
    agent_response = run_agent(req.message, req.session_id)
    return ChatResponse(response=agent_response)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)
