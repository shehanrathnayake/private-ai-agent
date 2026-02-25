from fastapi import FastAPI
from pydantic import BaseModel
from app.agent import run_agent
from app.config import APP_HOST, APP_PORT
from app.bootstrap import ensure_runtime_environment

# Ensure filesystem is ready before anything else
ensure_runtime_environment()

# --- Initialize Modular Skills ---
from app.skills import skill_registry
import os
skill_registry.discover_local_skills()
config_path = os.path.join(os.path.dirname(__file__), "skills", "config.json")
skill_registry.discover_mcp_skills(config_path)
# ---------------------------------

# --- Inner Monologue: start background thinking thread ---
from app.memory import memory_manager
from app.inner_loop import InnerLoop
import app.inner_loop as _inner_loop_module

_loop = InnerLoop(memory_manager)
_inner_loop_module.inner_loop_instance = _loop  # expose for /debug loop
_loop.start()
# ---------------------------------------------------------

from app.session_broker import session_broker

app = FastAPI(title="Private AI Agent API")

class ChatRequest(BaseModel):
    message: str
    session_id: str = "auto"

class ChatResponse(BaseModel):
    response: str

@app.get("/")
def read_root():
    return {"status": "online", "message": "Private AI Agent is running"}

@app.post("/ask", response_model=ChatResponse)
async def ask(req: ChatRequest):
    # Resolve the session ID automatically if 'auto' or not provided
    resolved_id = session_broker.resolve_session_id(req.message, req.session_id)
    agent_response = await run_agent(req.message, resolved_id)
    return ChatResponse(response=agent_response)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)

