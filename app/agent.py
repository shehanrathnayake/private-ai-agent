from app.openrouter import run_openrouter
from app.config import SYSTEM_PROMPT
from app.memory import memory_manager

def run_agent(user_input: str, session_id: str) -> str:
    # 1. Save user message to SQLite
    memory_manager.add_message(session_id, "user", user_input)
    
    # 2. Retrieve history and context
    history = memory_manager.get_history(session_id, limit=10)
    summary = memory_manager.get_summary(session_id)
    knowledge = memory_manager.get_knowledge()
    
    # 3. Build the prompt with memory
    prompt_sections = [SYSTEM_PROMPT]
    
    if knowledge:
        prompt_sections.append(f"CORE KNOWLEDGE:\n{knowledge}")
    
    if summary:
        prompt_sections.append(f"PREVIOUS SUMMARY:\n{summary}")
        
    prompt_sections.append("CONVERSATION HISTORY:")
    for msg in history:
        role_label = "Assistant" if msg["role"] == "assistant" else "User"
        prompt_sections.append(f"{role_label}: {msg['content']}")
        
    # We add the final indicator for the LLM
    prompt_sections.append("Assistant:")
    
    full_prompt = "\n\n".join(prompt_sections)
    
    # 4. Get response from LLM
    agent_response = run_openrouter(full_prompt)
    
    # 5. Save assistant response to SQLite
    memory_manager.add_message(session_id, "assistant", agent_response)
    
    # 6. Periodic Auto-summarization (Every 10 messages)
    try:
        msg_count = memory_manager.get_message_count(session_id)
        if msg_count > 0 and msg_count % 10 == 0:
            summarize_session(session_id)
    except Exception as e:
        print(f"Summarization error: {e}")
        
    return agent_response

def summarize_session(session_id: str):
    """Asks the LLM to summarize the session and saves it to a Markdown file."""
    # Get last 20 messages for context
    history = memory_manager.get_history(session_id, limit=20)
    current_summary = memory_manager.get_summary(session_id)
    
    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history])
    
    summary_prompt = f"""
    Below is a conversation history and an existing summary. 
    Create a concise, updated summary (max 3-4 sentences) that captures the key points, user preferences, and important facts learned so far.
    
    EXISTING SUMMARY:
    {current_summary if current_summary else "No summary yet."}
    
    LATEST MESSAGES:
    {history_text}
    
    UPDATED SUMMARY:
    """
    
    new_summary = run_openrouter(summary_prompt)
    if not new_summary.startswith("Error"):
        memory_manager.save_summary(session_id, new_summary.strip())
