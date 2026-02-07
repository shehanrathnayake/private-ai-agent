from datetime import datetime
from app.openrouter import run_openrouter
from app.config import SYSTEM_PROMPT, SUMMARY_THRESHOLD
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
    
    # 6. Periodic Auto-summarization (Every N messages)
    try:
        msg_count = memory_manager.get_message_count(session_id)
        if msg_count > 0 and msg_count % SUMMARY_THRESHOLD == 0:
            summarize_session(session_id)
    except Exception as e:
        # Safety: Log the error and continue normally
        print(f"Summarization error: {e}")
        
    return agent_response

def summarize_session(session_id: str):
    """Asks the LLM to consolidate conversation into a structured Markdown summary."""
    # 1. Fetch messages since last summary (the last chunk of THRESHOLD messages)
    history = memory_manager.get_messages_since_last_summary(session_id, limit=SUMMARY_THRESHOLD)
    current_summary = memory_manager.get_summary(session_id)
    
    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history])
    
    summary_prompt = f"""
    You are a memory consolidation module. Your task is to update a session summary based on the latest interaction.
    
    RULES:
    - Extract ONLY explicit, stable facts.
    - Extract communication or content preferences.
    - Identify ongoing tasks or unresolved topics as 'Open Threads'.
    - IGNORE jokes, filler, greetings, and transient conversation.
    - Prefer correctness over completeness.
    - If a section has no information, keep it empty.
    
    EXISTING SUMMARY:
    {current_summary if current_summary else "No summary yet."}
    
    LATEST MESSAGES SINCE LAST SUMMARY:
    {history_text}
    
    MANDATORY OUTPUT STRUCTURE:
    # Session Summary: {session_id}

    ## Known Facts
    - <explicit facts only>

    ## Preferences
    - <communication or content preferences>

    ## Open Threads
    - <ongoing tasks or unresolved topics>

    ## Last Updated
    <ISO 8601 timestamp>
    """
    
    new_summary = run_openrouter(summary_prompt)
    
    # Safety: Do not write if the LLM call failed
    if not new_summary.startswith("Error"):
        # Append/Ensure the timestamp if the LLM provided a placeholder or old one
        timestamp = datetime.now().isoformat()
        # Basic cleanup: ensuring the structure is maintained if LLM omitted it
        if "## Last Updated" in new_summary:
            lines = new_summary.split("\n")
            cleaned_lines = []
            for line in lines:
                if line.strip().startswith("## Last Updated"):
                    cleaned_lines.append("## Last Updated")
                    cleaned_lines.append(timestamp)
                    break # Stop adding lines after updating timestamp
                cleaned_lines.append(line)
            final_markdown = "\n".join(cleaned_lines)
        else:
            final_markdown = new_summary.strip() + f"\n\n## Last Updated\n{timestamp}"
            
        memory_manager.write_session_summary(session_id, final_markdown)
