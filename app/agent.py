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
            print(f"[MEMORY] Summarization triggered for session {session_id} (count: {msg_count})")
            summarize_session(session_id)
    except Exception as e:
        # Safety: Log the error and continue normally
        print(f"[MEMORY] Summarization error: {e}")
        
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
    - CONSOLIDATE: Use the existing summary and the latest messages to create an updated narrative. Do not delete stable facts from the previous summary unless they have been explicitly contradicted or changed.
    - SCOPE: Extract ONLY explicit, stable facts and user/assistant-stated preferences.
    - DO NOT INCLUDE: Instructions, future plans, internal reasoning, jokes, filler, or greetings.
    - STRUCTURE: You must strictly follow the mandatory output structure provided below.
    - ACCURACY: Prefer correctness over completeness. If a section has no information, leave it empty.
    
    EXISTING SUMMARY:
    {current_summary if current_summary else "No summary yet."}
    
    LATEST MESSAGES SINCE LAST SUMMARY ({SUMMARY_THRESHOLD} messages):
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
    
    # 2. Safety & Validation
    if new_summary.startswith("Error"):
        print(f"[MEMORY] Summarization failed for {session_id}: API Error")
        return

    # Mandatory Header Protection
    mandatory_headers = ["# Session Summary:", "## Known Facts", "## Preferences", "## Open Threads"]
    if not all(header in new_summary for header in mandatory_headers):
        print(f"[MEMORY] Summarization rejected for {session_id}: Malformed output (missing headers)")
        return

    # 3. Success: Finalize and Write
    timestamp = datetime.now().isoformat()
    
    # Ensure the timestamp is correctly placed at the end
    if "## Last Updated" in new_summary:
        parts = new_summary.split("## Last Updated")
        final_markdown = parts[0].strip() + f"\n\n## Last Updated\n{timestamp}"
    else:
        final_markdown = new_summary.strip() + f"\n\n## Last Updated\n{timestamp}"
        
    memory_manager.write_session_summary(session_id, final_markdown)
    print(f"[MEMORY] Summary updated successfully: memory/summaries/{session_id}.md")
