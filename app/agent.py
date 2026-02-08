import os
import json
from datetime import datetime
from app.openrouter import run_openrouter
from app.config import (
    SYSTEM_PROMPT, SUMMARY_THRESHOLD, IDENTITY_UPDATE_INTERVAL
)
from app.memory import memory_manager

from app.tools import tool_manager

# global trace for debugging
LAST_TRACE = {}

def run_agent(user_input: str, session_id: str) -> str:
    global LAST_TRACE
    
    # 1. Intercept Debug Commands
    if user_input.startswith("/debug "):
        cmd = user_input.split(" ")[1].lower()
        if cmd == "memory":
            summary = memory_manager.get_summary(session_id)
            identity = ""
            if os.path.exists("memory/identity.md"):
                with open("memory/identity.md", "r") as f: identity = f.read()
            return f"[DEBUG MEMORY]\n\nSESSION SUMMARY:\n{summary}\n\nIDENTITY:\n{identity}"
        
        elif cmd == "trace":
            return f"[DEBUG TRACE]\n\n{json.dumps(LAST_TRACE, indent=2)}"
            
        elif cmd == "identity":
            res = memory_manager.get_identity(user_input, return_raw=True)
            return f"[DEBUG IDENTITY]\n\nSimilarity Score: {res['similarity']:.4f}\n\nContent:\n{res['content']}"

    # 2. Save user message to SQLite
    memory_manager.add_message(session_id, "user", user_input)
    
    # 3. Memory Trace Collection & Injection
    trace = {
        "timestamp": datetime.now().isoformat(),
        "input": user_input,
        "deterministic": {},
        "associative": [],
        "identity": {"triggered": False, "score": 0.0},
        "tool_calls": []
    }
    
    # Phase 2: Deterministic Recall
    relevant_sections = memory_manager.get_relevant_memory(session_id, user_input)
    trace["deterministic"] = {k: True for k in relevant_sections.keys()}
    
    # Phase 3: Associative Recall (with confidence gating)
    skip_content = list(relevant_sections.values())
    raw_associative = memory_manager.get_associative_memory(user_input, skip_content=skip_content, return_raw=True)
    
    associative_injections = []
    for mem in raw_associative:
        sim = mem['similarity']
        trace["associative"].append({"content": mem['content'][:50], "sim": sim, "type": mem['type']})
        if sim >= 0.85:
            associative_injections.append(f"[{mem['type']}]: {mem['content']}")
        elif sim >= 0.70:
            associative_injections.append(f"[{mem['type']} (Potential Match)]: I recall something similar - {mem['content']}")
            
    # Phase 3: Identity-Aware Recall
    id_res = memory_manager.get_identity(user_input, return_raw=True)
    trace["identity"] = {"score": id_res['similarity']}
    identity_prompt = ""
    if id_res['similarity'] >= 0.85:
        identity_prompt = f"IDENTITY (Self-Model):\n{id_res['content']}"
        trace["identity"]["triggered"] = True
    elif id_res['similarity'] >= 0.70:
        identity_prompt = f"IDENTITY (Potential Preference Match):\nNote: The user may prefer - {id_res['content']}"
        trace["identity"]["triggered"] = True
        trace["identity"]["hedged"] = True

    knowledge = memory_manager.get_knowledge()
    
    # 4. Phase 4: Tool Schema Injection
    tool_schemas = tool_manager.get_tool_schemas()
    
    # 5. Build the prompt
    prompt_sections = [
        SYSTEM_PROMPT,
        f"\nAVAILABLE TOOLS:\n{tool_schemas}\n"
        "TO CALL A TOOL, USE THIS JSON FORMAT AT THE END OF YOUR RESPONSE:\n"
        "ACTION: {\"tool\": \"tool_name\", \"params\": {\"arg\": \"val\"}, \"reasoning\": \"why\"}\n"
    ]
    
    if knowledge: prompt_sections.append(f"CORE KNOWLEDGE:\n{knowledge}")
    if identity_prompt: prompt_sections.append(identity_prompt)
    if relevant_sections:
        relevant_text = "\n\n".join([f"RECALLED {k.upper()}:\n{v}" for k, v in relevant_sections.items()])
        prompt_sections.append(f"RELEVANT SESSION MEMORY:\n{relevant_text}")
    if associative_injections:
        prompt_sections.append("[PHASE3] ASSOCIATIVE MEMORIES:\n" + "\n".join(associative_injections))
        
    history = memory_manager.get_history(session_id, limit=10)
    prompt_sections.append("CONVERSATION HISTORY:")
    for msg in history:
        role_label = "Assistant" if msg["role"] == "assistant" else "User"
        prompt_sections.append(f"{role_label}: {msg['content']}")
        
    prompt_sections.append("Assistant:")
    full_prompt = "\n\n".join(prompt_sections)
    
    # 6. Get Response
    agent_response = run_openrouter(full_prompt)
    
    # 7. Phase 4: Parse & Execute Tool Calls
    final_output = agent_response
    if "ACTION:" in agent_response:
        try:
            parts = agent_response.split("ACTION:")
            if len(parts) > 1:
                action_json_str = parts[1].strip()
                if "}" in action_json_str:
                    action_json_str = action_json_str[:action_json_str.rfind("}")+1]
                
                try:
                    action_data = json.loads(action_json_str)
                    tool_name = action_data.get("tool")
                    tool_params = action_data.get("params", {})
                    reasoning = action_data.get("reasoning", "No reasoning provided.")
                    
                    trace["tool_calls"].append({"tool": tool_name, "params": tool_params, "reasoning": reasoning})
                    
                    # Safety Check
                    requires_approval = tool_manager.tools.get(tool_name, {}).get("requires_approval", False)
                    is_approved = any(word in user_input.lower() for word in ["proceed", "approve", "do it", "yes", "ok"])
                    
                    if requires_approval and not is_approved:
                        final_output = (
                            f"{parts[0].strip()}\n\n"
                            f"⚠️ [SAFETY] I want to call '{tool_name}' for the following reason: {reasoning}.\n"
                            f"Parameters: {tool_params}\n"
                            f"Shall I proceed? Please say 'proceed' or 'approve'."
                        )
                    else:
                        tool_result = tool_manager.invoke(tool_name, tool_params, session_id)
                        result_str = json.dumps(tool_result)
                        final_output = f"{parts[0].strip()}\n\n[SYSTEM] Tool '{tool_name}' executed. Result: {result_str}"
                except json.JSONDecodeError as je:
                    print(f"[PHASE4] JSON Parse Error: {je}")
                    final_output = agent_response + f"\n\n[SYSTEM] Error parsing Action JSON: {je}"
        except Exception as e:
            print(f"[PHASE4] General Tool Error: {e}")
            final_output = agent_response + f"\n\n[SYSTEM] Internal error during action processing: {e}"

    LAST_TRACE = trace 
    memory_manager.add_message(session_id, "assistant", final_output)
    
    # 8. Periodic Maintenance
    try:
        msg_count = memory_manager.get_message_count(session_id)
        if msg_count > 0 and msg_count % SUMMARY_THRESHOLD == 0:
            print(f"[MEMORY] Maintenance cycle triggered ({msg_count} msgs)")
            summarize_session(session_id)
            memory_manager.detect_drift(session_id)
            summary_files = [f for f in os.listdir("memory/summaries") if f.endswith(".md")]
            if len(summary_files) > 0 and len(summary_files) % IDENTITY_UPDATE_INTERVAL == 0:
                memory_manager.update_identity()
    except Exception as e:
        print(f"[MEMORY] Maintenance Error: {e}")
        
    return final_output

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
    - OPEN THREADS: A thread is Active if it appears in the latest summary. A thread is RESOLVED if the user explicitly says it is resolved, OR the assistant explicitly confirms completion. Resolved threads MUST be removed and never reappear. No inference allowed.
    - DO NOT INCLUDE: Instructions, generic plans, internal reasoning, jokes, filler, or greetings.
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
