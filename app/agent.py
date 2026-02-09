import os
import json
from datetime import datetime
from app.openrouter import run_openrouter
from app.config import (
    SYSTEM_PROMPT, SUMMARY_THRESHOLD, IDENTITY_UPDATE_INTERVAL,
    REINFORCE_AMOUNT_ASSOCIATIVE, REINFORCE_AMOUNT_DETERMINISTIC,
    REINFORCE_AMOUNT_IDENTITY, EFFECTIVE_THRESHOLD
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
        "predicted": [],
        "identity": {"triggered": False, "score": 0.0},
        "tool_calls": []
    }
    
    # Phase 2: Deterministic Recall
    # Phase 2: Deterministic Recall
    relevant_res = memory_manager.get_relevant_memory(session_id, user_input)
    relevant_sections = relevant_res["sections"]
    deterministic_vector_ids = relevant_res["vector_ids"]
    trace["deterministic"] = {k: True for k in relevant_sections.keys()}
    
    # Phase 3 & 6: Unified Cross-Session Context (Aging-Aware & Predictive)
    skip_content = list(relevant_sections.values())
    cross_session_summary = memory_manager.summarize_cross_session_context(user_input, top_k=5)
    trace["cross_session_summary"] = cross_session_summary
    
    # Still need individual IDs for reinforcement logic and tracing
    raw_associative = memory_manager.get_aging_aware_associative(user_input, skip_content=skip_content)
    used_associative_vector_ids = []
    for mem in raw_associative:
        used_associative_vector_ids.append(mem['vector_id'])
        trace["associative"].append({
            "content": mem['content'][:50], 
            "score": mem['effective_score'], 
            "type": mem['type']
        })

    # Phase 3: Identity-Aware Recall
    id_res = memory_manager.get_identity(user_input, return_raw=True)
    trace["identity"] = {"score": id_res['similarity']}
    identity_prompt = ""
    identity_vector_ids = id_res.get("vector_ids", [])
    used_identity_vector_ids = []
    
    if id_res['similarity'] >= 0.85:
        identity_prompt = f"IDENTITY (Self-Model):\n{id_res['content']}"
        trace["identity"]["triggered"] = True
        used_identity_vector_ids = identity_vector_ids
    elif id_res['similarity'] >= 0.70:
        identity_prompt = f"IDENTITY (Potential Preference Match):\nNote: The user may prefer - {id_res['content']}"
        trace["identity"]["triggered"] = True
        trace["identity"]["hedged"] = True
        used_identity_vector_ids = identity_vector_ids

    knowledge = memory_manager.get_knowledge()
    
    # 4. Phase 4: Tool Schema Injection
    tool_schemas = tool_manager.get_tool_schemas()
    
    # 5. Build the prompt
    prompt_sections = [
        SYSTEM_PROMPT,
        f"\nAVAILABLE TOOLS:\n{tool_schemas}\n"
        "TO CALL A TOOL, YOU MUST USE THE DELIMITER ---ACTION--- AT THE VERY END OF YOUR RESPONSE.\n"
        "ANYTHING AFTER THIS DELIMITER MUST BE VALID JSON.\n"
        "FORMAT:\n"
        "---ACTION---\n"
        "{\"tool\": \"tool_name\", \"params\": {\"arg\": \"val\"}, \"reasoning\": \"why\"}\n"
    ]
    
    if knowledge: prompt_sections.append(f"CORE KNOWLEDGE:\n{knowledge}")
    if identity_prompt: prompt_sections.append(identity_prompt)
    if relevant_sections:
        relevant_text = "\n\n".join([f"RECALLED {k.upper()}:\n{v}" for k, v in relevant_sections.items()])
        prompt_sections.append(f"RELEVANT SESSION MEMORY:\n{relevant_text}")
    
    if cross_session_summary:
        prompt_sections.append(cross_session_summary)
        
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
    if "---ACTION---" in agent_response:
        try:
            # Use rsplit to honor only the FINAL delimiter
            parts = agent_response.rsplit("---ACTION---", 1)
            if len(parts) > 1:
                prefix_text = parts[0].strip()
                action_json_str = parts[1].strip()
                
                # Clean up markdown blocks
                if "```json" in action_json_str:
                    action_json_str = action_json_str.split("```json")[1].split("```")[0].strip()
                elif "```" in action_json_str:
                    action_json_str = action_json_str.split("```")[1].split("```")[0].strip()

                try:
                    action_data = json.loads(action_json_str)
                    tool_name = action_data.get("tool")
                    tool_params = action_data.get("params", {})
                    reasoning = action_data.get("reasoning", "No reasoning provided.")
                    
                    # 1. Unknown Tool Check (Fail Fast)
                    if tool_name not in tool_manager.tools:
                        final_output = f"{prefix_text}\n\n[SYSTEM] Unknown tool '{tool_name}'. Action aborted."
                    else:
                        trace["tool_calls"].append({"tool": tool_name, "params": tool_params, "reasoning": reasoning})
                        
                        # 2. Safety & Approval Check
                        tool_info = tool_manager.tools[tool_name]
                        requires_approval = tool_info.get("requires_approval", False)
                        
                        # Exact match approval
                        is_approved = user_input.strip() in ["/approve", "/proceed"]
                        
                        if requires_approval and not is_approved:
                            final_output = (
                                f"{prefix_text}\n\n"
                                f"⚠️ [SAFETY] I want to call '{tool_name}' for the following reason: {reasoning}.\n"
                                f"Parameters: {tool_params}\n"
                                f"Shall I proceed? Please type '/approve' or '/proceed' to confirm."
                            )
                        else:
                            # 3. Execution
                            tool_result = tool_manager.invoke(tool_name, tool_params, session_id)
                            result_str = json.dumps(tool_result)
                            final_output = f"{prefix_text}\n\n[SYSTEM] Tool '{tool_name}' executed. Result: {result_str}"
                            
                            # 4. Success Persistence (Add to Associative Memory)
                            if tool_result.get("status") == "success":
                                action_summary = f"Tool executed: {tool_name} -> {result_str[:100]}"
                                memory_manager.add_vector(action_summary, session_id, "Open Threads", salience=0.8)
                                
                except json.JSONDecodeError as je:
                    # abort action safely
                    final_output = agent_response + f"\n\n[SYSTEM] Action aborted: Malformed JSON after delimiter."
        except Exception as e:
            print(f"[PHASE4] General Tool Error: {e}")
            final_output = agent_response + f"\n\n[SYSTEM] Internal error during action processing."

    LAST_TRACE = trace 
    memory_manager.add_message(session_id, "assistant", final_output)
    
    # 8. Phase 5.2: Memory Reinforcement
    # Reinforce memories that were successfully recalled and used this turn
    # Use sets to avoid duplicate reinforcement in the same turn (Issue 6)
    for vid in set(used_associative_vector_ids):
        memory_manager.reinforce_memory(vid, REINFORCE_AMOUNT_ASSOCIATIVE, source="associative")
    for vid in set(deterministic_vector_ids):
        memory_manager.reinforce_memory(vid, REINFORCE_AMOUNT_DETERMINISTIC, source="deterministic")
    for vid in set(used_identity_vector_ids):
        memory_manager.reinforce_memory(vid, REINFORCE_AMOUNT_IDENTITY, source="identity")

    # 9. Periodic Maintenance
    try:
        msg_count = memory_manager.get_message_count(session_id)
        if msg_count > 0 and msg_count % SUMMARY_THRESHOLD == 0:
            print(f"[MEMORY] Maintenance cycle triggered ({msg_count} msgs)")
            summarize_session(session_id)
            memory_manager.detect_drift(session_id)
            memory_manager.decay_salience()
            memory_manager.compress_and_merge_memory()
            
            # Identity formation triggers based on total summary volume (experience), not file count.
            update_count = memory_manager.get_summary_update_count()
            if update_count > 0 and update_count % IDENTITY_UPDATE_INTERVAL == 0:
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
