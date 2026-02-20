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

from app.bootstrap import (
    IDENTITY_FILE
)

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
            if os.path.exists(IDENTITY_FILE):
                with open(IDENTITY_FILE, "r") as f: identity = f.read()
            return f"[DEBUG MEMORY]\n\nSESSION SUMMARY:\n{summary}\n\nIDENTITY:\n{identity}"
        
        elif cmd == "trace":
            return f"[DEBUG TRACE]\n\n{json.dumps(LAST_TRACE, indent=2)}"
            
        elif cmd == "identity":
            res = memory_manager.get_identity(user_input, return_raw=True)
            return f"[DEBUG IDENTITY]\n\nSimilarity Score: {res['similarity']:.4f}\n\nContent:\n{res['content']}"

        elif cmd == "introspection":
            from app.introspection import Introspection
            introspection = Introspection(memory_manager)
            return introspection.generate_introspection_report(session_id)

        elif cmd == "thoughts":
            # Sub-command: /debug thoughts [clear|journal]
            parts = user_input.strip().split()
            sub = parts[2].lower() if len(parts) > 2 else ""

            if sub == "clear":
                try:
                    import sqlite3
                    from app.memory import METADATA_DB_PATH
                    with sqlite3.connect(METADATA_DB_PATH) as conn:
                        cursor = conn.cursor()
                        cursor.execute(
                            "DELETE FROM vector_metadata WHERE session_id = '__inner__'"
                        )
                        deleted = cursor.rowcount
                        conn.commit()
                    return (f"[DEBUG THOUGHTS] Cleared {deleted} inner thought(s) from memory.\n"
                            f"Note: FAISS index will be rebuilt on next restart.")
                except Exception as e:
                    return f"[DEBUG THOUGHTS] Clear failed: {e}"

            elif sub == "journal":
                try:
                    import os
                    from app.bootstrap import THOUGHTS_DIR
                    today = datetime.now().strftime("%Y-%m-%d")
                    journal_path = os.path.join(THOUGHTS_DIR, f"{today}.md")
                    if os.path.exists(journal_path):
                        with open(journal_path, "r", encoding="utf-8") as f:
                            content = f.read()
                        return f"[DEBUG THOUGHTS — JOURNAL {today}]\n\n{content}"
                    return f"[DEBUG THOUGHTS] No journal file for today ({today}) yet."
                except Exception as e:
                    return f"[DEBUG THOUGHTS] Journal read failed: {e}"

            else:
                # Default: show stored thoughts with full detail
                try:
                    import sqlite3
                    from app.memory import METADATA_DB_PATH
                    with sqlite3.connect(METADATA_DB_PATH) as conn:
                        cursor = conn.cursor()
                        cursor.execute("""
                            SELECT content, salience, timestamp
                            FROM vector_metadata
                            WHERE session_id = '__inner__' AND merged = 0
                              AND type IN ('Inner Thought', 'Reflection')
                            ORDER BY salience DESC, timestamp DESC
                            LIMIT 10
                        """)
                        rows = cursor.fetchall()
                        cursor.execute(
                            "SELECT COUNT(*) FROM vector_metadata WHERE session_id = '__inner__' AND merged = 0"
                        )
                        total = cursor.fetchone()[0]

                    if not rows:
                        return "[DEBUG THOUGHTS]\n\nNo inner thoughts stored yet."

                    lines = [f"[DEBUG THOUGHTS] {total} thought(s) stored (showing top 10 by salience)\n"]
                    for i, (content, salience, ts) in enumerate(rows, 1):
                        # Trim timestamp to HH:MM on date
                        try:
                            dt = datetime.fromisoformat(ts)
                            ts_fmt = dt.strftime("%b %d %H:%M")
                        except Exception:
                            ts_fmt = ts[:16]
                        lines.append(f"{i}. [{ts_fmt}] salience={salience:.2f}")
                        lines.append(f"   {content}")
                        lines.append("")
                    return "\n".join(lines)

                except Exception as e:
                    return f"[DEBUG THOUGHTS] Read failed: {e}"

        elif cmd == "loop":
            try:
                from app.inner_loop import inner_loop_instance
                status = inner_loop_instance.get_status() if inner_loop_instance else "Not initialized"
                return f"[DEBUG LOOP]\n\n{status}"
            except ImportError:
                return "[DEBUG LOOP]\n\nInner loop module not yet active (Phase C)."

    # 2. Save user message to SQLite
    memory_manager.add_message(session_id, "user", user_input)
    
    # 3. Memory Trace Collection & Injection
    trace = {
        "timestamp": datetime.now().isoformat(),
        "input": user_input,
        "deterministic": {},
        "associative": [],
        "predicted": [],
        "stitched_context": False,
        "identity": {"triggered": False, "score": 0.0},
        "tool_calls": []
    }
    
    # Phase 2: Session Stitching & Temporal Context
    # If this is a new session (< 3 messages) OR user asks "temporal" words, fetch previous summary
    temporal_keywords = ["earlier", "before", "yesterday", "last time", "previous", "remember"]
    is_temporal_query = any(k in user_input.lower() for k in temporal_keywords)
    msg_count = memory_manager.get_message_count(session_id)
    
    previous_summary = ""
    if msg_count < 5 or is_temporal_query:
        previous_summary = memory_manager.get_previous_summary(session_id)
        if previous_summary:
            trace["stitched_context"] = True
            print(f"[CONTEXT] Session Stitching active: pulled previous summary into prompt.")
        else:
            print(f"[CONTEXT] Session Stitching attempted but no previous summary found.")

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
    
    # Behavioral Rules (Phase E)
    behavioral_rules = memory_manager.get_behavioral_rules()
    if behavioral_rules:
        prompt_sections.append(f"[BEHAVIORAL RULES]:\n{behavioral_rules}")
    if relevant_sections:
        relevant_text = "\n\n".join([f"RECALLED {k.upper()}:\n{v}" for k, v in relevant_sections.items()])
        prompt_sections.append(f"RELEVANT SESSION MEMORY:\n{relevant_text}")
    
    if cross_session_summary:
        prompt_sections.append(cross_session_summary)

    if previous_summary:
        prompt_sections.append(
            f"--- CRITICAL RECENT CONTEXT ---\n"
            f"The current session is new, but this is the summary of your VERY LAST conversation with the user. "
            f"If the user asks 'what did we do earlier' or 'remember', use the information below as your primary source of truth.\n\n"
            f"{previous_summary}\n"
            f"--- END RECENT CONTEXT ---"
        )

    # Phase A4: Inner Thought Injection (Meta-Cognitive awareness)
    # If the user asks about Astra's persona, we pull the MOST SALIENT thoughts regardless of recent activity
    persona_query = any(k in user_input.lower() for k in ["who are you", "persona", "personality", "yourself", "how do you feel"])
    inner_thoughts = memory_manager.get_recent_inner_thoughts(top_k=5 if persona_query else 2, user_input=user_input)
    if inner_thoughts:
        prompt_sections.append(inner_thoughts)
    elif persona_query:
        # Fallback: if similarity gated it, pull just the top-salience ones anyway
        inner_thoughts = memory_manager.get_recent_inner_thoughts(top_k=3, user_input="")
        if inner_thoughts:
            prompt_sections.append(inner_thoughts)
        
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

    # Guard: detect API/LLM error responses — do NOT store them in memory.
    # Error strings poison message history, summaries, and associative memory.
    # The user still sees the error, but it stays ephemeral.
    _ERROR_PREFIXES = (
        "Error calling OpenRouter API:",
        "Error calling Gemini API:",
        "[SYSTEM] Internal error",
    )
    is_error_response = any(final_output.startswith(p) for p in _ERROR_PREFIXES)

    if not is_error_response:
        memory_manager.add_message(session_id, "assistant", final_output)
    else:
        print(f"[AGENT] Error response detected — skipping memory storage. ({final_output[:80]}...)")
    
    # 8. Phase 5.2: Memory Reinforcement + Maintenance — skip entirely on error responses
    if not is_error_response:
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

                # Identity and Behavioral Rule formation
                update_count = memory_manager.get_summary_update_count()
                if update_count > 0 and update_count % IDENTITY_UPDATE_INTERVAL == 0:
                    memory_manager.update_identity()
                    
                    # Consolidate Behavioral Rules (Phase E)
                    from app.introspection import Introspection
                    ins = Introspection(memory_manager)
                    ins.consolidate_behavioral_rules()
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
    from app.bootstrap import SUMMARIES_DIR
    print(f"[MEMORY] Summary updated successfully: {SUMMARIES_DIR}/{session_id}.md")

    # [PHASE 7] Automatic Introspection Check
    try:
        from app.introspection import Introspection
        introspection = Introspection(memory_manager)
        report = introspection.generate_introspection_report(session_id, flag_high_priority=True)
        if "[HIGH PRIORITY]" in report or "Contradictions Detected" in report:
            print(f"[INTROSPECTION] Report Generated:\n{report}")
    except Exception as e:
        print(f"[INTROSPECTION] Error during check: {e}")
