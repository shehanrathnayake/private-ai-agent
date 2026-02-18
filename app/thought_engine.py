"""
thought_engine.py — Phase B: The Private Thinker

Responsible for:
  1. Building a perception snapshot from the agent's current memory state
  2. Generating a private, structured inner thought via the LLM
  3. Returning a parsed Thought dataclass — never shown directly to the user

Design principles followed:
  - Thoughts start at low salience (INNER_THOUGHT_SALIENCE = 0.3)
  - "Nothing new" guard: skips LLM call if no user activity since last thought
  - Strict output parser with silent fallback — never crashes the loop
  - Completely separate prompt from the user-facing SYSTEM_PROMPT
"""

import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from app.config import (
    INNER_THOUGHT_SALIENCE,
)
from app.bootstrap import AGENT_DB, IDENTITY_FILE, KNOWLEDGE_FILE, SUMMARIES_DIR as SUMMARIES_PATH


# ---------------------------------------------------------------------------
# Thought dataclass — the structured output of one thinking cycle
# ---------------------------------------------------------------------------

@dataclass
class Thought:
    raw_thought: str            # The private reasoning text
    salience: float             # 0.0–1.0, starts low
    should_act: bool            # Does this thought require an external action?
    action_hint: str            # What kind of action, if any
    memory_worthy: bool         # Should this be stored in long-term memory?
    skipped: bool = False       # True if the cycle was skipped (nothing new)
    skip_reason: str = ""       # Why it was skipped, for logging


# ---------------------------------------------------------------------------
# PerceptionBuilder — reads the agent's current memory state
# ---------------------------------------------------------------------------

class PerceptionBuilder:
    """
    Builds a plain-text perception snapshot from memory.
    This is the agent's "window to the world" — entirely internal.
    No LLM calls here, just reading existing data.
    """

    def __init__(self, memory_manager):
        self.mm = memory_manager

    def build(self, last_thought_timestamp: Optional[datetime] = None) -> dict:
        """
        Returns a dict with:
          - time_now: current ISO timestamp
          - time_since_last_message: seconds since last user message (or None)
          - last_messages: last 3 user/assistant messages across all sessions
          - open_threads: current open threads from most recent session summary
          - identity_snippet: first 500 chars of identity.md (if exists)
          - knowledge_snippet: first 300 chars of knowledge.md (if exists)
          - pending_reminders: count of pending reminders in DB
          - pending_tasks: count of pending tasks in DB
          - has_new_activity: True if user messaged since last thought
        """
        now = datetime.now()
        perception = {
            "time_now": now.isoformat(),
            "time_since_last_message": None,
            "last_messages": [],
            "open_threads": "",
            "identity_snippet": "",
            "knowledge_snippet": "",
            "pending_reminders": 0,
            "pending_tasks": 0,
            "has_new_activity": False,
        }

        # 1. Last messages + time since last user message
        try:
            with sqlite3.connect(AGENT_DB) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT role, content, timestamp
                    FROM messages
                    ORDER BY timestamp DESC
                    LIMIT 6
                """)
                rows = cursor.fetchall()

            if rows:
                # Most recent message timestamp
                last_ts_str = rows[0][2]
                try:
                    last_ts = datetime.fromisoformat(last_ts_str)
                    delta_seconds = (now - last_ts).total_seconds()
                    perception["time_since_last_message"] = delta_seconds

                    # "Nothing new" guard: has there been activity since last thought?
                    if last_thought_timestamp is None or last_ts > last_thought_timestamp:
                        perception["has_new_activity"] = True
                except (ValueError, TypeError):
                    perception["has_new_activity"] = True  # Assume new if can't parse

                # Last 3 messages (reversed to chronological order)
                for role, content, ts in reversed(rows[:6]):
                    label = "User" if role == "user" else "Agent"
                    perception["last_messages"].append(
                        f"{label}: {content[:120]}{'...' if len(content) > 120 else ''}"
                    )
        except Exception as e:
            print(f"[THOUGHT] PerceptionBuilder: message read error: {e}")

        # 2. Open threads from most recent session summary
        try:
            if os.path.exists(SUMMARIES_PATH):
                summary_files = sorted(
                    [f for f in os.listdir(SUMMARIES_PATH) if f.endswith(".md")],
                    key=lambda x: os.path.getmtime(os.path.join(SUMMARIES_PATH, x)),
                    reverse=True
                )
                if summary_files:
                    latest = os.path.join(SUMMARIES_PATH, summary_files[0])
                    with open(latest, "r", encoding="utf-8") as f:
                        summary_text = f.read()
                    sections = self.mm.parse_summary_sections(summary_text)
                    perception["open_threads"] = sections.get("Open Threads", "").strip()
        except Exception as e:
            print(f"[THOUGHT] PerceptionBuilder: summary read error: {e}")

        # 3. Identity snippet
        try:
            if os.path.exists(IDENTITY_FILE):
                with open(IDENTITY_FILE, "r", encoding="utf-8") as f:
                    perception["identity_snippet"] = f.read()[:500]
        except Exception as e:
            print(f"[THOUGHT] PerceptionBuilder: identity read error: {e}")

        # 4. Knowledge snippet
        try:
            if os.path.exists(KNOWLEDGE_FILE):
                with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
                    perception["knowledge_snippet"] = f.read()[:300]
        except Exception as e:
            print(f"[THOUGHT] PerceptionBuilder: knowledge read error: {e}")

        # 5. Pending reminders and tasks
        try:
            with sqlite3.connect(AGENT_DB) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT COUNT(*) FROM reminders WHERE status = 'pending'"
                )
                perception["pending_reminders"] = cursor.fetchone()[0]

                cursor.execute(
                    "SELECT COUNT(*) FROM tasks WHERE status = 'todo'"
                )
                perception["pending_tasks"] = cursor.fetchone()[0]
        except Exception as e:
            # Tables may not exist yet — that's fine
            pass

        return perception

    def format_for_prompt(self, perception: dict) -> str:
        """Formats the perception dict into a clean text block for the LLM."""
        lines = []

        lines.append(f"Current time: {perception['time_now']}")

        if perception["time_since_last_message"] is not None:
            mins = int(perception["time_since_last_message"] // 60)
            lines.append(f"Time since last user message: {mins} minute(s) ago")
        else:
            lines.append("Time since last user message: Unknown (no messages yet)")

        if perception["pending_reminders"] > 0 or perception["pending_tasks"] > 0:
            lines.append(
                f"Pending items: {perception['pending_reminders']} reminder(s), "
                f"{perception['pending_tasks']} task(s)"
            )

        if perception["open_threads"]:
            lines.append(f"\nOpen threads from last session:\n{perception['open_threads']}")

        if perception["last_messages"]:
            lines.append("\nRecent conversation:")
            lines.extend(f"  {m}" for m in perception["last_messages"])

        if perception["identity_snippet"]:
            lines.append(f"\nUser self-model (excerpt):\n{perception['identity_snippet']}")

        if perception["knowledge_snippet"]:
            lines.append(f"\nCore knowledge (excerpt):\n{perception['knowledge_snippet']}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ThoughtEngine — generates private thoughts via the LLM
# ---------------------------------------------------------------------------

class ThoughtEngine:
    """
    Generates one private inner thought per cycle.
    Uses a completely separate prompt from the user-facing system prompt.
    """

    # The private thought prompt — introspective, not conversational
    THOUGHT_PROMPT_TEMPLATE = """\
You are the internal reasoning module of an AI assistant named Astra.
You are thinking PRIVATELY. The user cannot see this. No response is needed.

Your job is to silently reflect on the current situation and decide:
1. What is worth remembering or flagging?
2. Is there anything the user might need next time they talk to you?
3. Are there any open tasks or threads that seem stalled or important?
4. What patterns do you notice about the user's work or interests?

Keep your thought focused, specific, and grounded in the actual context below.
Do NOT repeat obvious facts. Do NOT be generic. Think like a thoughtful assistant
who has been quietly observing and wants to be genuinely useful.

--- CURRENT STATE ---
{perception}
--- END STATE ---

Respond ONLY in this exact format (no extra text before or after):
THOUGHT: <your private reasoning, 1-3 sentences, specific and grounded>
SALIENCE: <a number from 0.1 to 0.9 — how important/useful is this thought?>
ACT: <yes or no — does this require an external action right now?>
ACTION_HINT: <if ACT is yes, briefly describe what action; otherwise write "none">
STORE: <yes or no — is this worth storing in long-term memory?>
"""

    def __init__(self, memory_manager):
        self.mm = memory_manager
        self.perception_builder = PerceptionBuilder(memory_manager)

    def think(self, last_thought_timestamp: Optional[datetime] = None) -> Thought:
        """
        Main entry point. Builds perception, calls LLM, parses output.
        Returns a Thought — never raises an exception.
        """
        # Step 1: Build perception snapshot
        perception = self.perception_builder.build(last_thought_timestamp)

        # Step 2: "Nothing new" guard
        # If no user activity since last thought AND last thought was recent, skip
        if not perception["has_new_activity"]:
            delta = perception.get("time_since_last_message")
            # If last message was more than 24 hours ago and nothing new, skip
            if delta is not None and delta > 86400:
                return Thought(
                    raw_thought="",
                    salience=0.0,
                    should_act=False,
                    action_hint="none",
                    memory_worthy=False,
                    skipped=True,
                    skip_reason="No user activity in >24h and no new messages since last thought."
                )

        # Step 3: Format perception for prompt
        perception_text = self.perception_builder.format_for_prompt(perception)

        # Step 4: Call LLM with private thought prompt
        try:
            from app.openrouter import run_openrouter
            prompt = self.THOUGHT_PROMPT_TEMPLATE.format(perception=perception_text)
            raw_response = run_openrouter(prompt)
        except Exception as e:
            print(f"[THOUGHT] LLM call failed: {e}")
            return Thought(
                raw_thought="",
                salience=0.0,
                should_act=False,
                action_hint="none",
                memory_worthy=False,
                skipped=True,
                skip_reason=f"LLM error: {e}"
            )

        # Step 5: Parse the structured response
        return self._parse_response(raw_response)

    def _parse_response(self, raw: str) -> Thought:
        """
        Parses the structured LLM output into a Thought.
        Silent fallback on any parse error — never crashes.
        """
        defaults = {
            "THOUGHT": "",
            "SALIENCE": str(INNER_THOUGHT_SALIENCE),
            "ACT": "no",
            "ACTION_HINT": "none",
            "STORE": "yes",
        }

        try:
            parsed = dict(defaults)
            for line in raw.strip().splitlines():
                for key in defaults:
                    if line.upper().startswith(f"{key}:"):
                        value = line[len(key) + 1:].strip()
                        parsed[key] = value
                        break

            # Validate and clamp salience
            try:
                salience = float(parsed["SALIENCE"])
                salience = max(0.1, min(0.9, salience))  # Never allow 0.0 or 1.0 from LLM
            except (ValueError, TypeError):
                salience = INNER_THOUGHT_SALIENCE

            thought_text = parsed["THOUGHT"]

            # Reject empty or suspiciously short thoughts
            if not thought_text or len(thought_text) < 10:
                return Thought(
                    raw_thought="",
                    salience=0.0,
                    should_act=False,
                    action_hint="none",
                    memory_worthy=False,
                    skipped=True,
                    skip_reason="LLM returned empty or malformed thought."
                )

            return Thought(
                raw_thought=thought_text,
                salience=salience,
                should_act=parsed["ACT"].lower() == "yes",
                action_hint=parsed["ACTION_HINT"],
                memory_worthy=parsed["STORE"].lower() == "yes",
            )

        except Exception as e:
            print(f"[THOUGHT] Parse error: {e} | Raw: {raw[:100]}")
            return Thought(
                raw_thought="",
                salience=0.0,
                should_act=False,
                action_hint="none",
                memory_worthy=False,
                skipped=True,
                skip_reason=f"Parse error: {e}"
            )
