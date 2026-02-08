import sqlite3
import os
from datetime import datetime
from typing import List, Dict

DB_PATH = "memory/agent_memory.db"
SUMMARIES_PATH = "memory/summaries"

class MemoryManager:
    def __init__(self):
        self._init_db()
        if not os.path.exists(SUMMARIES_PATH):
            os.makedirs(SUMMARIES_PATH)

    def _init_db(self):
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            # Create messages table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Create index for faster retrieval
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_session ON messages(session_id)")
            conn.commit()

    def add_message(self, session_id: str, role: str, content: str):
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                (session_id, role, content)
            )
            conn.commit()

    def get_history(self, session_id: str, limit: int = 10) -> List[Dict[str, str]]:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT role, content FROM messages WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?",
                (session_id, limit)
            )
            rows = cursor.fetchall()
            # Return in chronological order
            return [{"role": row[0], "content": row[1]} for row in reversed(rows)]

    def get_message_count(self, session_id: str) -> int:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,))
            return cursor.fetchone()[0]

    def get_summary(self, session_id: str) -> str:
        summary_file = os.path.join(SUMMARIES_PATH, f"{session_id}.md")
        if os.path.exists(summary_file):
            with open(summary_file, "r", encoding="utf-8") as f:
                return f.read()
        return ""

    def get_messages_since_last_summary(self, session_id: str, limit: int) -> List[Dict[str, str]]:
        """Fetches the last N messages which represent the chunk since the last summary trigger."""
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT role, content FROM messages WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?",
                (session_id, limit)
            )
            rows = cursor.fetchall()
            return [{"role": row[0], "content": row[1]} for row in reversed(rows)]

    def write_session_summary(self, session_id: str, markdown: str):
        summary_file = os.path.join(SUMMARIES_PATH, f"{session_id}.md")
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(markdown)

    def save_summary(self, session_id: str, summary: str):
        # Kept for backward compatibility if needed, but redirects to write_session_summary
        self.write_session_summary(session_id, summary)

    def get_knowledge(self) -> str:
        knowledge_file = "memory/knowledge.md"
        if os.path.exists(knowledge_file):
            with open(knowledge_file, "r", encoding="utf-8") as f:
                return f.read()
        return ""

    def parse_summary_sections(self, summary_text: str) -> Dict[str, str]:
        """Parses the summary text into structured sections based on mandatory headers."""
        sections = {
            "Known Facts": "",
            "Preferences": "",
            "Open Threads": ""
        }
        
        current_section = None
        lines = summary_text.split("\n")
        
        for line in lines:
            if "## Known Facts" in line:
                current_section = "Known Facts"
            elif "## Preferences" in line:
                current_section = "Preferences"
            elif "## Open Threads" in line:
                current_section = "Open Threads"
            elif "## Last Updated" in line:
                current_section = None
            elif current_section and line.strip():
                # Avoid capturing placeholders like <explicit facts only>
                if not (line.strip().startswith("- <") and line.strip().endswith(">")):
                    sections[current_section] += line + "\n"
        
        return {k: v.strip() for k, v in sections.items()}

    def get_relevant_memory(self, session_id: str, user_input: str) -> str:
        """Deterministic rule-based recall logic to select relevant sections of the summary."""
        summary_text = self.get_summary(session_id)
        if not summary_text:
            return ""
            
        sections = self.parse_summary_sections(summary_text)
        relevant_parts = []
        user_input_lower = user_input.lower()
        
        # Rule 1: Name/Identity -> Known Facts
        name_keywords = ["name", "who am i", "remember me", "identity"]
        if any(kw in user_input_lower for kw in name_keywords) and sections["Known Facts"]:
            relevant_parts.append(f"RECALLED KNOWN FACTS:\n{sections['Known Facts']}")
            
        # Rule 2: Preferences/Explicit Likes -> Preferences
        pref_keywords = ["preference", "like", "dislike", "favorite", "style", "prefer"]
        if any(kw in user_input_lower for kw in pref_keywords) and sections["Preferences"]:
            relevant_parts.append(f"RECALLED PREFERENCES:\n{sections['Preferences']}")
            
        # Rule 3: Continuation/Status -> Open Threads
        cont_keywords = ["continue", "what about", "status", "next", "todo", "progress", "unfinished"]
        if any(kw in user_input_lower for kw in cont_keywords) and sections["Open Threads"]:
            relevant_parts.append(f"RECALLED OPEN THREADS:\n{sections['Open Threads']}")
            
        return "\n\n".join(relevant_parts)

# Global instance
memory_manager = MemoryManager()
