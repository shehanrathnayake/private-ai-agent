import sqlite3
from app.skills.base import BaseSkill
from app.bootstrap import AGENT_DB

class ReminderSkill(BaseSkill):
    name = "set_reminder"
    description = "Set a reminder for the user. Use this skill when the user asks to be reminded of something later."
    requires_approval = False

    async def execute(self, session_id: str, content: str, due_date: str = "TBD") -> str:
        """Saves a reminder to the local database."""
        with sqlite3.connect(AGENT_DB) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO reminders (session_id, content, due_date) VALUES (?, ?, ?)",
                (session_id, content, due_date)
            )
            conn.commit()
        return f"Reminder set: '{content}' for {due_date}"
