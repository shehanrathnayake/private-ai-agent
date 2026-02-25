import sqlite3
from app.skills.base import BaseSkill
from app.bootstrap import AGENT_DB

class TaskSkill(BaseSkill):
    name = "create_task"
    description = "Add a task to the user's todo list. This skill is useful for project-level guidance and tracking."
    requires_approval = True # Persistent modification

    async def execute(self, session_id: str, title: str, priority: str = "medium") -> str:
        """Creates a task in the tracking system."""
        with sqlite3.connect(AGENT_DB) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO tasks (session_id, title, priority) VALUES (?, ?, ?)",
                (session_id, title, priority)
            )
            conn.commit()
        return f"Task created: [{priority.upper()}] {title}"
