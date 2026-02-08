import json
import logging
from typing import Dict, Any, List, Callable, Optional
from datetime import datetime
import sqlite3
from app.config import VECTOR_DB_PATH

# Setup tool audit logging
LOG_FILE = "memory/tool_audit.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [TOOL_AUDIT] %(message)s"
)

class ToolManager:
    """
    Manages registration, schema discovery, and safe execution of agent tools.
    """
    def __init__(self, db_path: str = "memory/agent_memory.db"):
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.db_path = db_path
        self._init_audit_table()

    def _init_audit_table(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tool_invocations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    tool_name TEXT,
                    parameters TEXT,
                    result TEXT,
                    status TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS reminders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    content TEXT,
                    due_date TEXT,
                    status TEXT DEFAULT 'pending',
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    title TEXT,
                    priority TEXT,
                    status TEXT DEFAULT 'todo',
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def register_tool(self, name: str, description: str, parameters: Dict[str, Any], func: Callable, requires_approval: bool = False):
        """
        Registers a tool with a schema.
        """
        self.tools[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "func": func,
            "requires_approval": requires_approval
        }
        print(f"[PHASE4] Tool Registered: {name} (Approval Required: {requires_approval})")

    def get_tool_schemas(self) -> str:
        """
        Returns a formatted string of available tools for the agent's system prompt.
        """
        if not self.tools:
            return "No tools available."
        
        schemas = []
        for name, info in self.tools.items():
            schema = {
                "name": name,
                "description": info["description"],
                "parameters": info["parameters"],
                "requires_approval": info["requires_approval"]
            }
            schemas.append(json.dumps(schema, indent=2))
        return "\n---\n".join(schemas)

    def invoke(self, name: str, params: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """
        Executes a tool with the given parameters and logs the interaction.
        """
        if name not in self.tools:
            return {"error": f"Tool '{name}' not found."}

        tool = self.tools[name]
        
        # Safety Check: If it requires approval, we return a pending status (unless pre-approved logic is added)
        # For this phase, we assume the Agent logic calls this after interpretation.
        
        logging.info(f"Session: {session_id} | Invoking: {name} | Params: {params}")
        
        try:
            # Execute the actual function
            result = tool["func"](session_id, **params)
            status = "success"
            
            # Log to DB
            self._log_to_db(session_id, name, params, result, status)
            return {"status": status, "result": result}
            
        except Exception as e:
            status = "error"
            error_msg = str(e)
            self._log_to_db(session_id, name, params, error_msg, status)
            logging.error(f"Tool Error ({name}): {error_msg}")
            return {"status": status, "error": error_msg}

    def _log_to_db(self, session_id: str, name: str, params: Dict, result: Any, status: str):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO tool_invocations (session_id, tool_name, parameters, result, status) VALUES (?, ?, ?, ?, ?)",
                (session_id, name, json.dumps(params), str(result), status)
            )
            conn.commit()

# --- Example Core Tools Implementation ---

def set_reminder(session_id: str, content: str, due_date: str = "TBD"):
    """Saves a reminder to the local database."""
    with sqlite3.connect("memory/agent_memory.db") as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO reminders (session_id, content, due_date) VALUES (?, ?, ?)",
            (session_id, content, due_date)
        )
        conn.commit()
    return f"Reminder set: '{content}' for {due_date}"

def create_task(session_id: str, title: str, priority: str = "medium"):
    """Creates a task in the tracking system."""
    with sqlite3.connect("memory/agent_memory.db") as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO tasks (session_id, title, priority) VALUES (?, ?, ?)",
            (session_id, title, priority)
        )
        conn.commit()
    return f"Task created: [{priority.upper()}] {title}"

def update_memory_explicitly(session_id: str, fact_type: str, content: str):
    """Allows the agent to explicitly push a new fact into associative memory."""
    from app.memory import memory_manager
    # Salience is high for explicit memory updates
    memory_manager.add_vector(content, session_id, fact_type, salience=1.0)
    return f"Memory hardened: Recorded {fact_type} -> '{content}'"

# Global instance
tool_manager = ToolManager()

# Register Tools
tool_manager.register_tool(
    name="set_reminder",
    description="Set a reminder for the user. Use this when the user asks to be reminded of something later.",
    parameters={
        "content": "The text of the reminder.",
        "due_date": "Optional date/time string (default: TBD)"
    },
    func=set_reminder,
    requires_approval=False # Low-risk
)

tool_manager.register_tool(
    name="create_task",
    description="Add a task to the user's todo list.",
    parameters={
        "title": "Short title of the task.",
        "priority": "low, medium, or high (default: medium)"
    },
    func=create_task,
    requires_approval=True # Persistent modification
)

tool_manager.register_tool(
    name="update_memory",
    description="Explicitly record a new fact or preference to avoid forgetting.",
    parameters={
        "fact_type": "Known Fact, Preference, or Open Thread",
        "content": "The specific detail to remember."
    },
    func=update_memory_explicitly,
    requires_approval=False # Core internal function
)
