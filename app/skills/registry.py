import json
import logging
import sqlite3
import importlib
import os
import pkgutil
from typing import Dict, Any, List, Optional, Type
from app.bootstrap import SKILL_AUDIT_LOG, AGENT_DB
from app.skills.base import BaseSkill

# Setup skill audit logging
LOG_FILE = SKILL_AUDIT_LOG
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [SKILL_AUDIT] %(message)s"
)

class SkillRegistry:
    """
    Manages discovery, registration, and execution of modular skills.
    Replaces the old ToolManager.
    """
    def __init__(self, db_path: str = AGENT_DB):
        self.skills: Dict[str, BaseSkill] = {}
        self.db_path = db_path
        self._init_audit_table()

    def _init_audit_table(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Migration: Ensure existing table is renamed if necessary, then create new one
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tool_invocations'")
            if cursor.fetchone():
                cursor.execute("ALTER TABLE tool_invocations RENAME TO skill_invocations")
                cursor.execute("ALTER TABLE skill_invocations RENAME COLUMN tool_name TO skill_name")
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS skill_invocations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    skill_name TEXT,
                    parameters TEXT,
                    result TEXT,
                    status TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def register_skill(self, skill: BaseSkill):
        self.skills[skill.name] = skill
        print(f"[SKILL] Registered: {skill.name}")

    def discover_local_skills(self):
        """
        Dynamically imports all modules in app.skills.local and registers Skill classes.
        """
        try:
            import app.skills.local as local_package
            path = os.path.dirname(local_package.__file__)
            
            for loader, module_name, is_pkg in pkgutil.iter_modules([path]):
                full_module_name = f"app.skills.local.{module_name}"
                module = importlib.import_module(full_module_name)
                
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and 
                        issubclass(attr, BaseSkill) and 
                        attr is not BaseSkill and
                        attr_name != "MCPSkill"):
                        # Instantiate and register
                        skill_instance = attr()
                        self.register_skill(skill_instance)
        except Exception as e:
            print(f"[SKILL] Local discovery error: {e}")

    def discover_mcp_skills(self, config_path: str):
        """
        Loads tools from configured MCP servers.
        """
        from app.skills.mcp.client import MCPManager
        manager = MCPManager(config_path)
        mcp_skills = manager.load_mcp_skills()
        for skill in mcp_skills:
            self.register_skill(skill)

    def get_skill_schemas(self) -> str:
        if not self.skills:
            return "No skills available."
        
        schemas = []
        for name, skill in self.skills.items():
            schemas.append(json.dumps(skill.get_schema(), indent=2))
        return "\n---\n".join(schemas)

    async def invoke(self, name: str, params: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        if name not in self.skills:
            return {"status": "error", "error": f"Skill '{name}' not found."}

        skill = self.skills[name]
        logging.info(f"Session: {session_id} | Invoking: {name} | Params: {params}")
        
        try:
            # Execute the skill
            result = await skill.execute(session_id, **params)
            status = "success"
            
            # Log to DB
            self._log_to_db(session_id, name, params, result, status)
            return {"status": status, "result": result}
            
        except Exception as e:
            status = "error"
            error_msg = str(e)
            self._log_to_db(session_id, name, params, error_msg, status)
            logging.error(f"Skill Error ({name}): {error_msg}")
            return {"status": status, "error": error_msg}

    def _log_to_db(self, session_id: str, name: str, params: Dict, result: Any, status: str):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO skill_invocations (session_id, skill_name, parameters, result, status) VALUES (?, ?, ?, ?, ?)",
                (session_id, name, json.dumps(params), str(result), status)
            )
            conn.commit()

# Global instance
skill_registry = SkillRegistry()
