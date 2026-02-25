from app.skills.base import BaseSkill

class MemoryUpdateSkill(BaseSkill):
    name = "update_memory"
    description = "Explicitly record a new fact or preference to avoid forgetting. This skill updates long-term associative memory."
    requires_approval = False

    async def execute(self, session_id: str, fact_type: str, content: str) -> str:
        """Allows the agent to explicitly push a new fact into associative memory."""
        from app.memory import memory_manager
        # Salience is high for explicit memory updates
        memory_manager.add_vector(content, session_id, fact_type, salience=1.0)
        return f"Memory hardened: Recorded {fact_type} -> '{content}'"
