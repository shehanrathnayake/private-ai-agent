from typing import Dict, Any, Optional
import inspect

class BaseSkill:
    """
    Base class for all skills. 
    Inherit from this to create a new modular skill.
    """
    name: str = "base_skill"
    description: str = "Base skill description"
    requires_approval: bool = False

    def get_schema(self) -> Dict[str, Any]:
        """
        Generates a JSON schema for the skill's execute method.
        Uses type hints and docstrings.
        """
        sig = inspect.signature(self.execute)
        parameters = {}
        
        for name, param in sig.parameters.items():
            if name in ["self", "session_id"]:
                continue
                
            param_info = {
                "type": "string" # Default to string
            }
            
            if param.annotation == int:
                param_info["type"] = "integer"
            elif param.annotation == bool:
                param_info["type"] = "boolean"
            elif param.annotation == float:
                param_info["type"] = "number"
            
            # Extract description from docstring if possible (advanced implementation)
            # For now, we'll keep it simple or allow manual overrides
            parameters[name] = param_info

        return {
            "name": self.name,
            "description": self.description,
            "parameters": parameters,
            "requires_approval": self.requires_approval
        }

    async def execute(self, session_id: str, **kwargs) -> Any:
        """
        Executes the skill logic. Must be implemented by subclasses.
        """
        raise NotImplementedError("Each skill must implement its own execute method.")
