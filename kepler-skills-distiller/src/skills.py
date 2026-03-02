from typing import List, Dict, Any
from pydantic import BaseModel, Field

class ScientificSkill(BaseModel):
    """
    Represents a scientific expert skill following the SkillsBench taxonomy.
    Used for scientific reasoning and equation discovery.
    """
    name: str = Field(..., description="The name of the expert skill.")
    domain: str = Field(..., description="The scientific domain (e.g., 'Physics', 'Fluid Dynamics').")
    description: str = Field(..., description="A clear description of the skill's purpose.")
    preconditions: List[str] = Field(
        default_factory=list, 
        description="Requirements or state that must be true to apply this skill."
    )
    procedural_steps: List[str] = Field(
        ..., 
        description="The step-by-step procedure to execute this skill."
    )
    expected_effects: List[str] = Field(
        default_factory=list, 
        description="The anticipated outcome or discovery resulting from this skill."
    )

    def to_dict(self) -> Dict[str, Any]:
        """Converts the skill to a dictionary."""
        return self.model_dump()

    def to_json(self) -> str:
        """Serializes the skill to a formatted JSON string."""
        return self.model_dump_json(indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScientificSkill":
        """Creates a ScientificSkill instance from a dictionary."""
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, json_str: str) -> "ScientificSkill":
        """Creates a ScientificSkill instance from a JSON string."""
        return cls.model_validate_json(json_str)
