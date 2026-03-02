import pytest
from pydantic import ValidationError
from src.skills import ScientificSkill

def test_valid_skill_creation():
    """Tests that a valid skill passes validation."""
    skill_data = {
        "name": "Conservation Law Detection",
        "domain": "Physics",
        "description": "Detects if the sum of squares of two variables is constant.",
        "preconditions": ["Data contains at least two numerical variables"],
        "procedural_steps": [
            "Calculate x^2 + y^2 for all time steps",
            "Check if the variance of the sum is below a threshold"
        ],
        "expected_effects": ["Identify potential circular motion or harmonic oscillator"]
    }
    skill = ScientificSkill.from_dict(skill_data)
    assert skill.name == "Conservation Law Detection"
    assert len(skill.procedural_steps) == 2

def test_serialization_deserialization():
    """Tests JSON serialization and deserialization."""
    skill = ScientificSkill(
        name="Unit Analysis",
        domain="General Science",
        description="Checks for dimensional consistency.",
        procedural_steps=["Identify units of each variable", "Verify equation balance"]
    )
    json_str = skill.to_json()
    new_skill = ScientificSkill.from_json(json_str)
    assert new_skill.name == skill.name
    assert new_skill.procedural_steps == skill.procedural_steps

def test_missing_required_fields():
    """Tests that missing required fields raise ValidationError."""
    invalid_data = {
        "name": "Incomplete Skill",
        # Missing domain and procedural_steps
    }
    with pytest.raises(ValidationError):
        ScientificSkill.from_dict(invalid_data)

def test_invalid_field_types():
    """Tests that invalid field types raise ValidationError."""
    invalid_data = {
        "name": "Wrong Type Skill",
        "domain": "Physics",
        "description": "Description",
        "procedural_steps": "This should be a list, not a string"
    }
    with pytest.raises(ValidationError):
        ScientificSkill.from_dict(invalid_data)
