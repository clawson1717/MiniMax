import os
import shutil
import pytest
from src.skill_generator import ExpertSkillGenerator
from src.skills import ScientificSkill

@pytest.fixture
def temp_skills_dir(tmp_path):
    d = tmp_path / "skills"
    d.mkdir()
    return d

def test_generate_skill(temp_skills_dir):
    generator = ExpertSkillGenerator(skills_dir=str(temp_skills_dir))
    
    trajectory = {
        "analysis": "Conservation of energy observed.",
        "hypotheses": ["E = mc^2", "F = ma"],
        "proposed_expressions": ["m * c**2", "m * a"]
    }
    metrics = {"mse": 0.001, "r2": 0.99}
    
    skill = generator.generate_skill(
        trajectory=trajectory,
        metrics=metrics,
        skill_name="Relativity Discovery"
    )
    
    assert isinstance(skill, ScientificSkill)
    assert skill.name == "Relativity Discovery"
    assert "Conservation of energy" in skill.procedural_steps[0]
    assert "E = mc^2" in skill.procedural_steps[1]
    
    # Check if file exists
    skill_file = temp_skills_dir / "relativity_discovery.json"
    assert skill_file.exists()
    
    # Verify content
    with open(skill_file, "r") as f:
        data = f.read()
        loaded_skill = ScientificSkill.from_json(data)
        assert loaded_skill.name == skill.name

def test_save_skill(temp_skills_dir):
    generator = ExpertSkillGenerator(skills_dir=str(temp_skills_dir))
    skill = ScientificSkill(
        name="Test Skill",
        domain="Chemistry",
        description="Testing save",
        procedural_steps=["Step 1", "Step 2"]
    )
    
    path = generator.save_skill(skill)
    assert path.name == "test_skill.json"
    assert path.exists()
