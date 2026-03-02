import pytest
from src.grouping import EmbodimentGrouper

def test_embodiment_grouper_basic():
    agents = [
        {"id": "agent1", "morphology": "math"},
        {"id": "agent2", "morphology": "math"},
        {"id": "agent3", "morphology": "coding"},
        {"id": "agent4", "morphology": "coding"},
        {"id": "agent5", "morphology": "reasoning"},
    ]
    
    grouper = EmbodimentGrouper(agents)
    groups = grouper.get_groups()
    
    assert "math" in groups
    assert "coding" in groups
    assert "reasoning" in groups
    assert len(groups["math"]) == 2
    assert "agent1" in groups["math"]
    assert "agent2" in groups["math"]
    assert len(groups["coding"]) == 2
    assert len(groups["reasoning"]) == 1

def test_embodiment_grouper_dict_morphology():
    agents = [
        {"id": "a1", "morphology": {"expertise": "physics", "level": "senior"}},
        {"id": "a2", "morphology": {"expertise": "physics", "level": "junior"}},
        {"id": "a3", "morphology": {"style": "creative"}},
    ]
    
    grouper = EmbodimentGrouper(agents)
    groups = grouper.get_groups()
    
    assert "physics" in groups
    assert "creative" in groups
    assert "a1" in groups["physics"]
    assert "a2" in groups["physics"]
    assert "a3" in groups["creative"]

def test_embodiment_grouper_empty():
    grouper = EmbodimentGrouper([])
    groups = grouper.get_groups()
    assert groups == {}
