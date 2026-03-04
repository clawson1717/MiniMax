import time
from src.node import InteractionNode

def test_interaction_node_initialization():
    node = InteractionNode(
        id="node-1",
        agent_id="agent-alpha",
        input_payload={"query": "hello"},
        output_payload={"response": "world"},
        metadata={"score": 0.95}
    )
    assert node.id == "node-1"
    assert node.agent_id == "agent-alpha"
    assert node.input_payload == {"query": "hello"}
    assert node.output_payload == {"response": "world"}
    assert node.causal_parents == []
    assert isinstance(node.timestamp, float)
    assert node.metadata == {"score": 0.95}
    assert node.is_root() is True

def test_interaction_node_with_parents():
    node = InteractionNode(
        id="node-2",
        agent_id="agent-beta",
        input_payload="some input",
        output_payload="some output",
        causal_parents=["node-1"]
    )
    assert node.causal_parents == ["node-1"]
    assert node.is_root() is False

def test_payload_hash_consistency():
    node1 = InteractionNode(
        id="node-1",
        agent_id="agent-a",
        input_payload={"a": 1, "b": 2},
        output_payload={"result": 3}
    )
    
    # Same payloads, different metadata/id/timestamp should produce same hash
    node2 = InteractionNode(
        id="node-2",
        agent_id="agent-b",
        input_payload={"b": 2, "a": 1}, # Swapped order but dicts are same
        output_payload={"result": 3},
        timestamp=time.time() + 100
    )
    
    hash1 = node1.payload_hash()
    hash2 = node2.payload_hash()
    
    assert hash1 == hash2
    assert len(hash1) == 64 # SHA-256 length in hex

def test_payload_hash_difference():
    node1 = InteractionNode(
        id="node-1",
        agent_id="agent-a",
        input_payload="input 1",
        output_payload="output"
    )
    node2 = InteractionNode(
        id="node-2",
        agent_id="agent-a",
        input_payload="input 2",
        output_payload="output"
    )
    
    assert node1.payload_hash() != node2.payload_hash()
