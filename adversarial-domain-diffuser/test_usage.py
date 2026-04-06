#!/usr/bin/env python3
"""Comprehensive programmatic usage test for ADD."""
import sys
import traceback

results = []

def test(name, fn):
    try:
        fn()
        results.append((name, "PASS", ""))
        print(f"  ✓ {name}")
    except Exception as e:
        results.append((name, "FAIL", f"{type(e).__name__}: {e}"))
        print(f"  ✗ {name}: {type(e).__name__}: {e}")
        traceback.print_exc()

# ========== 1. AdversarialGenerator ==========
print("\n=== AdversarialGenerator ===")

def test_adv_gen_basic():
    from src.adversarial_gen import AdversarialGenerator
    gen = AdversarialGenerator()
    q = gen.generate_question("test prompt", "target response", "expert ref")
    assert isinstance(q, str)
    assert len(q) > 0
test("AdversarialGenerator basic", test_adv_gen_basic)

def test_adv_gen_empty_strings():
    from src.adversarial_gen import AdversarialGenerator
    gen = AdversarialGenerator()
    q = gen.generate_question("", "", "")
    assert isinstance(q, str)
test("AdversarialGenerator empty strings", test_adv_gen_empty_strings)

def test_adv_gen_none_client():
    from src.adversarial_gen import AdversarialGenerator
    gen = AdversarialGenerator(model_client=None)
    q = gen.generate_question("p", "t", "e")
    assert isinstance(q, str)
test("AdversarialGenerator None client", test_adv_gen_none_client)

# ========== 2. DomainReasoningAgent ==========
print("\n=== DomainReasoningAgent ===")

def test_agent_basic():
    from src.reasoning_agent import DomainReasoningAgent
    agent = DomainReasoningAgent(domain="legal")
    resp = agent.process("What are the elements of negligence?")
    assert isinstance(resp, str)
test("DomainReasoningAgent.process() (README example)", test_agent_basic)

def test_agent_generate():
    from src.reasoning_agent import DomainReasoningAgent
    agent = DomainReasoningAgent(domain="legal")
    resp = agent.generate_response("What are the elements of negligence?")
    assert isinstance(resp, str)
    assert len(resp) > 0
test("DomainReasoningAgent.generate_response()", test_agent_generate)

def test_agent_empty_domain():
    from src.reasoning_agent import DomainReasoningAgent
    agent = DomainReasoningAgent(domain="")
    resp = agent.generate_response("test")
    assert isinstance(resp, str)
test("DomainReasoningAgent empty domain", test_agent_empty_domain)

def test_agent_empty_query():
    from src.reasoning_agent import DomainReasoningAgent
    agent = DomainReasoningAgent(domain="legal")
    resp = agent.generate_response("")
    assert isinstance(resp, str)
test("DomainReasoningAgent empty query", test_agent_empty_query)

# ========== 3. DiffusionPolicy ==========
print("\n=== DiffusionPolicy ===")

def test_diffusion_basic():
    from src.diffusion import DiffusionPolicy
    policy = DiffusionPolicy()
    action = policy.sample_action(context="legal_reasoning")
    assert action is not None
    assert action.shape == (1, 1)
test("DiffusionPolicy basic (README example)", test_diffusion_basic)

def test_diffusion_custom_dims():
    from src.diffusion import DiffusionPolicy
    policy = DiffusionPolicy(action_dim=5, horizon=10, num_diffusion_steps=20)
    action = policy.sample_action(context="test")
    assert action.shape == (10, 5)
test("DiffusionPolicy custom dims", test_diffusion_custom_dims)

def test_diffusion_zero_steps():
    from src.diffusion import DiffusionPolicy
    policy = DiffusionPolicy(num_diffusion_steps=0)
    action = policy.sample_action(context="test")
    assert action is not None
test("DiffusionPolicy 0 steps", test_diffusion_zero_steps)

def test_diffusion_none_context():
    from src.diffusion import DiffusionPolicy
    policy = DiffusionPolicy()
    action = policy.sample_action(conditioning_context=None)
    assert action is not None
test("DiffusionPolicy None context", test_diffusion_none_context)

# ========== 4. OMADOrchestrator ==========
print("\n=== OMADOrchestrator ===")

def test_omad_basic():
    from src.omad import OMADOrchestrator
    from src.diffusion import DiffusionPolicy
    p1 = DiffusionPolicy()
    p2 = DiffusionPolicy()
    orch = OMADOrchestrator(
        agents={"agent_1": p1, "agent_2": p2},
        agent_metadata=[{"id": "agent_1", "domain": "legal"}, {"id": "agent_2", "domain": "medical"}]
    )
    result = orch.coordinate(query="Analyze this case...")
    print(f"    coordinate() returned: {type(result)}")
test("OMADOrchestrator.coordinate() (README signature)", test_omad_basic)

def test_omad_step():
    from src.omad import OMADOrchestrator
    from src.diffusion import DiffusionPolicy
    p1 = DiffusionPolicy()
    p2 = DiffusionPolicy()
    orch = OMADOrchestrator(
        agents={"agent_1": p1, "agent_2": p2},
        agent_metadata=[{"id": "agent_1", "domain": "legal"}, {"id": "agent_2", "domain": "medical"}]
    )
    result = orch.step("test context")
    assert "individual_trajectories" in result
    assert "consensus_path" in result
test("OMADOrchestrator.step()", test_omad_step)

def test_omad_empty_agents():
    from src.omad import OMADOrchestrator
    orch = OMADOrchestrator(agents={})
    result = orch.step("test")
    print(f"    Empty agents step result: {result}")
test("OMADOrchestrator empty agents", test_omad_empty_agents)

def test_omad_no_metadata():
    from src.omad import OMADOrchestrator
    from src.diffusion import DiffusionPolicy
    p1 = DiffusionPolicy()
    orch = OMADOrchestrator(agents={"a1": p1})
    result = orch.step("test")
    assert result is not None
test("OMADOrchestrator no metadata", test_omad_no_metadata)

# ========== 5. EmbodimentGrouper ==========
print("\n=== EmbodimentGrouper ===")

def test_grouper_basic():
    from src.grouping import EmbodimentGrouper
    grouper = EmbodimentGrouper([
        {"id": "legal_1", "morphology": {"expertise": "law"}},
        {"id": "medical_1", "morphology": {"expertise": "medicine"}},
        {"id": "legal_2", "morphology": {"expertise": "law"}},
    ])
    groups = grouper.group_agents([
        {"id": "legal_1", "morphology": {"expertise": "law"}},
        {"id": "medical_1", "morphology": {"expertise": "medicine"}},
        {"id": "legal_2", "morphology": {"expertise": "law"}},
    ])
    print(f"    groups: {groups}")
test("EmbodimentGrouper.group_agents() (README signature)", test_grouper_basic)

def test_grouper_get_groups():
    from src.grouping import EmbodimentGrouper
    grouper = EmbodimentGrouper([
        {"id": "legal_1", "morphology": {"expertise": "law"}},
        {"id": "medical_1", "morphology": {"expertise": "medicine"}},
        {"id": "legal_2", "morphology": {"expertise": "law"}},
    ])
    groups = grouper.get_groups()
    assert "law" in groups
    assert "medicine" in groups
    assert len(groups["law"]) == 2
test("EmbodimentGrouper.get_groups()", test_grouper_get_groups)

def test_grouper_empty():
    from src.grouping import EmbodimentGrouper
    grouper = EmbodimentGrouper([])
    groups = grouper.get_groups()
    assert isinstance(groups, dict)
test("EmbodimentGrouper empty list", test_grouper_empty)

def test_grouper_none_morphology():
    from src.grouping import EmbodimentGrouper
    grouper = EmbodimentGrouper([
        {"id": "a1"},
        {"id": "a2", "morphology": None},
    ])
    groups = grouper.get_groups()
    print(f"    groups with None morphology: {groups}")
test("EmbodimentGrouper None morphology", test_grouper_none_morphology)

def test_grouper_missing_id():
    from src.grouping import EmbodimentGrouper
    grouper = EmbodimentGrouper([
        {"morphology": "science"},
    ])
    groups = grouper.get_groups()
    print(f"    groups with missing id: {groups}")
test("EmbodimentGrouper missing 'id' key", test_grouper_missing_id)

# ========== 6. AgentEnvironment ==========
print("\n=== AgentEnvironment ===")

def test_env_basic():
    from src.environment import AgentEnvironment
    from src.reasoning_agent import DomainReasoningAgent
    from src.omad import OMADOrchestrator
    from src.diffusion import DiffusionPolicy
    
    p1 = DiffusionPolicy()
    orch = OMADOrchestrator(agents={"a1": p1})
    env = AgentEnvironment(orchestrator=orch)
    
    legal_agent = DomainReasoningAgent(domain="legal")
    medical_agent = DomainReasoningAgent(domain="medical")
    env.register_agent(legal_agent)
    env.register_agent(medical_agent)
    
    result = env.process_query("What are the legal implications of this diagnosis?")
    assert "query" in result
    assert "responses" in result
test("AgentEnvironment (README example)", test_env_basic)

def test_env_no_agents():
    from src.environment import AgentEnvironment
    env = AgentEnvironment()
    result = env.process_query("test query")
    assert result["responses"] == {}
test("AgentEnvironment no agents", test_env_no_agents)

def test_env_empty_query():
    from src.environment import AgentEnvironment
    from src.reasoning_agent import DomainReasoningAgent
    env = AgentEnvironment()
    env.register_agent(DomainReasoningAgent(domain="legal"))
    result = env.process_query("")
    assert result is not None
test("AgentEnvironment empty query", test_env_empty_query)

def test_env_many_iterations():
    from src.environment import AgentEnvironment
    from src.reasoning_agent import DomainReasoningAgent
    env = AgentEnvironment()
    env.register_agent(DomainReasoningAgent(domain="legal"))
    result = env.process_query("test", iterations=50)
    assert len(result["blackboard_history"]) == 50
test("AgentEnvironment 50 iterations", test_env_many_iterations)

def test_env_zero_iterations():
    from src.environment import AgentEnvironment
    from src.reasoning_agent import DomainReasoningAgent
    env = AgentEnvironment()
    env.register_agent(DomainReasoningAgent(domain="legal"))
    result = env.process_query("test", iterations=0)
    assert result["blackboard_history"] == []
test("AgentEnvironment 0 iterations", test_env_zero_iterations)

# ========== 7. IntegratedAdversarialLoop ==========
print("\n=== IntegratedAdversarialLoop ===")

def test_loop_basic():
    from src.integrated_loop import IntegratedAdversarialLoop
    loop = IntegratedAdversarialLoop(
        agent_configs=[
            {"id": "agent_legal", "domain": "legal", "morphology": {"expertise": "law"}},
            {"id": "agent_medical", "domain": "medical", "morphology": {"expertise": "medicine"}},
        ],
        expert_reference="Expert-level explanation...",
        max_iterations=3
    )
    result = loop.run_iteration("Explain informed consent in medical treatment")
    assert "final_gap_score" in result
    assert "history" in result
    assert len(result["history"]) == 3
test("IntegratedAdversarialLoop basic (README)", test_loop_basic)

def test_loop_1_iteration():
    from src.integrated_loop import IntegratedAdversarialLoop
    loop = IntegratedAdversarialLoop(
        agent_configs=[{"id": "a1", "domain": "legal"}],
        expert_reference="expert",
        max_iterations=1
    )
    result = loop.run_iteration("test")
    assert len(result["history"]) == 1
test("Loop 1 iteration", test_loop_1_iteration)

def test_loop_0_iterations():
    from src.integrated_loop import IntegratedAdversarialLoop
    loop = IntegratedAdversarialLoop(
        agent_configs=[{"id": "a1", "domain": "legal"}],
        expert_reference="expert",
        max_iterations=0
    )
    result = loop.run_iteration("test")
    print(f"    0 iteration result: {result}")
test("Loop 0 iterations", test_loop_0_iterations)

def test_loop_empty_configs():
    from src.integrated_loop import IntegratedAdversarialLoop
    loop = IntegratedAdversarialLoop(
        agent_configs=[],
        expert_reference="expert",
        max_iterations=2
    )
    result = loop.run_iteration("test")
    print(f"    Empty configs result keys: {result.keys()}, gap: {result.get('final_gap_score')}")
test("Loop empty agent configs", test_loop_empty_configs)

def test_loop_missing_domain():
    from src.integrated_loop import IntegratedAdversarialLoop
    loop = IntegratedAdversarialLoop(
        agent_configs=[{"id": "a1"}],  # no domain key
        expert_reference="expert",
        max_iterations=1
    )
    result = loop.run_iteration("test")
    assert result is not None
test("Loop missing domain in config", test_loop_missing_domain)

def test_loop_missing_id():
    from src.integrated_loop import IntegratedAdversarialLoop
    try:
        loop = IntegratedAdversarialLoop(
            agent_configs=[{"domain": "legal"}],  # no id key
            expert_reference="expert",
            max_iterations=1
        )
        result = loop.run_iteration("test")
        print(f"    No error, result: {result.get('final_gap_score')}")
    except KeyError as e:
        raise  # re-raise for test framework
test("Loop missing id in config", test_loop_missing_id)

def test_loop_large_input():
    from src.integrated_loop import IntegratedAdversarialLoop
    loop = IntegratedAdversarialLoop(
        agent_configs=[{"id": "a1", "domain": "legal"}],
        expert_reference="x" * 100000,
        max_iterations=1
    )
    result = loop.run_iteration("y" * 100000)
    assert result is not None
test("Loop very large input strings", test_loop_large_input)

def test_loop_none_expert():
    from src.integrated_loop import IntegratedAdversarialLoop
    try:
        loop = IntegratedAdversarialLoop(
            agent_configs=[{"id": "a1", "domain": "legal"}],
            expert_reference=None,
            max_iterations=1
        )
        result = loop.run_iteration("test")
        print(f"    None expert result: {result.get('final_gap_score')}")
    except Exception as e:
        raise
test("Loop None expert_reference", test_loop_none_expert)

def test_loop_none_query():
    from src.integrated_loop import IntegratedAdversarialLoop
    loop = IntegratedAdversarialLoop(
        agent_configs=[{"id": "a1", "domain": "legal"}],
        expert_reference="expert",
        max_iterations=1
    )
    try:
        result = loop.run_iteration(None)
        print(f"    None query result: {result.get('final_gap_score')}")
    except Exception as e:
        raise
test("Loop None initial_context", test_loop_none_query)

# ========== 8. Evaluator ==========
print("\n=== Evaluator ===")

def test_evaluator_basic():
    from src.evaluation import Evaluator
    evaluator = Evaluator()
    benchmarks = evaluator.load_benchmarks()
    assert "LegalBench" in benchmarks
    assert "MedicalQA" in benchmarks
test("Evaluator.load_benchmarks()", test_evaluator_basic)

def test_evaluator_run():
    from src.evaluation import Evaluator
    evaluator = Evaluator(results_dir="/tmp/add_test_results")
    benchmarks = evaluator.load_benchmarks()
    report = evaluator.run_evaluation("LegalBench", benchmarks["LegalBench"], max_iterations=2, num_runs=1)
    assert report is not None
    assert "results" in report
test("Evaluator.run_evaluation()", test_evaluator_run)

def test_evaluator_empty_items():
    from src.evaluation import Evaluator
    evaluator = Evaluator(results_dir="/tmp/add_test_results2")
    report = evaluator.run_evaluation("EmptyDomain", [], max_iterations=1, num_runs=1)
    assert report["results"] == []
test("Evaluator empty benchmark items", test_evaluator_empty_items)

# ========== 9. 100 agents test ==========
print("\n=== Scale Tests ===")

def test_100_agents():
    from src.integrated_loop import IntegratedAdversarialLoop
    configs = [{"id": f"agent_{i}", "domain": f"domain_{i}", "morphology": {"expertise": f"exp_{i}"}} for i in range(100)]
    loop = IntegratedAdversarialLoop(
        agent_configs=configs,
        expert_reference="expert text",
        max_iterations=1
    )
    result = loop.run_iteration("test query")
    assert len(result["history"]) == 1
test("100 agents, 1 iteration", test_100_agents)

# ========== SUMMARY ==========
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
passed = sum(1 for _, s, _ in results if s == "PASS")
failed = sum(1 for _, s, _ in results if s == "FAIL")
print(f"Passed: {passed}, Failed: {failed}, Total: {len(results)}")
print()
for name, status, msg in results:
    if status == "FAIL":
        print(f"  FAILED: {name}")
        print(f"    {msg}")
