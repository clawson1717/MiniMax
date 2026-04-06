#!/usr/bin/env python3
"""Corrected comprehensive tests after learning actual APIs."""
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

# ========== README vs Reality API mismatches ==========
print("\n=== README API Mismatch Tests ===")

def test_readme_agent_process():
    """README says: agent.process("query") but actual method is generate_response()"""
    from src.reasoning_agent import DomainReasoningAgent
    agent = DomainReasoningAgent(domain="legal")
    assert not hasattr(agent, 'process'), "process() shouldn't exist (README says it does)"
test("README claims .process() exists - MISMATCH", test_readme_agent_process)

def test_readme_diffusion_context():
    """README says: policy.sample_action(context="legal_reasoning") but actual param is conditioning_context"""
    from src.diffusion import DiffusionPolicy
    policy = DiffusionPolicy()
    try:
        policy.sample_action(context="legal_reasoning")
        results[-1] = (results[-1][0], "FAIL", "Should have failed - README param name is wrong")
    except TypeError:
        pass  # Expected
test("README claims context= param - MISMATCH", test_readme_diffusion_context)

def test_readme_omad_coordinate():
    """README says: orchestrator.coordinate(query="...") but actual param is agent_trajectories"""
    from src.omad import OMADOrchestrator
    from src.diffusion import DiffusionPolicy
    orch = OMADOrchestrator(agents={"a1": DiffusionPolicy()})
    try:
        orch.coordinate(query="test")
        results[-1] = (results[-1][0], "FAIL", "Should have failed - README param name is wrong")
    except TypeError:
        pass  # Expected
test("README claims .coordinate(query=) - MISMATCH", test_readme_omad_coordinate)

def test_readme_grouper_group_agents():
    """README says: grouper.group_agents([...]) but actual method is get_groups()"""
    from src.grouping import EmbodimentGrouper
    grouper = EmbodimentGrouper([])
    assert not hasattr(grouper, 'group_agents'), "group_agents() shouldn't exist (README says it does)"
test("README claims .group_agents() exists - MISMATCH", test_readme_grouper_group_agents)

def test_readme_grouper_int_keys():
    """README says groups = {0: ["legal_1", "legal_2"], 1: ["medical_1"]} (int keys)
    but actual uses string keys like 'law', 'medicine'"""
    from src.grouping import EmbodimentGrouper
    grouper = EmbodimentGrouper([
        {"id": "legal_1", "morphology": {"expertise": "law"}},
        {"id": "medical_1", "morphology": {"expertise": "medicine"}},
        {"id": "legal_2", "morphology": {"expertise": "law"}},
    ])
    groups = grouper.get_groups()
    assert all(isinstance(k, str) for k in groups.keys()), "Keys are strings, not ints as README shows"
test("README shows int group keys but actual uses strings", test_readme_grouper_int_keys)

# ========== Correct API usage tests ==========
print("\n=== Correct API Usage ===")

def test_agent_correct():
    from src.reasoning_agent import DomainReasoningAgent
    agent = DomainReasoningAgent(domain="legal")
    resp = agent.generate_response("What are the elements of negligence?")
    assert isinstance(resp, str) and len(resp) > 0
test("DomainReasoningAgent.generate_response()", test_agent_correct)

def test_diffusion_correct():
    from src.diffusion import DiffusionPolicy
    policy = DiffusionPolicy()
    action = policy.sample_action(conditioning_context="legal_reasoning")
    assert action is not None and action.shape == (1, 1)
test("DiffusionPolicy.sample_action(conditioning_context=)", test_diffusion_correct)

def test_omad_step_correct():
    from src.omad import OMADOrchestrator
    from src.diffusion import DiffusionPolicy
    orch = OMADOrchestrator(agents={"a1": DiffusionPolicy(), "a2": DiffusionPolicy()},
        agent_metadata=[{"id": "a1", "domain": "legal"}, {"id": "a2", "domain": "med"}])
    result = orch.step("test context")
    assert "consensus_path" in result
test("OMADOrchestrator.step()", test_omad_step_correct)

# ========== Edge Cases ==========
print("\n=== Edge Cases ===")

def test_loop_0_iterations():
    from src.integrated_loop import IntegratedAdversarialLoop
    loop = IntegratedAdversarialLoop(
        agent_configs=[{"id": "a1", "domain": "legal"}],
        expert_reference="expert", max_iterations=0
    )
    try:
        result = loop.run_iteration("test")
        # With 0 iterations, history is empty. result[-1] would fail:
        print(f"    Result: final_gap_score={result.get('final_gap_score')}")
        results[-1] = (results[-1][0], "FAIL", f"Should crash: history empty but code accesses [-1]. Got: {result}")
    except (IndexError, Exception) as e:
        print(f"    Expected crash: {e}")
test("Loop 0 iterations (crashes?)", test_loop_0_iterations)

def test_loop_empty_configs():
    from src.integrated_loop import IntegratedAdversarialLoop
    loop = IntegratedAdversarialLoop(
        agent_configs=[], expert_reference="expert", max_iterations=2
    )
    result = loop.run_iteration("test")
    print(f"    Empty configs gap: {result.get('final_gap_score')}")
test("Loop empty agent configs", test_loop_empty_configs)

def test_loop_missing_id():
    from src.integrated_loop import IntegratedAdversarialLoop
    try:
        loop = IntegratedAdversarialLoop(
            agent_configs=[{"domain": "legal"}],
            expert_reference="expert", max_iterations=1
        )
        result = loop.run_iteration("test")
        results[-1] = (results[-1][0], "FAIL", "Should have KeyError for missing 'id'")
    except KeyError as e:
        print(f"    Expected KeyError: {e}")
test("Loop missing 'id' in config -> KeyError", test_loop_missing_id)

def test_grouper_missing_id():
    from src.grouping import EmbodimentGrouper
    grouper = EmbodimentGrouper([{"morphology": "science"}])
    try:
        groups = grouper.get_groups()
        results[-1] = (results[-1][0], "FAIL", "Should KeyError on missing 'id'")
    except KeyError:
        pass  # Expected
test("EmbodimentGrouper missing 'id' -> KeyError", test_grouper_missing_id)

def test_grouper_none_morphology():
    from src.grouping import EmbodimentGrouper
    grouper = EmbodimentGrouper([{"id": "a1"}, {"id": "a2", "morphology": None}])
    try:
        groups = grouper.get_groups()
        print(f"    Groups: {groups}")
    except Exception as e:
        print(f"    Error: {e}")
test("EmbodimentGrouper None morphology", test_grouper_none_morphology)

def test_loop_none_expert():
    from src.integrated_loop import IntegratedAdversarialLoop
    loop = IntegratedAdversarialLoop(
        agent_configs=[{"id": "a1", "domain": "legal"}],
        expert_reference=None, max_iterations=1
    )
    try:
        result = loop.run_iteration("test")
        print(f"    None expert result: {result.get('final_gap_score')}")
    except Exception as e:
        print(f"    Error: {e}")
test("Loop None expert_reference", test_loop_none_expert)

def test_loop_none_query():
    from src.integrated_loop import IntegratedAdversarialLoop
    loop = IntegratedAdversarialLoop(
        agent_configs=[{"id": "a1", "domain": "legal"}],
        expert_reference="expert", max_iterations=1
    )
    try:
        result = loop.run_iteration(None)
        print(f"    None query result: {result.get('final_gap_score')}")
    except Exception as e:
        print(f"    Error: {e}")
test("Loop None initial_context", test_loop_none_query)

def test_diffusion_zero_steps():
    from src.diffusion import DiffusionPolicy
    policy = DiffusionPolicy(num_diffusion_steps=0)
    action = policy.sample_action("test")
    print(f"    0-step action: {action}")
test("DiffusionPolicy 0 diffusion steps", test_diffusion_zero_steps)

def test_env_no_agents():
    from src.environment import AgentEnvironment
    env = AgentEnvironment()
    result = env.process_query("test query")
    assert result["responses"] == {}
test("AgentEnvironment no agents", test_env_no_agents)

def test_env_zero_iterations():
    from src.environment import AgentEnvironment
    from src.reasoning_agent import DomainReasoningAgent
    env = AgentEnvironment()
    env.register_agent(DomainReasoningAgent(domain="legal"))
    result = env.process_query("test", iterations=0)
    assert result["blackboard_history"] == []
test("AgentEnvironment 0 iterations", test_env_zero_iterations)

# ========== CLI mismatch tests ==========
print("\n=== CLI Documentation Mismatches ===")

def test_cli_readme_eval_flag():
    """README says: python -m src.main --eval but actual CLI uses subcommands"""
    import subprocess
    r = subprocess.run(["python3", "-m", "src.main", "--eval"], capture_output=True, text=True)
    assert r.returncode != 0, "README --eval syntax should fail"
    print(f"    --eval fails as expected: {r.stderr.strip()[:80]}")
test("README --eval flag doesn't work (uses subcommands)", test_cli_readme_eval_flag)

def test_cli_readme_eval_domain():
    """README says: python -m src.main --eval --domain legal but domains are LegalBench/MedicalQA"""
    import subprocess
    r = subprocess.run(["python3", "-m", "src.main", "eval", "--domain", "legal"], capture_output=True, text=True)
    assert "not found" in r.stdout.lower() or "error" in r.stdout.lower()
    print(f"    --domain legal fails: {r.stdout.strip()[:80]}")
test("README --domain legal fails (key is LegalBench)", test_cli_readme_eval_domain)

# ========== Evaluator domain matching ==========
print("\n=== Evaluator domain matching ===")

def test_eval_domain_case():
    from src.evaluation import Evaluator
    evaluator = Evaluator(results_dir="/tmp/add_eval_test")
    benchmarks = evaluator.load_benchmarks()
    print(f"    Available domain keys: {list(benchmarks.keys())}")
    # README Quick Start says --domain legal but keys are LegalBench, MedicalQA
    assert "legal" not in benchmarks
    assert "medical" not in benchmarks
    assert "LegalBench" in benchmarks
    assert "MedicalQA" in benchmarks
test("Evaluator domain keys are LegalBench/MedicalQA not legal/medical", test_eval_domain_case)

# ========== Convergence behavior ==========
print("\n=== Convergence behavior ===")

def test_gap_score_consistency():
    """Check if gap score actually decreases across runs (convergence)"""
    from src.integrated_loop import IntegratedAdversarialLoop
    loop = IntegratedAdversarialLoop(
        agent_configs=[
            {"id": "a1", "domain": "legal", "morphology": {"expertise": "law"}},
            {"id": "a2", "domain": "medical", "morphology": {"expertise": "medicine"}},
        ],
        expert_reference="immunogenicity phenotypic off-target insertions genomic modification",
        max_iterations=5
    )
    result = loop.run_iteration("Discuss gene therapy regulatory framework")
    scores = [h["gap_score"] for h in result["history"]]
    print(f"    Gap scores: {scores}")
    # Check if generally decreasing (last < first)
    if scores[-1] >= scores[0]:
        print(f"    WARNING: Gap score didn't decrease! {scores[0]} -> {scores[-1]}")
    # Check for non-monotonic behavior (score goes up sometimes)
    increases = sum(1 for i in range(1, len(scores)) if scores[i] > scores[i-1])
    if increases > 0:
        print(f"    Non-monotonic: {increases} iteration(s) where gap increased")
test("Gap score convergence behavior", test_gap_score_consistency)

# ========== Scale test (smaller) ==========
print("\n=== Scale Tests ===")

def test_20_agents():
    from src.integrated_loop import IntegratedAdversarialLoop
    configs = [{"id": f"agent_{i}", "domain": f"domain_{i}", "morphology": {"expertise": f"exp_{i}"}} for i in range(20)]
    loop = IntegratedAdversarialLoop(
        agent_configs=configs, expert_reference="expert text", max_iterations=1
    )
    result = loop.run_iteration("test query")
    assert len(result["history"]) == 1
    print(f"    20 agents gap: {result['final_gap_score']}")
test("20 agents, 1 iteration", test_20_agents)

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
