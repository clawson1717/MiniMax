"""Tests for checklist module - CM2-style evaluation."""

import pytest
from src.checklist import (
    Checklist, 
    ChecklistItem, 
    ChecklistCriterion,
    ChecklistEvaluator,
    CheckStatus,
    create_checklist,
    evaluate_trajectory,
)


class TestChecklistCriterion:
    """Tests for ChecklistCriterion."""
    
    def test_criterion_creation(self):
        """Test creating a criterion."""
        criterion = ChecklistCriterion(description="Test criterion", weight=0.5)
        assert criterion.description == "Test criterion"
        assert criterion.weight == 0.5
        assert criterion.is_satisfied is False
    
    def test_criterion_invalid_weight(self):
        """Test that invalid weights raise errors."""
        with pytest.raises(ValueError):
            ChecklistCriterion(description="Test", weight=1.5)
        with pytest.raises(ValueError):
            ChecklistCriterion(description="Test", weight=-0.5)
    
    def test_criterion_evaluate_with_function(self):
        """Test evaluating criterion with a check function."""
        criterion = ChecklistCriterion(
            description="Has result",
            weight=1.0,
            check_function=lambda d: d.get("result_count", 0) > 0
        )
        
        # Not satisfied initially
        assert criterion.is_satisfied is False
        
        # Evaluate with data that doesn't satisfy
        result = criterion.evaluate({"result_count": 0})
        assert result is False
        assert criterion.is_satisfied is False
        
        # Evaluate with data that satisfies
        result = criterion.evaluate({"result_count": 5})
        assert result is True
        assert criterion.is_satisfied is True
    
    def test_criterion_monotonic(self):
        """Test that criterion satisfaction is monotonic."""
        criterion = ChecklistCriterion(
            description="Test",
            weight=1.0,
            check_function=lambda d: d.get("value", 0) > 5
        )
        
        criterion.evaluate({"value": 10})
        assert criterion.is_satisfied is True
        
        # Even with unsatisfying data, stays satisfied
        criterion.evaluate({"value": 0})
        assert criterion.is_satisfied is True
    
    def test_criterion_reset(self):
        """Test resetting criterion."""
        criterion = ChecklistCriterion(
            description="Test",
            weight=1.0,
            check_function=lambda d: True
        )
        
        criterion.evaluate({})
        assert criterion.is_satisfied is True
        
        criterion.reset()
        assert criterion.is_satisfied is False
    
    def test_criterion_no_function(self):
        """Test criterion without check function."""
        criterion = ChecklistCriterion(description="Test", weight=1.0)
        
        # Can't satisfy without check function
        result = criterion.evaluate({"some_data": True})
        assert result is False
        assert criterion.is_satisfied is False


class TestChecklist:
    """Tests for Checklist with CM2 criteria."""
    
    def test_initialization(self):
        """Test checklist initialization."""
        checklist = Checklist(name="My Tasks", task_type="search")
        assert checklist.name == "My Tasks"
        assert checklist.task_type == "search"
        assert len(checklist.criteria) == 0
        assert len(checklist.items) == 0
    
    def test_add_criterion(self):
        """Test adding criteria."""
        checklist = Checklist()
        criterion = checklist.add_criterion("Test criterion", weight=0.5)
        
        assert criterion.description == "Test criterion"
        assert criterion.weight == 0.5
        assert len(checklist.criteria) == 1
    
    def test_evaluate_no_criteria(self):
        """Test evaluating with no criteria."""
        checklist = Checklist()
        result = checklist.evaluate({"step": "data"})
        
        assert result["score"] == 0.0
        assert result["progress"] == 0.0
        assert result["is_complete"] is True  # Empty = complete
    
    def test_evaluate_single_criterion(self):
        """Test evaluating a single criterion."""
        checklist = Checklist()
        checklist.add_criterion(
            description="Found result",
            weight=1.0,
            check_function=lambda d: d.get("found", False)
        )
        
        # Not satisfied
        result = checklist.evaluate({"found": False})
        assert result["score"] == 0.0
        assert result["is_complete"] is False
        assert len(result["newly_satisfied"]) == 0
        
        # Satisfied
        result = checklist.evaluate({"found": True})
        assert result["score"] == 1.0
        assert result["progress"] == 100.0
        assert result["is_complete"] is True
        assert "Found result" in result["newly_satisfied"]
    
    def test_evaluate_multiple_criteria(self):
        """Test evaluating multiple criteria with weights."""
        checklist = Checklist()
        checklist.add_criterion("Step 1", weight=0.3, check_function=lambda d: d.get("step1", False))
        checklist.add_criterion("Step 2", weight=0.7, check_function=lambda d: d.get("step2", False))
        
        # No criteria satisfied
        result = checklist.evaluate({"step1": False, "step2": False})
        assert result["score"] == 0.0
        
        # Only step1 satisfied (30%)
        result = checklist.evaluate({"step1": True, "step2": False})
        assert result["score"] == 0.3
        
        # Both satisfied (100%)
        result = checklist.evaluate({"step1": True, "step2": True})
        assert result["score"] == 1.0
    
    def test_get_score(self):
        """Test getting aggregate score."""
        checklist = Checklist()
        checklist.add_criterion("A", weight=1.0, check_function=lambda d: True)
        checklist.add_criterion("B", weight=1.0, check_function=lambda d: False)
        
        # Only A satisfied = 0.5
        checklist.evaluate({"anything": True})
        assert checklist.get_score() == 0.5
    
    def test_get_score_empty(self):
        """Test score with no criteria."""
        checklist = Checklist()
        assert checklist.get_score() == 0.0
    
    def test_is_complete(self):
        """Test is_complete method."""
        checklist = Checklist()
        checklist.add_criterion("A", weight=1.0, check_function=lambda d: d.get("done", False))
        
        assert checklist.is_complete() is False
        
        checklist.evaluate({"done": True})
        assert checklist.is_complete() is True
    
    def test_get_failed_criteria(self):
        """Test getting failed criteria."""
        checklist = Checklist()
        c1 = checklist.add_criterion("A", weight=1.0, check_function=lambda d: True)
        c2 = checklist.add_criterion("B", weight=1.0, check_function=lambda d: False)
        
        checklist.evaluate({"anything": True})
        
        failed = checklist.get_failed_criteria()
        assert len(failed) == 1
        assert failed[0].description == "B"
    
    def test_get_satisfied_criteria(self):
        """Test getting satisfied criteria."""
        checklist = Checklist()
        c1 = checklist.add_criterion("A", weight=1.0, check_function=lambda d: True)
        c2 = checklist.add_criterion("B", weight=1.0, check_function=lambda d: False)
        
        checklist.evaluate({"anything": True})
        
        satisfied = checklist.get_satisfied_criteria()
        assert len(satisfied) == 1
        assert satisfied[0].description == "A"
    
    def test_reset(self):
        """Test resetting checklist."""
        checklist = Checklist()
        checklist.add_criterion("Test", weight=1.0, check_function=lambda d: True)
        checklist.evaluate({"anything": True})
        
        assert checklist.is_complete() is True
        
        checklist.reset()
        assert checklist.is_complete() is False
        assert checklist.get_score() == 0.0
    
    def test_to_dict(self):
        """Test converting to dictionary."""
        checklist = Checklist(name="Test", task_type="search")
        checklist.add_criterion("Test", weight=0.5, check_function=lambda d: True)
        
        d = checklist.to_dict()
        assert d["name"] == "Test"
        assert d["task_type"] == "search"
        assert len(d["criteria"]) == 1
    
    # Legacy tests (backward compatibility)
    def test_legacy_add_item(self):
        """Test legacy add_item method."""
        checklist = Checklist()
        item = checklist.add_item("task1", "Complete task 1")
        assert item.id == "task1"
        assert len(checklist.items) == 1
    
    def test_legacy_complete_item(self):
        """Test legacy complete method."""
        checklist = Checklist()
        checklist.add_item("task1", "Task 1")
        success = checklist.complete("task1")
        assert success is True
        assert checklist.get_item("task1").status == CheckStatus.COMPLETED
    
    def test_legacy_is_complete(self):
        """Test legacy is_complete with items."""
        checklist = Checklist()
        checklist.add_item("task1", "Task 1")
        checklist.add_item("task2", "Task 2")
        
        assert checklist.is_complete() is False
        checklist.complete("task1")
        checklist.complete("task2")
        assert checklist.is_complete() is True


class TestChecklistEvaluator:
    """Tests for ChecklistEvaluator."""
    
    def test_initialization(self):
        """Test evaluator initialization."""
        evaluator = ChecklistEvaluator()
        assert len(evaluator.checklists) == 0
        assert evaluator.current_checklist is None
    
    def test_create_navigation_checklist(self):
        """Test creating navigation checklist."""
        evaluator = ChecklistEvaluator()
        checklist = evaluator.create_checklist("navigation")
        
        assert checklist.task_type == "navigation"
        assert len(checklist.criteria) == 3
        
        # Check criteria exist
        descriptions = [c.description for c in checklist.criteria]
        assert "Reached target domain" in descriptions
        assert "Found interactive element" in descriptions
        assert "Successfully navigated to target page" in descriptions
    
    def test_create_search_checklist(self):
        """Test creating search checklist."""
        evaluator = ChecklistEvaluator()
        checklist = evaluator.create_checklist("search")
        
        assert checklist.task_type == "search"
        assert len(checklist.criteria) == 4
        
        descriptions = [c.description for c in checklist.criteria]
        assert "Submitted query" in descriptions
        assert "Received search results" in descriptions
        assert "Verified result relevance" in descriptions
    
    def test_create_form_checklist(self):
        """Test creating form checklist."""
        evaluator = ChecklistEvaluator()
        checklist = evaluator.create_checklist("form")
        
        assert checklist.task_type == "form"
        assert len(checklist.criteria) == 4
        
        descriptions = [c.description for c in checklist.criteria]
        assert "Located form" in descriptions
        assert "Filled required fields" in descriptions
        assert "Submitted successfully" in descriptions
    
    def test_create_info_extraction_checklist(self):
        """Test creating info extraction checklist."""
        evaluator = ChecklistEvaluator()
        checklist = evaluator.create_checklist("information_extraction")
        
        assert checklist.task_type == "information_extraction"
        assert len(checklist.criteria) == 3
    
    def test_create_unknown_checklist(self):
        """Test creating checklist for unknown type."""
        evaluator = ChecklistEvaluator()
        checklist = evaluator.create_checklist("unknown_type")
        
        assert checklist.task_type == "unknown_type"
        # Should have minimal default criteria
        assert len(checklist.criteria) >= 1
    
    def test_evaluate_step(self):
        """Test evaluating a step."""
        evaluator = ChecklistEvaluator()
        checklist = evaluator.create_checklist("navigation")
        
        step_data = {
            "url": "https://example.com",
            "target_domain": "example.com",
            "action": "click button",
            "observation": "Found submit button",
            "metadata": {"target_reached": True}
        }
        
        result = evaluator.evaluate_step(step_data)
        
        assert "score" in result
        assert "progress" in result
        assert "step_index" in result
    
    def test_evaluate_step_no_checklist(self):
        """Test evaluating without a checklist raises error."""
        evaluator = ChecklistEvaluator()
        
        with pytest.raises(ValueError):
            evaluator.evaluate_step({})
    
    def test_evaluation_history(self):
        """Test evaluation history tracking."""
        evaluator = ChecklistEvaluator()
        checklist = evaluator.create_checklist("navigation")
        
        # Evaluate multiple steps
        evaluator.evaluate_step({"url": "https://example.com", "target_domain": "example.com"})
        evaluator.evaluate_step({"url": "https://example.com", "target_domain": "example.com"})
        
        assert len(evaluator.evaluation_history) == 2
    
    def test_get_evaluation_summary(self):
        """Test getting evaluation summary."""
        evaluator = ChecklistEvaluator()
        checklist = evaluator.create_checklist("navigation")
        
        # Evaluate some steps
        evaluator.evaluate_step({"url": "https://example.com", "target_domain": "example.com", "metadata": {"target_reached": True}})
        
        summary = evaluator.get_evaluation_summary()
        
        assert "total_evaluations" in summary
        assert "final_score" in summary
        assert "is_complete" in summary
        assert summary["total_evaluations"] == 1
    
    def test_reset_evaluator(self):
        """Test resetting evaluator."""
        evaluator = ChecklistEvaluator()
        evaluator.create_checklist("navigation")
        evaluator.evaluate_step({"url": "https://example.com"})
        
        evaluator.reset()
        
        assert len(evaluator.evaluation_history) == 0
        assert evaluator.current_checklist is not None
        assert evaluator.current_checklist.get_score() == 0.0
    
    def test_check_domain_match(self):
        """Test domain matching check function."""
        evaluator = ChecklistEvaluator()
        
        # Matching domain
        assert evaluator._check_domain_match({
            "url": "https://example.com/page",
            "target_domain": "example.com"
        }) is True
        
        # Non-matching domain
        assert evaluator._check_domain_match({
            "url": "https://other.com/page",
            "target_domain": "example.com"
        }) is False
        
        # Via metadata
        assert evaluator._check_domain_match({
            "url": "https://any.com",
            "metadata": {"domain_match": True}
        }) is True
    
    def test_check_interactive_element(self):
        """Test interactive element detection."""
        evaluator = ChecklistEvaluator()
        
        assert evaluator._check_interactive_element({
            "observation": "Found a clickable button"
        }) is True
        
        assert evaluator._check_interactive_element({
            "observation": "Just some text content"
        }) is False
    
    def test_check_query_submitted(self):
        """Test query submission check."""
        evaluator = ChecklistEvaluator()
        
        # Via action
        assert evaluator._check_query_submitted({
            "action": "submit search"
        }) is True
        
        # Via metadata
        assert evaluator._check_query_submitted({
            "action": "click",
            "metadata": {"query_submitted": True}
        }) is True
    
    def test_check_results_received(self):
        """Test results receiving check."""
        evaluator = ChecklistEvaluator()
        
        assert evaluator._check_results_received({
            "observation": "Showing 10 results"
        }) is True
        
        assert evaluator._check_results_received({
            "observation": "Found 5 matches"
        }) is True
        
        # Via metadata
        assert evaluator._check_results_received({
            "observation": "",
            "metadata": {"results_count": 5}
        }) is True


class TestUtilityFunctions:
    """Tests for utility functions."""
    
    def test_create_checklist_convenience(self):
        """Test convenience function for creating checklists."""
        checklist = create_checklist("search")
        
        assert checklist is not None
        assert checklist.task_type == "search"
        assert len(checklist.criteria) > 0
    
    def test_evaluate_trajectory(self):
        """Test evaluating entire trajectory."""
        trajectory = [
            {"url": "https://example.com", "target_domain": "example.com", "metadata": {"domain_match": True}},
            {"url": "https://example.com", "target_domain": "example.com", "action": "submit", "metadata": {"query_submitted": True}},
        ]
        
        result = evaluate_trajectory(trajectory, "navigation")
        
        assert "final_score" in result
        assert "total_evaluations" in result
        assert result["total_evaluations"] == 2
    
    def test_evaluate_trajectory_empty(self):
        """Test evaluating empty trajectory."""
        result = evaluate_trajectory([], "search")
        
        assert result["total_evaluations"] == 0
        assert result["final_score"] == 0.0


class TestCM2Concepts:
    """Tests specifically for CM2 concepts."""
    
    def test_fine_grained_feedback(self):
        """Test that we get fine-grained feedback on partial completion."""
        checklist = Checklist()
        checklist.add_criterion("Step 1", weight=0.5, check_function=lambda d: d.get("step1"))
        checklist.add_criterion("Step 2", weight=0.5, check_function=lambda d: d.get("step2"))
        
        # Partial completion gives feedback
        result = checklist.evaluate({"step1": True, "step2": False})
        
        assert result["score"] == 0.5
        assert result["progress"] == 50.0
        assert result["is_complete"] is False
        assert "Step 1" in result["all_satisfied"]
        assert "Step 2" not in result["all_satisfied"]
    
    def test_credit_assignment(self):
        """Test credit assignment across multi-turn interactions."""
        evaluator = ChecklistEvaluator()
        checklist = evaluator.create_checklist("search")
        
        # Turn 1: Locate search interface
        result1 = evaluator.evaluate_step({
            "observation": "Search input found",
            "metadata": {"search_interface_found": True}
        })
        
        # Turn 2: Submit query
        result2 = evaluator.evaluate_step({
            "action": "submit query",
            "metadata": {"query_submitted": True}
        })
        
        # Turn 3: Receive results
        result3 = evaluator.evaluate_step({
            "observation": "Found 10 results",
            "metadata": {"results_count": 10}
        })
        
        # Each step should show progress
        assert result1["progress"] < result2["progress"] <= result3["progress"]
        
        # Final should show partial completion
        final = evaluator.get_evaluation_summary()
        assert final["final_progress"] > 0
    
    def test_weighted_scoring(self):
        """Test that weighted scoring works correctly."""
        checklist = Checklist()
        # High weight criterion
        checklist.add_criterion("Critical", weight=0.8, check_function=lambda d: d.get("critical"))
        # Low weight criterion
        checklist.add_criterion("Nice to have", weight=0.2, check_function=lambda d: d.get("nice"))
        
        # Only critical satisfied = 0.8
        result = checklist.evaluate({"critical": True, "nice": False})
        assert result["score"] == 0.8
        
        # Both satisfied = 1.0
        result = checklist.evaluate({"critical": True, "nice": True})
        assert result["score"] == 1.0
        
        # Only nice to have satisfied = 0.2
        checklist.reset()
        checklist.evaluate({"critical": False, "nice": True})
        assert checklist.get_score() == 0.2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
