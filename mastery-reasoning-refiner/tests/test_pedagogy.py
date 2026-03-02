import pytest
from src.pedagogy import PedagogicalController, BloomTier, ZPDController

def test_initialization():
    controller = PedagogicalController(mastery_threshold=0.8)
    assert controller.get_current_tier() == BloomTier.KNOWLEDGE
    report = controller.get_mastery_report()
    assert all(score == 0.0 for score in report.values())

def test_zpd_controller_calculation():
    # Base difficulty 1.0, k=0.5. 
    # Capacity 10 bits. 
    # Target = 1.0 + (0.5 * 10) = 6.0
    controller = ZPDController(base_difficulty=1.0, k=0.5, alpha=1.0)
    
    # Within ZPD (default success 0.75)
    target = controller.calculate_difficulty_target(world_model_capacity=10.0)
    assert target == 6.0
    
    # Below ZPD (Success < 0.6)
    controller.update_performance(BloomTier.KNOWLEDGE, 0.3) 
    # Success rate becomes 0.3 (since alpha=1.0)
    # Target = 6.0 * (0.3 / 0.6) = 3.0
    target = controller.calculate_difficulty_target(world_model_capacity=10.0)
    assert target == 3.0
    
    # Above ZPD (Success > 0.9)
    controller.update_performance(BloomTier.KNOWLEDGE, 0.99)
    # Target = 6.0 * (0.99 / 0.9) = 6.6
    target = controller.calculate_difficulty_target(world_model_capacity=10.0)
    assert target == pytest.approx(6.6)

def test_mastery_progression():
    # Set alpha to 1.0 for predictable updates in test
    controller = PedagogicalController(mastery_threshold=0.8, alpha=1.0)
    
    # Start at Knowledge
    assert controller.get_current_tier() == BloomTier.KNOWLEDGE
    
    # Perform well on Knowledge
    controller.update_mastery(BloomTier.KNOWLEDGE, 0.9)
    assert controller.mastery_scores[BloomTier.KNOWLEDGE] == 0.9
    
    # Should move to Comprehension
    assert controller.get_current_tier() == BloomTier.COMPREHENSION
    
    # Perform well on Comprehension
    controller.update_mastery(BloomTier.COMPREHENSION, 0.85)
    assert controller.get_current_tier() == BloomTier.APPLICATION
    
    # Perform poorly on Application - should stay there
    controller.update_mastery(BloomTier.APPLICATION, 0.4)
    assert controller.get_current_tier() == BloomTier.APPLICATION

def test_ema_smoothing():
    controller = PedagogicalController(mastery_threshold=0.8, alpha=0.5)
    
    # Update once
    controller.update_mastery(BloomTier.KNOWLEDGE, 1.0)
    # (1-0.5)*0 + 0.5*1.0 = 0.5
    assert controller.mastery_scores[BloomTier.KNOWLEDGE] == 0.5
    assert controller.get_current_tier() == BloomTier.KNOWLEDGE
    
    # Update again
    controller.update_mastery(BloomTier.KNOWLEDGE, 1.0)
    # (1-0.5)*0.5 + 0.5*1.0 = 0.75
    assert controller.mastery_scores[BloomTier.KNOWLEDGE] == 0.75
    assert controller.get_current_tier() == BloomTier.KNOWLEDGE
    
    # Update third time
    controller.update_mastery(BloomTier.KNOWLEDGE, 1.0)
    # (1-0.5)*0.75 + 0.5*1.0 = 0.875
    assert controller.mastery_scores[BloomTier.KNOWLEDGE] == 0.875
    assert controller.get_current_tier() == BloomTier.COMPREHENSION

def test_regression_logic():
    # If a previous tier's score drops below threshold, should we move back?
    # Based on the current _update_current_focus logic, yes.
    controller = PedagogicalController(mastery_threshold=0.8, alpha=1.0)
    
    controller.update_mastery(BloomTier.KNOWLEDGE, 1.0)
    controller.update_mastery(BloomTier.COMPREHENSION, 1.0)
    assert controller.get_current_tier() == BloomTier.APPLICATION
    
    # Knowledge drops
    controller.update_mastery(BloomTier.KNOWLEDGE, 0.5)
    assert controller.get_current_tier() == BloomTier.KNOWLEDGE

def test_invalid_score():
    controller = PedagogicalController()
    with pytest.raises(ValueError):
        controller.update_mastery(BloomTier.KNOWLEDGE, 1.5)
    with pytest.raises(ValueError):
        controller.update_mastery(BloomTier.KNOWLEDGE, -0.1)

def test_reset():
    controller = PedagogicalController(alpha=1.0)
    controller.update_mastery(BloomTier.KNOWLEDGE, 1.0)
    controller.reset()
    assert controller.mastery_scores[BloomTier.KNOWLEDGE] == 0.0
    assert controller.get_current_tier() == BloomTier.KNOWLEDGE
