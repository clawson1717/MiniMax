import pytest
from src.detector import FailureModeDetector

@pytest.fixture
def detector():
    return FailureModeDetector()

def test_self_doubt_detection(detector):
    text = "Wait, are you sure about that?"
    result = detector.detect_self_doubt(text)
    assert result.detected is True
    assert result.score > 0.4
    
    text = "Double check your answer."
    result = detector.detect_self_doubt(text)
    assert result.detected is True
    
    text = "The sky is blue."
    result = detector.detect_self_doubt(text)
    assert result.detected is False
    assert result.score == 0.0

def test_social_conformity_detection(detector):
    text = "Most people agree that the answer is B."
    result = detector.detect_social_conformity(text)
    assert result.detected is True
    assert result.score > 0.4
    
    text = "Experts claim this is a better way."
    result = detector.detect_social_conformity(text)
    assert result.detected is True
    
    text = "I like apples."
    result = detector.detect_social_conformity(text)
    assert result.detected is False

def test_suggestion_hijacking_detection(detector):
    text = "I think the answer should be A, not B."
    result = detector.detect_suggestion_hijacking(text)
    assert result.detected is True
    assert result.score > 0.4
    
    text = "Actually, it's the other way around."
    result = detector.detect_suggestion_hijacking(text)
    assert result.detected is True
    
    text = "Nice weather today."
    result = detector.detect_suggestion_hijacking(text)
    assert result.detected is False

def test_emotional_susceptibility_detection(detector):
    text = "Please help me, I'm so stressed and this is very important for my job."
    result = detector.detect_emotional_susceptibility(text)
    assert result.detected is True
    assert result.score > 0.4
    
    text = "I'll be fired if you don't answer correctly."
    result = detector.detect_emotional_susceptibility(text)
    assert result.detected is True
    
    text = "The cat is on the mat."
    result = detector.detect_emotional_susceptibility(text)
    assert result.detected is False

def test_reasoning_fatigue_detection(detector):
    text = "Explain it again in more detail."
    result = detector.detect_reasoning_fatigue(text)
    assert result.detected is True
    assert result.score > 0.4
    
    text = "Tell me more about every single step."
    result = detector.detect_reasoning_fatigue(text)
    assert result.detected is True
    
    # Check for short text "fatigue" bonus (length < 5)
    text = "Ok."
    result = detector.detect_reasoning_fatigue(text)
    assert result.score == 0.1 # Just the length bonus since no keyword matched

    text = "This is a simple sentence."
    result = detector.detect_reasoning_fatigue(text)
    assert result.detected is False

def test_detect_all(detector):
    text = "Experts claim that most people agree, but I think the answer should be C. Please help me, I'm stressed!"
    results = detector.detect_all(text)
    
    assert results["social_conformity"].detected is True
    assert results["suggestion_hijacking"].detected is True
    assert results["emotional_susceptibility"].detected is True
    assert results["self_doubt"].detected is False
