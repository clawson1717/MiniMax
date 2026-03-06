import pytest
import re
from src.model_wrapper import DraftWrapper

def test_draft_wrapper_wrap_prompt():
    wrapper = DraftWrapper()
    prompt = "What is the capital of France?"
    wrapped = wrapper.wrap_prompt(prompt)
    
    assert prompt in wrapped
    assert wrapper.DRAFT_START in wrapped
    assert wrapper.DRAFT_END in wrapped

def test_draft_wrapper_extract_draft_success():
    wrapper = DraftWrapper()
    output = f"Let's think. {wrapper.DRAFT_START} Reasoning here {wrapper.DRAFT_END} Final Answer: Paris"
    draft = wrapper.extract_draft(output)
    
    assert draft == "Reasoning here"

def test_draft_wrapper_extract_draft_empty():
    wrapper = DraftWrapper()
    output = "Paris is the capital."
    draft = wrapper.extract_draft(output)
    
    assert draft == ""

def test_draft_wrapper_extract_draft_multiline():
    wrapper = DraftWrapper()
    output = f"""
    {wrapper.DRAFT_START}
    Line 1
    Line 2
    {wrapper.DRAFT_END}
    Final Answer
    """
    draft = wrapper.extract_draft(output)
    
    assert "Line 1" in draft
    assert "Line 2" in draft

def test_draft_wrapper_compress_cot_short():
    wrapper = DraftWrapper()
    cot = "Short CoT"
    compressed = wrapper.compress_cot(cot)
    assert compressed == cot

def test_draft_wrapper_compress_cot_long():
    wrapper = DraftWrapper()
    cot = "X" * 150
    compressed = wrapper.compress_cot(cot)
    assert len(compressed) <= 100
    assert compressed.endswith("...")
