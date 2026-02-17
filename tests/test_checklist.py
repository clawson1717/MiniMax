"""Tests for checklist module."""

import pytest
from src.checklist import Checklist, ChecklistItem, CheckStatus


class TestChecklistItem:
    """Tests for ChecklistItem."""
    
    def test_item_creation(self):
        """Test creating a checklist item."""
        item = ChecklistItem(id="1", description="Do something")
        assert item.id == "1"
        assert item.description == "Do something"
        assert item.status == CheckStatus.PENDING
    
    def test_item_to_dict(self):
        """Test converting item to dict."""
        item = ChecklistItem(id="1", description="Test")
        d = item.to_dict()
        assert d["id"] == "1"
        assert d["status"] == "pending"


class TestChecklist:
    """Tests for Checklist."""
    
    def test_initialization(self):
        """Test checklist initialization."""
        checklist = Checklist(name="My Tasks")
        assert checklist.name == "My Tasks"
        assert len(checklist.items) == 0
    
    def test_add_item(self):
        """Test adding items."""
        checklist = Checklist()
        item = checklist.add_item("task1", "Complete task 1")
        assert item.id == "task1"
        assert len(checklist.items) == 1
    
    def test_get_item(self):
        """Test getting an item."""
        checklist = Checklist()
        checklist.add_item("task1", "Task 1")
        item = checklist.get_item("task1")
        assert item is not None
        assert item.description == "Task 1"
    
    def test_complete_item(self):
        """Test completing an item."""
        checklist = Checklist()
        checklist.add_item("task1", "Task 1")
        success = checklist.complete("task1")
        assert success is True
        assert checklist.get_item("task1").status == CheckStatus.COMPLETED
    
    def test_is_complete(self):
        """Test checking if all complete."""
        checklist = Checklist()
        checklist.add_item("task1", "Task 1")
        checklist.add_item("task2", "Task 2")
        assert checklist.is_complete() is False
        checklist.complete("task1")
        checklist.complete("task2")
        assert checklist.is_complete() is True
    
    def test_dependencies(self):
        """Test item dependencies."""
        checklist = Checklist()
        checklist.add_item("task1", "Task 1")
        checklist.add_item("task2", "Task 2", depends_on=["task1"])
        
        # Can't complete task2 before task1
        success = checklist.complete("task2")
        assert success is False
        
        checklist.complete("task1")
        success = checklist.complete("task2")
        assert success is True
    
    def test_get_ready_items(self):
        """Test getting ready items."""
        checklist = Checklist()
        checklist.add_item("task1", "Task 1")
        checklist.add_item("task2", "Task 2", depends_on=["task1"])
        
        ready = checklist.get_ready_items()
        assert len(ready) == 1
        assert ready[0].id == "task1"
    
    def test_get_progress(self):
        """Test progress tracking."""
        checklist = Checklist()
        checklist.add_item("task1", "Task 1")
        checklist.add_item("task2", "Task 2")
        checklist.complete("task1")
        
        progress = checklist.get_progress()
        assert progress["total"] == 2
        assert progress["completed"] == 1
        assert progress["percent_complete"] == 50
    
    def test_reset(self):
        """Test resetting checklist."""
        checklist = Checklist()
        checklist.add_item("task1", "Task 1")
        checklist.complete("task1")
        checklist.reset()
        assert checklist.get_item("task1").status == CheckStatus.PENDING
