"""Tests for skill selector module."""

import pytest
from src.skill_selector import SkillSelector, FallbackSelector, SkillMatch
from src.skill import Skill, SkillLibrary, SkillMetadata, SkillResult, SkillCategory


class DummyFileSkill(Skill):
    """A file I/O skill for testing."""
    
    def _create_metadata(self):
        return SkillMetadata(
            name="file_read",
            description="Read content from a file",
            category=SkillCategory.FILE_IO,
            tags=["file", "io", "read"]
        )
    
    def execute(self, context):
        return SkillResult(success=True)


class DummyWebSkill(Skill):
    """A web skill for testing."""
    
    def _create_metadata(self):
        return SkillMetadata(
            name="web_search",
            description="Search the web for information",
            category=SkillCategory.WEB,
            tags=["web", "search", "internet"]
        )
    
    def execute(self, context):
        return SkillResult(success=True)


class DummyCodeSkill(Skill):
    """A code skill for testing."""
    
    def _create_metadata(self):
        return SkillMetadata(
            name="code_execute",
            description="Execute code or scripts",
            category=SkillCategory.CODE,
            tags=["code", "execute", "run"]
        )
    
    def execute(self, context):
        return SkillResult(success=True)


class TestSkillSelector:
    """Tests for SkillSelector."""
    
    def test_selector_creation(self):
        """Test creating a skill selector."""
        library = SkillLibrary()
        selector = SkillSelector(library)
        
        assert selector.library is library
    
    def test_select_with_matching_skill(self):
        """Test selecting a skill that matches."""
        library = SkillLibrary()
        library.register(DummyFileSkill())
        
        selector = SkillSelector(library)
        matches = selector.select("read a file from disk")
        
        assert len(matches) > 0
        assert matches[0].skill_name == "file_read"
        assert matches[0].score > 0
    
    def test_select_no_match(self):
        """Test selecting with no matching skills."""
        library = SkillLibrary()
        library.register(DummyFileSkill())
        
        selector = SkillSelector(library)
        matches = selector.select("make coffee")
        
        # Should return empty or very low scores
        assert all(m.score == 0 for m in matches)
    
    def test_select_web_skill(self):
        """Test selecting a web skill."""
        library = SkillLibrary()
        library.register(DummyWebSkill())
        
        selector = SkillSelector(library)
        matches = selector.select("search the internet")
        
        assert len(matches) > 0
        assert "web_search" in [m.skill_name for m in matches]
    
    def test_select_top(self):
        """Test selecting top N skills."""
        library = SkillLibrary()
        library.register(DummyFileSkill())
        library.register(DummyWebSkill())
        library.register(DummyCodeSkill())
        
        selector = SkillSelector(library)
        top_skills = selector.select_top("read file and search web", n=2)
        
        assert len(top_skills) <= 2
    
    def test_extract_keywords(self):
        """Test keyword extraction."""
        library = SkillLibrary()
        selector = SkillSelector(library)
        
        keywords = selector._extract_keywords("read a file from disk")
        
        assert "read" in keywords
        assert "file" in keywords
    
    def test_build_index(self):
        """Test building keyword index."""
        library = SkillLibrary()
        library.register(DummyFileSkill())
        
        selector = SkillSelector(library)
        selector.build_index()
        
        assert len(selector._keyword_index) > 0
    
    def test_get_suggestions(self):
        """Test getting suggestions from partial input."""
        library = SkillLibrary()
        library.register(DummyFileSkill())
        library.register(DummyWebSkill())
        
        selector = SkillSelector(library)
        suggestions = selector.get_suggestions("file")
        
        assert "file_read" in suggestions
    
    def test_category_match(self):
        """Test matching by category in context."""
        library = SkillLibrary()
        library.register(DummyFileSkill())
        
        selector = SkillSelector(library)
        
        # With domain context
        matches = selector.select(
            "do something", 
            context={"domain": "file_io"}
        )
        
        assert len(matches) > 0
        assert matches[0].skill_name == "file_read"
    
    def test_tag_match(self):
        """Test matching by tags."""
        library = SkillLibrary()
        library.register(DummyFileSkill())
        
        selector = SkillSelector(library)
        matches = selector.select("I need to work with files")
        
        # Should match via "file" tag
        assert any(m.skill_name == "file_read" for m in matches)
    
    def test_multiple_skills_ranked(self):
        """Test that multiple skills are ranked correctly."""
        library = SkillLibrary()
        library.register(DummyFileSkill())  # Matches "read"
        library.register(DummyWebSkill())    # Doesn't match
        
        selector = SkillSelector(library)
        matches = selector.select("read a file")
        
        assert matches[0].skill_name == "file_read"
        assert matches[0].score > 0


class TestSkillMatch:
    """Tests for SkillMatch dataclass."""
    
    def test_skill_match_creation(self):
        """Test creating a SkillMatch."""
        match = SkillMatch(
            skill_name="test_skill",
            score=0.8,
            reason="perfect match"
        )
        
        assert match.skill_name == "test_skill"
        assert match.score == 0.8
        assert match.reason == "perfect match"


class DummyUtilitySkill(Skill):
    """A utility skill for testing fallback."""
    
    def _create_metadata(self):
        return SkillMetadata(
            name="utility_skill",
            description="A general purpose utility skill",
            category=SkillCategory.UTILITY,
            tags=["utility", "general"]
        )
    
    def execute(self, context):
        return SkillResult(success=True)


class TestFallbackSelector:
    """Tests for FallbackSelector."""
    
    def test_fallback_selector_creation(self):
        """Test creating a fallback selector."""
        library = SkillLibrary()
        selector = FallbackSelector(library)
        
        assert isinstance(selector, SkillSelector)
        assert selector.fallback_skill == "utility"
    
    def test_select_with_fallback_no_match(self):
        """Test fallback when no good match."""
        library = SkillLibrary()
        library.register(DummyFileSkill())
        library.register(DummyUtilitySkill())  # Add utility for fallback
        
        selector = FallbackSelector(library)
        matches = selector.select_with_fallback("make coffee", min_score=0.5)
        
        # Should have fallback
        assert len(matches) > 0
    
    def test_select_with_fallback_good_match(self):
        """Test fallback doesn't override good match."""
        library = SkillLibrary()
        library.register(DummyFileSkill())
        
        selector = FallbackSelector(library)
        matches = selector.select_with_fallback("read a file", min_score=0.1)
        
        assert matches[0].skill_name == "file_read"
