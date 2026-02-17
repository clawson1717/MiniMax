"""Tests for skill module."""

import pytest
from src.skill import (
    Skill, SkillLibrary, SkillRegistry, SkillMetadata, SkillResult,
    SkillCategory, skill
)


class TestSkillMetadata:
    """Tests for SkillMetadata."""
    
    def test_metadata_creation(self):
        """Test creating skill metadata."""
        metadata = SkillMetadata(
            name="test_skill",
            description="A test skill",
            category=SkillCategory.UTILITY
        )
        
        assert metadata.name == "test_skill"
        assert metadata.description == "A test skill"
        assert metadata.category == SkillCategory.UTILITY
    
    def test_metadata_to_dict(self):
        """Test converting metadata to dict."""
        metadata = SkillMetadata(
            name="test_skill",
            description="A test skill",
            category=SkillCategory.CODE,
            tags=["testing", "example"]
        )
        
        data = metadata.to_dict()
        
        assert data["name"] == "test_skill"
        assert data["category"] == "code"
        assert "testing" in data["tags"]


class TestSkillResult:
    """Tests for SkillResult."""
    
    def test_successful_result(self):
        """Test successful skill result."""
        result = SkillResult(
            success=True,
            output={"data": "test"}
        )
        
        assert result.success is True
        assert result.output == {"data": "test"}
        assert result.error is None
    
    def test_failed_result(self):
        """Test failed skill result."""
        result = SkillResult(
            success=False,
            error="Something went wrong"
        )
        
        assert result.success is False
        assert result.error == "Something went wrong"
    
    def test_result_with_metadata(self):
        """Test result with metadata."""
        result = SkillResult(
            success=True,
            output="result",
            metadata={"execution_time": 0.5}
        )
        
        assert result.metadata["execution_time"] == 0.5


class TestSkillCategory:
    """Tests for SkillCategory enum."""
    
    def test_all_categories(self):
        """Test all skill categories."""
        assert SkillCategory.FILE_IO.value == "file_io"
        assert SkillCategory.CODE.value == "code"
        assert SkillCategory.SEARCH.value == "search"
        assert SkillCategory.WEB.value == "web"
        assert SkillCategory.DATA.value == "data"
        assert SkillCategory.UTILITY.value == "utility"
        assert SkillCategory.CUSTOM.value == "custom"


class DummySkill(Skill):
    """A dummy skill for testing."""
    
    def _create_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="dummy",
            description="A dummy skill for testing",
            category=SkillCategory.UTILITY
        )
    
    def execute(self, context):
        return SkillResult(success=True, output="dummy executed")


class TestSkill:
    """Tests for Skill base class."""
    
    def test_skill_creation(self):
        """Test creating a skill."""
        skill = DummySkill()
        
        assert skill.metadata.name == "dummy"
        assert skill.metadata.category == SkillCategory.UTILITY
    
    def test_skill_describe(self):
        """Test skill describe method."""
        skill = DummySkill()
        
        assert "dummy" in skill.describe().lower()
    
    def test_skill_validate(self):
        """Test skill validate method."""
        skill = DummySkill()
        
        valid, error = skill.validate({})
        
        assert valid is True
        assert error is None
    
    def test_skill_execute(self):
        """Test skill execute method."""
        skill = DummySkill()
        result = skill.execute({})
        
        assert result.success is True
        assert result.output == "dummy executed"
    
    def test_skill_hooks(self):
        """Test skill hooks."""
        skill = DummySkill()
        hook_called = []
        
        def callback(data):
            hook_called.append(data)
        
        skill.register_hook("test_event", callback)
        skill._trigger_hook("test_event", "test_data")
        
        assert hook_called == ["test_data"]
    
    def test_get_dependencies(self):
        """Test getting skill dependencies."""
        skill = DummySkill()
        
        assert skill.get_dependencies() == []


class TestSkillLibrary:
    """Tests for SkillLibrary."""
    
    def test_library_creation(self):
        """Test creating a skill library."""
        library = SkillLibrary()
        
        assert library.list_skills() == []
    
    def test_register_skill(self):
        """Test registering a skill."""
        library = SkillLibrary()
        skill = DummySkill()
        
        library.register(skill)
        
        assert "dummy" in library.list_skills()
    
    def test_get_skill(self):
        """Test getting a skill."""
        library = SkillLibrary()
        skill = DummySkill()
        library.register(skill)
        
        retrieved = library.get("dummy")
        
        assert retrieved is not None
        assert retrieved.metadata.name == "dummy"
    
    def test_list_by_category(self):
        """Test listing skills by category."""
        library = SkillLibrary()
        skill = DummySkill()
        library.register(skill)
        
        skills = library.list_skills(SkillCategory.UTILITY)
        
        assert "dummy" in skills
    
    def test_search_skills(self):
        """Test searching skills."""
        library = SkillLibrary()
        skill = DummySkill()
        library.register(skill)
        
        results = library.search("dummy")
        
        assert len(results) == 1
        assert results[0].metadata.name == "dummy"
    
    def test_get_by_tag(self):
        """Test getting skills by tag."""
        library = SkillLibrary()
        
        # Create skill with tags
        skill = DummySkill()
        skill._metadata = SkillMetadata(
            name="tagged_skill",
            description="A skill with tags",
            category=SkillCategory.UTILITY,
            tags=["important", "testing"]
        )
        library.register(skill)
        
        results = library.get_by_tag("important")
        
        assert len(results) == 1
        assert results[0].metadata.name == "tagged_skill"
    
    def test_clear_library(self):
        """Test clearing library."""
        library = SkillLibrary()
        library.register(DummySkill())
        
        library.clear()
        
        assert library.list_skills() == []


class TestSkillRegistry:
    """Tests for global skill registry."""
    
    def test_get_instance(self):
        """Test getting registry instance."""
        registry = SkillRegistry.get_instance()
        
        assert isinstance(registry, SkillLibrary)
    
    def test_register_skill_class(self):
        """Test registering a skill class."""
        # Clear first
        SkillRegistry._instance = None
        
        SkillRegistry.register_skill(DummySkill)
        
        skill = SkillRegistry.get_skill("dummy")
        
        assert skill is not None
    
    def test_list_skills_from_registry(self):
        """Test listing skills from registry."""
        # Clear first
        SkillRegistry._instance = None
        SkillRegistry.register_skill(DummySkill)
        
        skills = SkillRegistry.list_skills()
        
        assert "dummy" in skills


class TestSkillDecorator:
    """Tests for skill decorator."""
    
    def test_skill_decorator(self):
        """Test the @skill decorator - basic functionality."""
        from src.skill import skill as skill_decorator
        
        # Just verify the decorator function exists and is callable
        assert callable(skill_decorator)
        # The full decorator test would require more setup
