"""Core skill classes for the adaptive skill composer."""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
from abc import ABC, abstractmethod
from enum import Enum
import json


class SkillCategory(Enum):
    """Categories of skills."""
    FILE_IO = "file_io"
    CODE = "code"
    SEARCH = "search"
    WEB = "web"
    DATA = "data"
    UTILITY = "utility"
    CUSTOM = "custom"


@dataclass
class SkillMetadata:
    """Metadata for a skill."""
    name: str
    description: str
    category: SkillCategory
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    returns: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "parameters": self.parameters,
            "returns": self.returns,
            "dependencies": self.dependencies,
            "tags": self.tags,
        }


@dataclass
class SkillResult:
    """Result of skill execution."""
    success: bool
    output: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "metadata": self.metadata,
        }


class Skill(ABC):
    """Base class for all skills."""
    
    def __init__(self):
        self._metadata: Optional[SkillMetadata] = None
        self._hooks: Dict[str, Callable] = {}
    
    @property
    def metadata(self) -> SkillMetadata:
        """Get skill metadata."""
        if self._metadata is None:
            self._metadata = self._create_metadata()
        return self._metadata
    
    @abstractmethod
    def _create_metadata(self) -> SkillMetadata:
        """Create skill metadata. Override in subclass."""
        pass
    
    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> SkillResult:
        """Execute the skill. Override in subclass."""
        pass
    
    def validate(self, context: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate input context. Override in subclass if needed."""
        return True, None
    
    def describe(self) -> str:
        """Get human-readable description."""
        return self.metadata.description
    
    def get_dependencies(self) -> List[str]:
        """Get list of skill names this skill depends on."""
        return self.metadata.dependencies
    
    def register_hook(self, event: str, callback: Callable) -> None:
        """Register a hook callback for skill events."""
        self._hooks[event] = callback
    
    def _trigger_hook(self, event: str, data: Any = None) -> None:
        """Trigger a registered hook."""
        if event in self._hooks:
            self._hooks[event](data)


class SkillLibrary:
    """Manages a collection of skills."""
    
    def __init__(self):
        self._skills: Dict[str, Skill] = {}
        self._categories: Dict[SkillCategory, List[str]] = {}
    
    def register(self, skill: Skill) -> None:
        """Register a skill in the library."""
        name = skill.metadata.name
        self._skills[name] = skill
        
        # Track by category
        category = skill.metadata.category
        if category not in self._categories:
            self._categories[category] = []
        if name not in self._categories[category]:
            self._categories[category].append(name)
    
    def get(self, name: str) -> Optional[Skill]:
        """Get a skill by name."""
        return self._skills.get(name)
    
    def list_skills(self, category: Optional[SkillCategory] = None) -> List[str]:
        """List skill names, optionally filtered by category."""
        if category is None:
            return list(self._skills.keys())
        return self._categories.get(category, [])
    
    def get_by_tag(self, tag: str) -> List[Skill]:
        """Get skills matching a tag."""
        return [
            skill for skill in self._skills.values()
            if tag in skill.metadata.tags
        ]
    
    def search(self, query: str) -> List[Skill]:
        """Search skills by name or description."""
        query_lower = query.lower()
        results = []
        for skill in self._skills.values():
            if query_lower in skill.metadata.name.lower():
                results.append(skill)
            elif query_lower in skill.metadata.description.lower():
                results.append(skill)
        return results
    
    def get_metadata(self, name: str) -> Optional[SkillMetadata]:
        """Get metadata for a skill."""
        skill = self.get(name)
        return skill.metadata if skill else None
    
    def all_metadata(self) -> List[SkillMetadata]:
        """Get metadata for all skills."""
        return [skill.metadata for skill in self._skills.values()]
    
    def clear(self) -> None:
        """Clear all skills."""
        self._skills.clear()
        self._categories.clear()


class SkillRegistry:
    """Global skill registry for discovering skills."""
    
    _instance: Optional[SkillLibrary] = None
    
    @classmethod
    def get_instance(cls) -> SkillLibrary:
        """Get the global skill library instance."""
        if cls._instance is None:
            cls._instance = SkillLibrary()
        return cls._instance
    
    @classmethod
    def register_skill(cls, skill_class: type) -> None:
        """Register a skill class."""
        library = cls.get_instance()
        skill = skill_class()
        library.register(skill)
    
    @classmethod
    def get_skill(cls, name: str) -> Optional[Skill]:
        """Get a skill by name."""
        return cls.get_instance().get(name)
    
    @classmethod
    def list_skills(cls, category: Optional[SkillCategory] = None) -> List[str]:
        """List available skills."""
        return cls.get_instance().list_skills(category)


def skill(name: str, description: str, category: SkillCategory, 
          parameters: List[Dict[str, Any]] = None):
    """Decorator to register a skill class."""
    def decorator(cls):
        original_create_metadata = cls._create_metadata
        
        def new_create_metadata(self):
            metadata = original_create_metadata(self)
            return SkillMetadata(
                name=name,
                description=description,
                category=category,
                parameters=parameters or [],
            )
        
        cls._create_metadata = new_create_metadata
        return cls
    
    return decorator
