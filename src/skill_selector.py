"""Skill selector for matching tasks to skills."""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re

from src.skill import Skill, SkillLibrary, SkillMetadata, SkillCategory


@dataclass
class SkillMatch:
    """A match between a task and a skill."""
    skill_name: str
    score: float
    reason: str


class SkillSelector:
    """Selects appropriate skills for a given task."""
    
    def __init__(self, library: SkillLibrary):
        """Initialize skill selector.
        
        Args:
            library: The skill library to select from.
        """
        self.library = library
        self._keyword_index: Dict[str, List[str]] = {}
    
    def select(self, task: str, context: Optional[Dict[str, Any]] = None) -> List[SkillMatch]:
        """Select skills for a task.
        
        Args:
            task: Task description.
            context: Optional context (domain, requirements, etc.).
            
        Returns:
            List of SkillMatches sorted by score.
        """
        # Get all skills
        skills = self.library.list_skills()
        matches = []
        
        task_lower = task.lower()
        
        for skill_name in skills:
            skill = self.library.get(skill_name)
            if skill is None:
                continue
            
            score, reason = self._compute_match(skill, task, task_lower, context)
            
            if score > 0:
                matches.append(SkillMatch(
                    skill_name=skill_name,
                    score=score,
                    reason=reason
                ))
        
        # Sort by score descending
        matches.sort(key=lambda m: m.score, reverse=True)
        
        return matches
    
    def _compute_match(
        self, 
        skill: Skill, 
        task: str, 
        task_lower: str,
        context: Optional[Dict[str, Any]]
    ) -> Tuple[float, str]:
        """Compute match score for a skill."""
        metadata = skill.metadata
        score = 0.0
        reasons = []
        
        # Check domain/context match
        if context and "domain" in context:
            domain = context["domain"].lower()
            if domain in [c.value for c in SkillCategory]:
                if metadata.category.value == domain:
                    score += 0.3
                    reasons.append(f"category match: {domain}")
        
        # Check keyword matches in name/description
        name_lower = metadata.name.lower()
        desc_lower = metadata.description.lower()
        
        # Extract key terms from task
        task_terms = self._extract_keywords(task_lower)
        
        # Match against name
        name_matches = sum(1 for term in task_terms if term in name_lower)
        if name_matches > 0:
            score += name_matches * 0.25
            reasons.append(f"{name_matches} term(s) in name")
        
        # Match against description
        desc_matches = sum(1 for term in task_terms if term in desc_lower)
        if desc_matches > 0:
            score += desc_matches * 0.15
            reasons.append(f"{desc_matches} term(s) in description")
        
        # Check tags
        for tag in metadata.tags:
            if tag.lower() in task_lower:
                score += 0.2
                reasons.append(f"tag match: {tag}")
        
        # Check dependencies - skills that provide needed capabilities
        if context and "required_skills" in context:
            required = context["required_skills"]
            for req in required:
                if req in metadata.dependencies:
                    score += 0.3
                    reasons.append(f"dependency: {req}")
        
        reason_str = ", ".join(reasons) if reasons else "no match"
        return score, reason_str
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        # Common programming/file操作 terms
        common_terms = [
            "read", "write", "file", "search", "code", "run", "execute",
            "web", "http", "api", "data", "parse", "format", "convert",
            "test", "debug", "build", "deploy", "git", "docker", "database",
            "query", "filter", "transform", "validate", "transform"
        ]
        
        # Extract words
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter to meaningful keywords
        keywords = [w for w in words if len(w) > 2]
        
        # Add common terms that appear
        for term in common_terms:
            if term in text:
                keywords.append(term)
        
        return list(set(keywords))
    
    def select_top(self, task: str, n: int = 3, context: Optional[Dict[str, Any]] = None) -> List[str]:
        """Select top N skills.
        
        Args:
            task: Task description.
            n: Number of skills to return.
            context: Optional context.
            
        Returns:
            List of skill names.
        """
        matches = self.select(task, context)
        return [m.skill_name for m in matches[:n]]
    
    def build_index(self) -> None:
        """Build keyword index for faster lookups."""
        self._keyword_index = {}
        
        for skill_name in self.library.list_skills():
            skill = self.library.get(skill_name)
            if skill is None:
                continue
            
            metadata = skill.metadata
            
            # Index by name tokens
            for token in metadata.name.lower().split('_'):
                if token not in self._keyword_index:
                    self._keyword_index[token] = []
                self._keyword_index[token].append(skill_name)
            
            # Index by category
            cat = metadata.category.value
            if cat not in self._keyword_index:
                self._keyword_index[cat] = []
            self._keyword_index[cat].append(skill_name)
    
    def get_suggestions(self, partial: str) -> List[str]:
        """Get skill suggestions from partial input.
        
        Args:
            partial: Partial skill name or term.
            
        Returns:
            List of matching skill names.
        """
        partial_lower = partial.lower()
        suggestions = set()
        
        for skill_name in self.library.list_skills():
            if partial_lower in skill_name.lower():
                suggestions.add(skill_name)
        
        return sorted(list(suggestions))


class FallbackSelector(SkillSelector):
    """Selector with fallback strategy for ambiguous tasks."""
    
    def __init__(self, library: SkillLibrary):
        super().__init__(library)
        self.fallback_skill = "utility"  # Default fallback
    
    def select_with_fallback(
        self, 
        task: str, 
        context: Optional[Dict[str, Any]] = None,
        min_score: float = 0.1
    ) -> List[SkillMatch]:
        """Select skills with fallback to default."""
        matches = self.select(task, context)
        
        # If no good matches, add fallback
        if not matches or matches[0].score < min_score:
            # Try to find a general-purpose skill
            for skill_name in self.library.list_skills():
                skill = self.library.get(skill_name)
                if skill and skill.metadata.category == SkillCategory.UTILITY:
                    matches.append(SkillMatch(
                        skill_name=skill_name,
                        score=0.1,
                        reason="fallback"
                    ))
                    break
        
        return matches
