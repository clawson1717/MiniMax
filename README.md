# Adaptive Skill Composer

An adaptive code agent that dynamically composes skills from a library based on task context, uses graph-based trajectory pruning to optimize skill execution sequences, and defends against adversarial skill manipulation attacks.

## Overview

Combines three ArXiv techniques:
- **SkillsBench** - Skill composition and selection
- **WebClipper** - Trajectory pruning for skill optimization
- **Multi-Turn Attack Defense** - Adversarial skill protection

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from src.skill import SkillLibrary, Skill, SkillResult, SkillCategory

# Create a skill
class MySkill(Skill):
    def _create_metadata(self):
        return SkillMetadata(
            name="my_skill",
            description="My custom skill",
            category=SkillCategory.UTILITY
        )
    
    def execute(self, context):
        return SkillResult(success=True, output="done")

# Register and use
library = SkillLibrary()
library.register(MySkill())
```

## Testing

```bash
pytest tests/
```
