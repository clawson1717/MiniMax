from setuptools import setup, find_packages

setup(
    name="sntr",
    version="0.1.0",
    description="Self-Healing via Neuro-Symbolic Trajectory Repair",
    author="OpenClaw",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pydantic>=2.0.0",
        "sentence-transformers>=2.2.0",
        "faiss-cpu>=1.7.0",
        "httpx>=0.24.0",
        "structlog>=23.0.0",
    ],
    extras_require={
        "isabelle": ["docker>=6.0.0"],
        "dev": ["pytest>=7.0.0", "pytest-asyncio>=0.21.0", "ruff>=0.1.0"],
    },
    entry_points={
        "console_scripts": [
            "sntr=sntr.cli:main",
        ],
    },
)
