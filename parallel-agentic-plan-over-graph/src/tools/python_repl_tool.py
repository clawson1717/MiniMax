"""PythonREPLTool - Simulated Python code execution for PAPoG.

This is a skeleton implementation that simulates code execution in a safe manner.
In a production system, this would use a proper sandboxed execution environment.
"""

from __future__ import annotations

from src.tools import ToolABC


class PythonREPLTool(ToolABC):
    """Simulated Python REPL tool.

    Executes Python code in a simulated sandbox environment. This skeleton
    provides safe mock execution without actually running user code.

    Parameters
    ----------
    name:
        Tool identifier. Defaults to "python_repl".
    description:
        Human-readable description. Defaults to a Python execution description.
    keywords:
        Keywords for task matching. Defaults to ["python", "code", "execute", "run", "compute", "calculate"].
    timeout:
        Simulated execution timeout in seconds. Defaults to 30.
    """

    def __init__(
        self,
        name: str = "python_repl",
        description: str = "Execute Python code in a sandboxed environment",
        keywords: list[str] | None = None,
        timeout: int = 30,
    ) -> None:
        self._name = name
        self._description = description
        self._keywords = keywords if keywords is not None else [
            "python",
            "code",
            "execute",
            "run",
            "compute",
            "calculate",
            "eval",
            "script",
        ]
        self._timeout = timeout

    @property
    def name(self) -> str:
        """Unique identifier for this tool."""
        return self._name

    @property
    def description(self) -> str:
        """Human-readable description of what this tool does."""
        return self._description

    @property
    def keywords(self) -> list[str]:
        """Keywords used to match this tool to task descriptions."""
        return list(self._keywords)

    @property
    def timeout(self) -> int:
        """Execution timeout in seconds."""
        return self._timeout

    def execute(self, input: str) -> str:
        """Execute Python code in a simulated sandbox.

        Parameters
        ----------
        input:
            Python code string to execute.

        Returns
        -------
        str
            Simulated execution output.
        """
        if not input or not input.strip():
            return "Error: Empty code input. Please provide Python code to execute."

        code = input.strip()

        # Check for potentially dangerous operations
        dangerous_patterns = [
            "import os",
            "import subprocess",
            "import sys",
            "open(",
            "__import__",
            "eval(",
            "exec(",
            "compile(",
        ]

        for pattern in dangerous_patterns:
            if pattern in code:
                return f"Security Error: Potentially dangerous operation detected: '{pattern}'. Sandbox execution blocked."

        # Simulate code execution
        return self._simulate_execution(code)

    def _simulate_execution(self, code: str) -> str:
        """Simulate Python code execution.

        Parameters
        ----------
        code:
            Python code string.

        Returns
        -------
        str
            Simulated output.
        """
        # Simple simulation - detect common patterns and return mock results
        lines = code.strip().split("\n")
        output_lines = []

        # Check for print statements and simulate output
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("print("):
                # Extract the print argument
                if '"' in stripped or "'" in stripped:
                    # String print
                    import re
                    match = re.search(r'print\(["\'](.+?)["\']\)', stripped)
                    if match:
                        output_lines.append(match.group(1))
                    else:
                        output_lines.append("[simulated output]")
                elif "range" in stripped:
                    match = re.search(r"range\((\d+)\)", stripped)
                    if match:
                        n = int(match.group(1))
                        output_lines.append(", ".join(str(i) for i in range(n)))
                    else:
                        output_lines.append("[range output]")
                else:
                    output_lines.append("[print output]")
            elif "=" in stripped and not stripped.startswith("#"):
                # Variable assignment
                output_lines.append(f"[executed: {stripped}]")
            elif stripped.startswith("def "):
                output_lines.append(f"[defined function: {stripped[4:20]}...]")
            elif stripped.startswith("for ") or stripped.startswith("while "):
                output_lines.append("[loop executed]")
            elif stripped.startswith("return "):
                output_lines.append(f"[returned: {stripped[7:]}]")

        if not output_lines:
            output_lines.append("[Code executed successfully - no output]")
        else:
            output_lines.append("")
            output_lines.append("[Simulated execution - sandbox mode]")

        return "\n".join(output_lines)

    def can_handle(self, task_description: str) -> bool:
        """Check if this tool can handle the given task.

        Parameters
        ----------
        task_description:
            The description of the task to check.

        Returns
        -------
        bool
            True if any keyword appears in the task description.
        """
        task_lower = task_description.lower()
        return any(kw.lower() in task_lower for kw in self._keywords)
