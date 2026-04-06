from __future__ import annotations
from typing import Any, Dict, List, Optional
from datetime import datetime
import json
import os

class LogLevel:
    """Log level constants."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class LogEntry:
    """Represents a single log entry."""
    def __init__(
        self,
        level: str,
        message: str,
        module: str,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.level = level
        self.message = message
        self.module = module
        self.timestamp = timestamp or datetime.now()
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert log entry to dictionary."""
        return {
            "level": self.level,
            "message": self.message,
            "module": self.module,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }

class Logger:
    """Simple logging system for VERITAS."""
    
    def __init__(
        self,
        log_file: Optional[str] = None,
        min_level: str = LogLevel.INFO,
        console_output: bool = True
    ):
        """
        Initialize the Logger.
        
        Args:
            log_file: Optional file path to write logs to
            min_level: Minimum log level to output
            console_output: Whether to output to console
        """
        self.log_file = log_file
        self.min_level = min_level
        self.console_output = console_output
        self.entries: List[LogEntry] = []
        self._setup_log_file()
    
    def _setup_log_file(self) -> None:
        """Setup log file if specified."""
        if self.log_file:
            log_dir = os.path.dirname(os.path.abspath(self.log_file))
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
    
    def _log(self, level: str, message: str, module: str = "core", **kwargs) -> None:
        """Internal logging method."""
        if self._should_log(level):
            entry = LogEntry(level, message, module, metadata=kwargs)
            self.entries.append(entry)
            
            # Output to console if enabled
            if self.console_output:
                timestamp = entry.timestamp.strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{timestamp}] [{level}] {module}: {message}")
            
            # Write to file if specified
            if self.log_file:
                with open(self.log_file, 'a') as f:
                    f.write(json.dumps(entry.to_dict()) + "\n")
    
    def _should_log(self, level: str) -> bool:
        """Check if the log level should be output."""
        levels = [
            LogLevel.DEBUG,
            LogLevel.INFO,
            LogLevel.WARNING,
            LogLevel.ERROR,
            LogLevel.CRITICAL
        ]
        return levels.index(level) >= levels.index(self.min_level)
    
    def debug(self, message: str, module: str = "core", **kwargs) -> None:
        """Log a debug message."""
        self._log(LogLevel.DEBUG, message, module, **kwargs)
    
    def info(self, message: str, module: str = "core", **kwargs) -> None:
        """Log an informational message."""
        self._log(LogLevel.INFO, message, module, **kwargs)
    
    def warning(self, message: str, module: str = "core", **kwargs) -> None:
        """Log a warning message."""
        self._log(LogLevel.WARNING, message, module, **kwargs)
    
    def error(self, message: str, module: str = "core", **kwargs) -> None:
        """Log an error message."""
        self._log(LogLevel.ERROR, message, module, **kwargs)
    
    def critical(self, message: str, module: str = "core", **kwargs) -> None:
        """Log a critical error message."""
        self._log(LogLevel.CRITICAL, message, module, **kwargs)
    
    def get_logs(self, level: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all log entries as dictionaries.
        
        Args:
            level: Optional level to filter by
            
        Returns:
            List of log entries as dictionaries
        """
        if level:
            return [
                entry.to_dict() 
                for entry in self.entries 
                if self._should_log(level) and entry.level == level
            ]
        return [entry.to_dict() for entry in self.entries]
    
    def clear_logs(self) -> None:
        """Clear all log entries."""
        self.entries = []
        if self.log_file and os.path.exists(self.log_file):
            os.remove(self.log_file)

# Convenience function for easy initialization
def create_logger(**kwargs) -> Logger:
    """Create and return a Logger instance."""
    return Logger(**kwargs)