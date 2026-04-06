#!/usr/bin/env python3
"""
Test Suite for Logger

Comprehensive tests for the VERITAS Logger component.
"""

import unittest
import sys
import os
import json
from datetime import datetime

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from veritas.logger import Logger, LogLevel, LogEntry


class TestLogger(unittest.TestCase):
    """Test logger functionality."""
    
    def test_logger_creation(self):
        """Test that logger can be created."""
        logger = Logger()
        self.assertIsNotNone(logger)
        self.assertEqual(logger.min_level, "INFO")
        self.assertEqual(len(logger.entries), 0)
    
    def test_logger_custom_level(self):
        """Test logger with custom minimum level."""
        logger = Logger(min_level="DEBUG")
        self.assertEqual(logger.min_level, "DEBUG")
        self.assertEqual(len(logger.entries), 0)
    
    def test_logging_methods(self):
        """Test all logging methods."""
        logger = Logger(console_output=False, log_file=None, min_level="DEBUG")
        
        # Log messages at all levels
        logger.debug("Test debug message", module="test", custom_field="value1")
        logger.info("Test info message", module="test", custom_field="value2")
        logger.warning("Test warning message", module="test", custom_field="value3")
        logger.error("Test error message", module="test", custom_field="value4")
        logger.critical("Test critical message", module="test", custom_field="value5")
        
        self.assertEqual(len(logger.entries), 5)
        
        # Check levels
        levels = [entry.level for entry in logger.entries]
        self.assertEqual(levels, ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        
        # Check messages
        messages = [entry.message for entry in logger.entries]
        expected_messages = [
            "Test debug message",
            "Test info message",
            "Test warning message",
            "Test error message",
            "Test critical message"
        ]
        self.assertEqual(messages, expected_messages)
        
        # Check metadata
        for i, entry in enumerate(logger.entries):
            self.assertEqual(entry.module, "test")
            self.assertIn("custom_field", entry.metadata)
            expected_values = ["value1", "value2", "value3", "value4", "value5"]
            self.assertEqual(entry.metadata["custom_field"], expected_values[i])
    
    def test_log_entry_structure(self):
        """Test that log entries have proper structure."""
        logger = Logger(console_output=False, log_file=None)
        timestamp = datetime.now().timestamp()
        logger.info("Test message", module="test_module", custom_field="value")
        
        entry = logger.entries[0]
        self.assertIsInstance(entry, LogEntry)
        self.assertEqual(entry.level, "INFO")
        self.assertEqual(entry.message, "Test message")
        self.assertEqual(entry.module, "test_module")
        self.assertIn("custom_field", entry.metadata)
        self.assertEqual(entry.metadata["custom_field"], "value")
        self.assertTrue(isinstance(entry.timestamp, float))
        self.assertTrue(entry.timestamp >= timestamp)
    
    def test_get_logs(self):
        """Test getting logs with filtering."""
        logger = Logger(console_output=False, log_file=None)
        
        # Log messages at different levels
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.debug("Debug message")  # This should be included since min_level=DEBUG
        
        # Get all logs
        all_logs = logger.get_logs()
        self.assertEqual(len(all_logs), 4)
        
        # Filter by level
        warning_logs = logger.get_logs(level="WARNING")
        self.assertEqual(len(warning_logs), 1)
        self.assertEqual(warning_logs[0]["level"], "WARNING")
        self.assertEqual(warning_logs[0]["message"], "Warning message")
        
        # Filter by module
        info_logs = logger.get_logs(module="nonexistent")
        self.assertEqual(len(info_logs), 0)
        
        # Filter by message content
        error_logs = logger.get_logs(message_contains="Error")
        self.assertEqual(len(error_logs), 1)
        self.assertEqual(error_logs[0]["level"], "ERROR")
        
        # Get logs with metadata
        logs_with_meta = logger.get_logs(include_metadata=True)
        self.assertTrue(all("metadata" in log for log in logs_with_meta))
    
    def test_clear_logs(self):
        """Test clearing logs."""
        logger = Logger(console_output=False, log_file=None)
        logger.info("Test message")
        self.assertEqual(len(logger.entries), 1)
        
        logger.clear_logs()
        self.assertEqual(len(logger.entries), 0)
        
        # Verify that new logs can be added after clearing
        logger.warning("Another message")
        self.assertEqual(len(logger.entries), 1)
    
    def test_log_level_filtering(self):
        """Test that log level filtering works correctly."""
        logger = Logger(min_level="WARNING")
        
        logger.debug("Debug message")  # Should not be logged
        logger.info("Info message")    # Should not be logged
        logger.warning("Warning message")  # Should be logged
        logger.error("Error message")      # Should be logged
        logger.critical("Critical message")  # Should be logged
        
        self.assertEqual(len(logger.entries), 3)
        levels = [entry.level for entry in logger.entries]
        self.assertEqual(levels, ["WARNING", "ERROR", "CRITICAL"])
    
    def test_log_to_file(self):
        """Test logging to a file."""
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "test.log")
            logger = Logger(log_file=log_file, console_output=False)
            
            logger.info("Test message")
            logger.warning("Another message")
            
            # Check that file was created and contains logs
            self.assertTrue(os.path.exists(log_file))
            
            with open(log_file, 'r') as f:
                content = f.read()
                self.assertIn("INFO: Test message", content)
                self.assertIn("WARNING: Another message", content)
    
    def test_console_output(self):
        """Test console output (captured to check it works)."""
        import io
        import sys
        
        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        try:
            logger = Logger(console_output=True, log_file=None, min_level="INFO")
            logger.info("Console message")
            logger.warning("Another console message")
            
            # Get captured output
            output = captured_output.getvalue()
            self.assertIn("INFO: Console message", output)
            self.assertIn("WARNING: Another console message", output)
        finally:
            # Restore stdout
            sys.stdout = sys.__stdout__
    
    def test_log_entry_json_serialization(self):
        """Test that log entries can be serialized to JSON."""
        logger = Logger(console_output=False, log_file=None)
        logger.info("Test message", module="test", custom_field="value")
        
        entry = logger.entries[0]
        entry_dict = entry.to_dict()
        
        self.assertIn("level", entry_dict)
        self.assertIn("message", entry_dict)
        self.assertIn("timestamp", entry_dict)
        self.assertIn("module", entry_dict)
        self.assertIn("metadata", entry_dict)
        
        # Test JSON serialization
        json_str = json.dumps(entry_dict)
        self.assertTrue(isinstance(json_str, str))
        
        # Parse back and verify
        parsed = json.loads(json_str)
        self.assertEqual(parsed["level"], "INFO")
        self.assertEqual(parsed["message"], "Test message")
    
    def test_get_logs_with_timestamp_filter(self):
        """Test getting logs with timestamp filtering."""
        logger = Logger(console_output=False, log_file=None)
        
        # Log messages at different times (simulate by setting timestamps)
        now = datetime.now().timestamp()
        
        logger.info("Old message", timestamp=now - 3600)  # 1 hour ago
        logger.warning("Recent message", timestamp=now - 300)  # 5 minutes ago
        logger.error("New message", timestamp=now - 60)  # 1 minute ago
        
        # Get logs from last hour
        recent_logs = logger.get_logs(since_timestamp=now - 3600)
        self.assertEqual(len(recent_logs), 3)  # All three are within last hour
        
        # Get logs from last 10 minutes
        very_recent_logs = logger.get_logs(since_timestamp=now - 600)
        self.assertEqual(len(very_recent_logs), 2)  # Only warning and error
        
        # Get logs from last minute
        latest_logs = logger.get_logs(since_timestamp=now - 60)
        self.assertEqual(len(latest_logs), 1)  # Only error


class TestLogLevel(unittest.TestCase):
    """Test LogLevel enum."""
    
    def test_log_level_values(self):
        """Test that log level values are correctly defined."""
        levels = list(LogLevel)
        self.assertEqual(len(levels), 5)
        self.assertIn(LogLevel.DEBUG, levels)
        self.assertIn(LogLevel.INFO, levels)
        self.assertIn(LogLevel.WARNING, levels)
        self.assertIn(LogLevel.ERROR, levels)
        self.assertIn(LogLevel.CRITICAL, levels)
        
        # Test value mapping
        self.assertEqual(LogLevel.DEBUG.value, 10)
        self.assertEqual(LogLevel.INFO.value, 20)
        self.assertEqual(LogLevel.WARNING.value, 30)
        self.assertEqual(LogLevel.ERROR.value, 40)
        self.assertEqual(LogLevel.CRITICAL.value, 50)
    
    def test_log_level_from_string(self):
        """Test converting strings to LogLevel."""
        level_map = {
            "DEBUG": LogLevel.DEBUG,
            "INFO": LogLevel.INFO,
            "WARNING": LogLevel.WARNING,
            "ERROR": LogLevel.ERROR,
            "CRITICAL": LogLevel.CRITICAL
        }
        
        for level_str, expected_level in level_map.items():
            level = LogLevel[level_str]
            self.assertEqual(level, expected_level)


if __name__ == '__main__':
    # Create test directory if it doesn't exist
    test_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    # Run tests
    unittest.main(verbosity=2)