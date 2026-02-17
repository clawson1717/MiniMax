"""Tests for browser tool module."""

import pytest
from src.browser_tool import (
    BrowserTool, BrowserAction, ActionResult, ActionType
)


class TestBrowserToolInitialization:
    """Tests for browser tool initialization."""
    
    def test_default_initialization(self):
        """Test browser tool initializes with defaults."""
        tool = BrowserTool()
        
        assert tool.headless is True
        assert tool.default_timeout == 30
        assert tool._page is None
        assert tool._browser is None
    
    def test_custom_initialization(self):
        """Test browser tool with custom config."""
        tool = BrowserTool(headless=False, timeout=60)
        
        assert tool.headless is False
        assert tool.default_timeout == 60
    
    def test_empty_history(self):
        """Test initial history is empty."""
        tool = BrowserTool()
        assert tool.get_history() == []


class TestBrowserAction:
    """Tests for BrowserAction dataclass."""
    
    def test_navigate_action(self):
        """Test navigate action creation."""
        action = BrowserAction(
            action_type=ActionType.NAVIGATE,
            url="https://example.com"
        )
        
        assert action.action_type == ActionType.NAVIGATE
        assert action.url == "https://example.com"
        assert action.selector is None
    
    def test_click_action(self):
        """Test click action creation."""
        action = BrowserAction(
            action_type=ActionType.CLICK,
            selector="#submit-btn"
        )
        
        assert action.action_type == ActionType.CLICK
        assert action.selector == "#submit-btn"
    
    def test_type_action(self):
        """Test type action creation."""
        action = BrowserAction(
            action_type=ActionType.TYPE,
            selector="#search",
            text="query"
        )
        
        assert action.action_type == ActionType.TYPE
        assert action.selector == "#search"
        assert action.text == "query"
    
    def test_action_with_timeout(self):
        """Test action with custom timeout."""
        action = BrowserAction(
            action_type=ActionType.WAIT,
            selector=".loading",
            timeout=60
        )
        
        assert action.timeout == 60


class TestActionResult:
    """Tests for ActionResult dataclass."""
    
    def test_successful_result(self):
        """Test successful action result."""
        action = BrowserAction(ActionType.NAVIGATE, url="https://example.com")
        result = ActionResult(
            success=True,
            action=action,
            observation="Navigated"
        )
        
        assert result.success is True
        assert result.observation == "Navigated"
        assert result.error is None
    
    def test_failed_result(self):
        """Test failed action result."""
        action = BrowserAction(ActionType.CLICK, selector="#missing")
        result = ActionResult(
            success=False,
            action=action,
            error="Element not found"
        )
        
        assert result.success is False
        assert result.error == "Element not found"


class TestBrowserToolNavigate:
    """Tests for navigate method."""
    
    def test_navigate_success(self):
        """Test successful navigation."""
        tool = BrowserTool()
        result = tool.navigate("https://example.com")
        
        assert result.success is True
        assert "example.com" in result.observation
        assert result.action.action_type == ActionType.NAVIGATE
    
    def test_navigate_with_timeout(self):
        """Test navigation with custom timeout."""
        tool = BrowserTool()
        result = tool.navigate("https://example.com", timeout=60)
        
        assert result.success is True
        assert result.action.timeout == 60


class TestBrowserToolClick:
    """Tests for click method."""
    
    def test_click_success(self):
        """Test successful click."""
        tool = BrowserTool()
        result = tool.click("#button")
        
        assert result.success is True
        assert "#button" in result.observation
    
    def test_click_with_selector(self):
        """Test click with various selectors."""
        tool = BrowserTool()
        
        result = tool.click(".class-selector")
        assert result.success is True
        
        result = tool.click("div[data-test='test']")
        assert result.success is True


class TestBrowserToolType:
    """Tests for type method."""
    
    def test_type_success(self):
        """Test successful type."""
        tool = BrowserTool()
        result = tool.type("#input", "hello world")
        
        assert result.success is True
        assert "hello world" in result.observation
    
    def test_type_clear_first(self):
        """Test type with clear_first."""
        tool = BrowserTool()
        
        # With clear
        result = tool.type("#input", "text", clear_first=True)
        assert result.success is True
        
        # Without clear
        result = tool.type("#input", "text", clear_first=False)
        assert result.success is True


class TestBrowserToolExtract:
    """Tests for extract method."""
    
    def test_extract_success(self):
        """Test successful extraction."""
        tool = BrowserTool()
        result = tool.extract(".content")
        
        assert result.success is True
        assert result.observation is not None
    
    def test_extract_with_attribute(self):
        """Test extraction with attribute."""
        tool = BrowserTool()
        result = tool.extract("a", attribute="href")
        
        assert result.success is True


class TestBrowserToolScreenshot:
    """Tests for screenshot method."""
    
    def test_screenshot_success(self):
        """Test successful screenshot."""
        tool = BrowserTool()
        result = tool.screenshot()
        
        assert result.success is True
        assert "screenshot" in result.observation.lower()


class TestBrowserToolWait:
    """Tests for wait method."""
    
    def test_wait_success(self):
        """Test successful wait."""
        tool = BrowserTool()
        result = tool.wait_for(".element")
        
        assert result.success is True
        assert ".element" in result.observation
    
    def test_wait_with_timeout(self):
        """Test wait with custom timeout."""
        tool = BrowserTool()
        result = tool.wait_for(".element", timeout=120)
        
        assert result.success is True
        assert result.action.timeout == 120


class TestBrowserToolPageState:
    """Tests for get_page_state method."""
    
    def test_get_page_state(self):
        """Test getting page state."""
        tool = BrowserTool()
        state = tool.get_page_state()
        
        assert "url" in state
        assert "title" in state
        assert state["url"] == "about:blank"
    
    def test_page_state_includes_elements(self):
        """Test page state includes interactive elements."""
        tool = BrowserTool()
        state = tool.get_page_state()
        
        assert "interactive_elements" in state


class TestBrowserToolExecute:
    """Tests for execute method."""
    
    def test_execute_navigate(self):
        """Test executing navigate action."""
        tool = BrowserTool()
        action = BrowserAction(action_type=ActionType.NAVIGATE, url="https://test.com")
        result = tool.execute(action)
        
        assert result.success is True
        assert len(tool.get_history()) == 1
    
    def test_execute_click(self):
        """Test executing click action."""
        tool = BrowserTool()
        action = BrowserAction(action_type=ActionType.CLICK, selector="#btn")
        result = tool.execute(action)
        
        assert result.success is True
    
    def test_execute_unknown_action(self):
        """Test executing unknown action type."""
        tool = BrowserTool()
        
        # Create action with unknown type via direct instantiation
        action = BrowserAction(action_type=ActionType.SCROLL)
        result = tool.execute(action)
        
        # Scroll is not in action_map, should fail
        assert result.success is False
        assert "Unknown action type" in result.error


class TestBrowserToolValidate:
    """Tests for validate_selector method."""
    
    def test_validate_selector_mock(self):
        """Test selector validation with mock page."""
        tool = BrowserTool()
        
        # Mock page always returns True
        assert tool.validate_selector("#anything") is True


class TestBrowserToolHistory:
    """Tests for action history management."""
    
    def test_history_tracking(self):
        """Test actions are tracked in history when using execute()."""
        tool = BrowserTool()
        
        tool.execute(BrowserAction(action_type=ActionType.NAVIGATE, url="https://example.com"))
        tool.execute(BrowserAction(action_type=ActionType.CLICK, selector="#button"))
        
        history = tool.get_history()
        assert len(history) == 2
    
    def test_clear_history(self):
        """Test clearing history."""
        tool = BrowserTool()
        
        tool.navigate("https://example.com")
        tool.clear_history()
        
        assert len(tool.get_history()) == 0


class TestBrowserToolClose:
    """Tests for close method."""
    
    def test_close_no_error(self):
        """Test closing tool without error."""
        tool = BrowserTool()
        
        # Should not raise
        tool.close()
        
        # Should be clean
        assert tool._page is None
        assert tool._browser is None
    
    def test_close_twice(self):
        """Test closing twice is safe."""
        tool = BrowserTool()
        
        tool.close()
        tool.close()  # Should not raise


class TestActionType:
    """Tests for ActionType enum."""
    
    def test_all_action_types(self):
        """Test all action types are defined."""
        expected_types = [
            "navigate", "click", "type", "extract", 
            "screenshot", "wait", "scroll", "select",
            "hover", "go_back", "go_forward", "refresh"
        ]
        
        actual = [a.value for a in ActionType]
        
        for expected in expected_types:
            assert expected in actual
