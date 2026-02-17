"""Browser tool integration for web automation."""

from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
from enum import Enum
import time
import re


class ActionType(Enum):
    """Supported browser action types."""
    NAVIGATE = "navigate"
    CLICK = "click"
    TYPE = "type"
    EXTRACT = "extract"
    SCREENSHOT = "screenshot"
    WAIT = "wait"
    SCROLL = "scroll"
    SELECT = "select"
    HOVER = "hover"
    GO_BACK = "go_back"
    GO_FORWARD = "go_forward"
    REFRESH = "refresh"


@dataclass
class BrowserAction:
    """Represents a browser action to execute."""
    action_type: ActionType
    selector: Optional[str] = None
    text: Optional[str] = None
    url: Optional[str] = None
    timeout: int = 30
    wait_for: Optional[str] = None  # CSS selector to wait for
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ActionResult:
    """Result of a browser action."""
    success: bool
    action: BrowserAction
    observation: str = ""
    screenshot: Optional[bytes] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BrowserTool:
    """
    Browser automation tool for web agent.
    
    Provides safe execution of common browser operations with:
    - Action validation
    - Timeout handling
    - Error recovery
    - Screenshot capture
    """
    
    def __init__(self, headless: bool = True, timeout: int = 30):
        """Initialize browser tool.
        
        Args:
            headless: Run browser in headless mode.
            timeout: Default timeout for actions in seconds.
        """
        self.headless = headless
        self.default_timeout = timeout
        self._page = None
        self._browser = None
        self._action_history: List[ActionResult] = []
        
    def connect(self, page=None, browser=None) -> None:
        """Connect to existing browser page.
        
        Args:
            page: Playwright page object or similar.
            browser: Browser instance.
        """
        self._page = page
        self._browser = browser
        
    def navigate(self, url: str, timeout: Optional[int] = None) -> ActionResult:
        """Navigate to a URL.
        
        Args:
            url: Target URL.
            timeout: Optional timeout override.
            
        Returns:
            ActionResult with navigation result.
        """
        start_time = time.time()
        action = BrowserAction(
            action_type=ActionType.NAVIGATE,
            url=url,
            timeout=timeout or self.default_timeout
        )
        
        try:
            if self._page is None:
                # Mock result for testing
                return ActionResult(
                    success=True,
                    action=action,
                    observation=f"Navigated to {url}",
                    execution_time=time.time() - start_time
                )
            
            # Real implementation would use Playwright/Selenium
            response = self._page.goto(url, timeout=action.timeout * 1000)
            
            return ActionResult(
                success=response is not None,
                action=action,
                observation=f"Navigated to {url}",
                metadata={"status": response.status if response else None},
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return ActionResult(
                success=False,
                action=action,
                error=str(e),
                execution_time=time.time() - start_time
            )
    
    def click(self, selector: str, timeout: Optional[int] = None) -> ActionResult:
        """Click an element.
        
        Args:
            selector: CSS selector for element.
            timeout: Optional timeout override.
            
        Returns:
            ActionResult with click result.
        """
        start_time = time.time()
        action = BrowserAction(
            action_type=ActionType.CLICK,
            selector=selector,
            timeout=timeout or self.default_timeout
        )
        
        try:
            if self._page is None:
                return ActionResult(
                    success=True,
                    action=action,
                    observation=f"Clicked element: {selector}",
                    execution_time=time.time() - start_time
                )
            
            self._page.click(selector, timeout=action.timeout * 1000)
            
            return ActionResult(
                success=True,
                action=action,
                observation=f"Clicked element: {selector}",
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return ActionResult(
                success=False,
                action=action,
                error=str(e),
                execution_time=time.time() - start_time
            )
    
    def type(self, selector: str, text: str, 
             clear_first: bool = True,
             timeout: Optional[int] = None) -> ActionResult:
        """Type text into an element.
        
        Args:
            selector: CSS selector for element.
            text: Text to type.
            clear_first: Clear existing text before typing.
            timeout: Optional timeout override.
            
        Returns:
            ActionResult with type result.
        """
        start_time = time.time()
        action = BrowserAction(
            action_type=ActionType.TYPE,
            selector=selector,
            text=text,
            timeout=timeout or self.default_timeout
        )
        
        try:
            if self._page is None:
                return ActionResult(
                    success=True,
                    action=action,
                    observation=f"Typed '{text}' into: {selector}",
                    execution_time=time.time() - start_time
                )
            
            if clear_first:
                self._page.fill(selector, "", timeout=action.timeout * 1000)
            self._page.fill(selector, text, timeout=action.timeout * 1000)
            
            return ActionResult(
                success=True,
                action=action,
                observation=f"Typed '{text}' into: {selector}",
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return ActionResult(
                success=False,
                action=action,
                error=str(e),
                execution_time=time.time() - start_time
            )
    
    def extract(self, selector: str, 
                attribute: Optional[str] = None,
                timeout: Optional[int] = None) -> ActionResult:
        """Extract content from page.
        
        Args:
            selector: CSS selector for element.
            attribute: Optional attribute to extract (textContent, href, src, etc.).
            timeout: Optional timeout override.
            
        Returns:
            ActionResult with extracted content.
        """
        start_time = time.time()
        action = BrowserAction(
            action_type=ActionType.EXTRACT,
            selector=selector,
            timeout=timeout or self.default_timeout
        )
        
        try:
            if self._page is None:
                return ActionResult(
                    success=True,
                    action=action,
                    observation=f"Mock extracted content from: {selector}",
                    execution_time=time.time() - start_time
                )
            
            if attribute and attribute != "textContent":
                content = self._page.get_attribute(selector, attribute, timeout=action.timeout * 1000)
            else:
                content = self._page.text_content(selector, timeout=action.timeout * 1000)
            
            return ActionResult(
                success=content is not None,
                action=action,
                observation=str(content) if content else "",
                metadata={"content": content},
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return ActionResult(
                success=False,
                action=action,
                error=str(e),
                execution_time=time.time() - start_time
            )
    
    def screenshot(self, path: Optional[str] = None) -> ActionResult:
        """Take a screenshot.
        
        Args:
            path: Optional path to save screenshot.
            
        Returns:
            ActionResult with screenshot data.
        """
        start_time = time.time()
        action = BrowserAction(action_type=ActionType.SCREENSHOT)
        
        try:
            if self._page is None:
                return ActionResult(
                    success=True,
                    action=action,
                    observation="Mock screenshot captured",
                    execution_time=time.time() - start_time
                )
            
            screenshot_bytes = self._page.screenshot()
            
            if path:
                with open(path, 'wb') as f:
                    f.write(screenshot_bytes)
            
            return ActionResult(
                success=True,
                action=action,
                observation="Screenshot captured",
                screenshot=screenshot_bytes,
                metadata={"size": len(screenshot_bytes)},
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return ActionResult(
                success=False,
                action=action,
                error=str(e),
                execution_time=time.time() - start_time
            )
    
    def wait_for(self, selector: str, timeout: Optional[int] = None) -> ActionResult:
        """Wait for an element to appear.
        
        Args:
            selector: CSS selector to wait for.
            timeout: Optional timeout override.
            
        Returns:
            ActionResult with wait result.
        """
        start_time = time.time()
        action = BrowserAction(
            action_type=ActionType.WAIT,
            selector=selector,
            wait_for=selector,
            timeout=timeout or self.default_timeout
        )
        
        try:
            if self._page is None:
                return ActionResult(
                    success=True,
                    action=action,
                    observation=f"Waited for: {selector}",
                    execution_time=time.time() - start_time
                )
            
            self._page.wait_for_selector(selector, timeout=action.timeout * 1000)
            
            return ActionResult(
                success=True,
                action=action,
                observation=f"Element appeared: {selector}",
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return ActionResult(
                success=False,
                action=action,
                error=str(e),
                execution_time=time.time() - start_time
            )
    
    def get_page_state(self) -> Dict[str, Any]:
        """Get structured representation of current page state.
        
        Returns:
            Dictionary with page state information.
        """
        if self._page is None:
            return {
                "url": "about:blank",
                "title": "Mock Page",
                "dom_state": {},
                "interactive_elements": []
            }
        
        try:
            # Get basic page info
            url = self._page.url
            title = self._page.title()
            
            # Extract interactive elements
            interactive_selectors = [
                "a", "button", "input", "select", "textarea",
                "[role='button']", "[role='textbox']", "[href]"
            ]
            
            elements = []
            for sel in interactive_selectors:
                try:
                    count = self._page.locator(sel).count()
                    if count > 0:
                        elements.append({
                            "selector": sel,
                            "count": count
                        })
                except:
                    pass
            
            return {
                "url": url,
                "title": title,
                "interactive_elements": elements,
                "timestamp": time.time()
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "url": "unknown"
            }
    
    def execute(self, action: BrowserAction) -> ActionResult:
        """Execute a browser action.
        
        Args:
            action: BrowserAction to execute.
            
        Returns:
            ActionResult from executed action.
        """
        action_map = {
            ActionType.NAVIGATE: lambda: self.navigate(action.url, action.timeout),
            ActionType.CLICK: lambda: self.click(action.selector, action.timeout),
            ActionType.TYPE: lambda: self.type(action.selector, action.text, timeout=action.timeout),
            ActionType.EXTRACT: lambda: self.extract(action.selector, timeout=action.timeout),
            ActionType.SCREENSHOT: lambda: self.screenshot(),
            ActionType.WAIT: lambda: self.wait_for(action.selector, action.timeout),
        }
        
        handler = action_map.get(action.action_type)
        if handler:
            result = handler()
            self._action_history.append(result)
            return result
        
        return ActionResult(
            success=False,
            action=action,
            error=f"Unknown action type: {action.action_type}"
        )
    
    def validate_selector(self, selector: str) -> bool:
        """Validate if a selector exists on the page.
        
        Args:
            selector: CSS selector to validate.
            
        Returns:
            True if selector matches at least one element.
        """
        if self._page is None:
            return True  # Mock always valid
            
        try:
            count = self._page.locator(selector).count()
            return count > 0
        except:
            return False
    
    def get_history(self) -> List[ActionResult]:
        """Get action execution history.
        
        Returns:
            List of ActionResults.
        """
        return self._action_history.copy()
    
    def clear_history(self) -> None:
        """Clear action history."""
        self._action_history = []
    
    def close(self) -> None:
        """Close browser and cleanup resources."""
        if self._page:
            try:
                self._page.close()
            except:
                pass
            self._page = None
            
        if self._browser:
            try:
                self._browser.close()
            except:
                pass
            self._browser = None
