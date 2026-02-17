"""Demo: Simple web search task."""

from src.agent import AdaptiveWebAgent, AgentConfig
from src.browser_tool import BrowserTool


def run_simple_search_demo():
    """Run a simple search demo."""
    print("="*60)
    print("Simple Search Demo")
    print("="*60)
    
    # Create agent
    config = AgentConfig(
        max_steps=10,
        debug=True
    )
    agent = AdaptiveWebAgent(config)
    agent.initialize()
    
    # Create browser
    browser = BrowserTool()
    
    # Define task
    task = {
        "url": "https://www.google.com",
        "goal": "Search for 'artificial intelligence' and verify results appear",
        "browser": browser
    }
    
    print(f"\nTask: {task['goal']}")
    print(f"URL: {task['url']}")
    print("\nRunning...")
    
    # Run task
    result = agent.run_task(task)
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Success: {result.get('success', False)}")
    print(f"Steps: {result.get('steps', 0)}")
    print(f"Tool calls: {result.get('tool_calls', 0)}")
    print(f"Checklist completion: {result.get('checklist_completion', 0)*100:.1f}%")
    
    if result.get('error'):
        print(f"Error: {result.get('error')}")
    
    print("="*60)


def run_navigation_demo():
    """Run a navigation demo."""
    print("="*60)
    print("Navigation Demo")
    print("="*60)
    
    # Create agent
    config = AgentConfig(max_steps=5, debug=True)
    agent = AdaptiveWebAgent(config)
    agent.initialize()
    
    browser = BrowserTool()
    
    task = {
        "url": "https://en.wikipedia.org/wiki/Python_(programming_language)",
        "goal": "Navigate to Wikipedia Python page and extract first paragraph",
        "browser": browser
    }
    
    print(f"\nTask: {task['goal']}")
    print("\nRunning...")
    
    result = agent.run_task(task)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Success: {result.get('success', False)}")
    print(f"Steps: {result.get('steps', 0)}")
    print("="*60)


if __name__ == "__main__":
    print("\nWhich demo would you like to run?")
    print("1. Simple Search (Google)")
    print("2. Navigation (Wikipedia)")
    print("3. Both")
    
    choice = input("\n> ").strip()
    
    if choice == "1":
        run_simple_search_demo()
    elif choice == "2":
        run_navigation_demo()
    elif choice == "3":
        run_simple_search_demo()
        print("\n")
        run_navigation_demo()
    else:
        print("Running search demo by default...")
        run_simple_search_demo()
