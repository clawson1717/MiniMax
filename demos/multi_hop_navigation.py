"""Demo: Multi-hop navigation task."""

from src.agent import AdaptiveWebAgent, AgentConfig
from src.browser_tool import BrowserTool


def run_multi_hop_demo():
    """Run a multi-hop navigation demo."""
    print("="*60)
    print("Multi-Hop Navigation Demo")
    print("="*60)
    print("This demo finds information requiring multiple steps:")
    "1. Navigate to Google"
    "2. Search for a topic"
    "3. Click on a result"
    "4. Extract information from the target page"
    
    # Create agent
    config = AgentConfig(
        max_steps=20,
        uncertainty_threshold=0.6,
        debug=True
    )
    agent = AdaptiveWebAgent(config)
    agent.initialize()
    
    browser = BrowserTool()
    
    task = {
        "url": "https://www.google.com",
        "goal": "Find who is the CEO of Google and extract their name",
        "browser": browser
    }
    
    print(f"\nTask: {task['goal']}")
    print("\nRunning... (this may take a while)")
    
    result = agent.run_task(task)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Success: {result.get('success', False)}")
    print(f"Steps: {result.get('steps', 0)}")
    print(f"Tool calls: {result.get('tool_calls', 0)}")
    print(f"Checklist completion: {result.get('checklist_completion', 0)*100:.1f}%")
    print(f"Average uncertainty: {result.get('uncertainty_avg', 0)*100:.1f}%")
    print("="*60)
    
    # Show trajectory
    if result.get('trajectory'):
        print("\nTrajectory:")
        for i, step in enumerate(result['trajectory'][:5]):
            print(f"  {i+1}. {step.get('action', 'unknown')}")


def run_form_filling_demo():
    """Run a form filling demo."""
    print("="*60)
    print("Form Filling Demo")
    print("="*60)
    
    config = AgentConfig(max_steps=15, debug=True)
    agent = AdaptiveWebAgent(config)
    agent.initialize()
    
    browser = BrowserTool()
    
    task = {
        "url": "https://httpbin.org/forms/post",
        "goal": "Fill in the contact form with name and email, then submit",
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
    print("\nSelect a demo:")
    print("1. Multi-hop (Find Google CEO)")
    print("2. Form Filling")
    
    choice = input("\n> ").strip()
    
    if choice == "1":
        run_multi_hop_demo()
    elif choice == "2":
        run_form_filling_demo()
    else:
        print("Running multi-hop demo...")
        run_multi_hop_demo()
