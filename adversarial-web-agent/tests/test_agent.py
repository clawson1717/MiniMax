import pytest
import asyncio
from src.agent import WebAgent

@pytest.mark.asyncio
async def test_agent_init():
    agent = WebAgent()
    assert agent.page is None
    await agent.start()
    assert agent.page is not None
    await agent.stop()

@pytest.mark.asyncio
async def test_agent_navigate():
    agent = WebAgent()
    await agent.start()
    await agent.navigate("https://example.com")
    title = await agent.page.title()
    assert title == "Example Domain"
    await agent.stop()

@pytest.mark.asyncio
async def test_agent_screenshot(tmp_path):
    agent = WebAgent()
    await agent.start()
    await agent.navigate("https://example.com")
    ss_path = tmp_path / "test.png"
    await agent.screenshot(str(ss_path))
    assert ss_path.exists()
    await agent.stop()
