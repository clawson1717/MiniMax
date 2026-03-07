import asyncio
from playwright.async_api import async_playwright

class WebAgent:
    def __init__(self):
        self.pw = None
        self.browser = None
        self.context = None
        self.page = None

    async def start(self):
        self.pw = await async_playwright().start()
        self.browser = await self.pw.chromium.launch(headless=True)
        self.context = await self.browser.new_context()
        self.page = await self.context.new_page()

    async def stop(self):
        if self.browser:
            await self.browser.close()
        if self.pw:
            await self.pw.stop()

    async def navigate(self, url: str):
        if not self.page:
            await self.start()
        await self.page.goto(url)

    async def click(self, selector: str):
        await self.page.click(selector)

    async def type(self, selector: str, text: str):
        await self.page.fill(selector, text)

    async def screenshot(self, path: str):
        await self.page.screenshot(path=path)
