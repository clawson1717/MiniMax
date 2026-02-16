/**
 * Unit tests for Agent class
 * Tests constructor, action methods, state tracking, and error handling
 */

const { Agent, MockBrowser } = require('../src/Agent');
const { ActionTypes, ActionFactory } = require('../src/Action');

describe('Agent', () => {
  let agent;
  let mockBrowser;

  beforeEach(() => {
    mockBrowser = new MockBrowser();
    agent = new Agent({
      name: 'TestAgent',
      maxSteps: 10,
      browser: mockBrowser,
      verbose: false
    });
  });

  afterEach(() => {
    agent = null;
    mockBrowser = null;
  });

  describe('Constructor', () => {
    test('should create agent with default config', () => {
      const defaultAgent = new Agent();
      expect(defaultAgent.name).toBe('Agent');
      expect(defaultAgent.maxSteps).toBe(100);
      expect(defaultAgent.verbose).toBe(false);
      expect(defaultAgent.stepCount).toBe(0);
      expect(defaultAgent.currentUrl).toBeNull();
    });

    test('should create agent with custom config', () => {
      expect(agent.name).toBe('TestAgent');
      expect(agent.maxSteps).toBe(10);
      expect(agent.browser).toBe(mockBrowser);
      expect(agent.verbose).toBe(false);
    });

    test('should initialize with empty state', () => {
      expect(agent.history).toEqual([]);
      expect(agent.errors).toEqual([]);
      expect(agent.stepCount).toBe(0);
      expect(agent.isRunning).toBe(false);
      expect(agent.lastAction).toBeNull();
      expect(agent.lastResult).toBeNull();
    });
  });

  describe('Navigate', () => {
    test('should navigate to URL and update state', async () => {
      const url = 'https://example.com';
      const result = await agent.navigate(url);

      expect(result.success).toBe(true);
      expect(result.url).toBe(url);
      expect(agent.currentUrl).toBe(url);
      expect(agent.stepCount).toBe(1);
    });

    test('should record navigation in history', async () => {
      await agent.navigate('https://example.com');
      
      expect(agent.history).toHaveLength(1);
      expect(agent.history[0].action.type).toBe(ActionTypes.NAVIGATE);
      expect(agent.history[0].action.params.url).toBe('https://example.com');
      expect(agent.history[0].step).toBe(1);
    });

    test('should track last action and result', async () => {
      await agent.navigate('https://example.com');
      
      expect(agent.lastAction).not.toBeNull();
      expect(agent.lastAction.type).toBe(ActionTypes.NAVIGATE);
      expect(agent.lastResult).toEqual({ success: true, url: 'https://example.com' });
    });
  });

  describe('Click', () => {
    test('should click element and return result', async () => {
      await agent.navigate('https://example.com');
      const result = await agent.click('#submit-button');

      expect(result.success).toBe(true);
      expect(result.selector).toBe('#submit-button');
      expect(agent.stepCount).toBe(2);
    });

    test('should record click in history', async () => {
      await agent.navigate('https://example.com');
      await agent.click('.btn-primary');

      const clickEntry = agent.history[1];
      expect(clickEntry.action.type).toBe(ActionTypes.CLICK);
      expect(clickEntry.action.params.selector).toBe('.btn-primary');
    });

    test('should throw error if clicking before navigation', async () => {
      await expect(agent.click('#button')).rejects.toThrow('No page loaded');
    });
  });

  describe('Type', () => {
    test('should type text into input', async () => {
      await agent.navigate('https://example.com');
      const result = await agent.type('#search-input', 'test query');

      expect(result.success).toBe(true);
      expect(result.selector).toBe('#search-input');
      expect(result.text).toBe('test query');
    });

    test('should record type action in history', async () => {
      await agent.navigate('https://example.com');
      await agent.type('#username', 'john_doe');

      const typeEntry = agent.history[1];
      expect(typeEntry.action.type).toBe(ActionTypes.TYPE);
      expect(typeEntry.action.params.selector).toBe('#username');
      expect(typeEntry.action.params.text).toBe('john_doe');
    });

    test('should throw error if typing before navigation', async () => {
      await expect(agent.type('#input', 'text')).rejects.toThrow('No page loaded');
    });
  });

  describe('Extract', () => {
    test('should extract text from element', async () => {
      await agent.navigate('https://example.com');
      const result = await agent.extract('.article-title');

      expect(result.success).toBe(true);
      expect(result.selector).toBe('.article-title');
      expect(result.text).toBe('Mock content for .article-title');
    });

    test('should record extract action in history', async () => {
      await agent.navigate('https://example.com');
      await agent.extract('#content');

      const extractEntry = agent.history[1];
      expect(extractEntry.action.type).toBe(ActionTypes.EXTRACT);
      expect(extractEntry.action.params.selector).toBe('#content');
    });

    test('should throw error if extracting before navigation', async () => {
      await expect(agent.extract('#content')).rejects.toThrow('No page loaded');
    });
  });

  describe('getPageContent', () => {
    test('should return page content after navigation', async () => {
      await agent.navigate('https://example.com');
      const content = await agent.getPageContent();

      expect(content).toContain('<html>');
      expect(content).toContain('https://example.com');
    });

    test('should return empty content before navigation', async () => {
      const content = await agent.getPageContent();
      expect(content).toBe('');
    });
  });

  describe('Scroll', () => {
    test('should scroll with default options', async () => {
      await agent.navigate('https://example.com');
      const result = await agent.scroll();

      expect(result.success).toBe(true);
      expect(result.direction).toBe('down');
      expect(result.amount).toBe(500);
    });

    test('should scroll with custom options', async () => {
      await agent.navigate('https://example.com');
      const result = await agent.scroll({ direction: 'up', amount: 300 });

      expect(result.direction).toBe('up');
      expect(result.amount).toBe(300);
    });

    test('should record scroll in history', async () => {
      await agent.navigate('https://example.com');
      await agent.scroll({ direction: 'down' });

      const scrollEntry = agent.history[1];
      expect(scrollEntry.action.type).toBe(ActionTypes.SCROLL);
    });
  });

  describe('State Tracking', () => {
    test('getState should return current state snapshot', async () => {
      await agent.navigate('https://example.com');
      await agent.click('#button');

      const state = agent.getState();
      expect(state.name).toBe('TestAgent');
      expect(state.currentUrl).toBe('https://example.com');
      expect(state.stepCount).toBe(2);
      expect(state.maxSteps).toBe(10);
      expect(state.historyLength).toBe(2);
      expect(state.errorCount).toBe(0);
    });

    test('getHistory should return all history entries', async () => {
      await agent.navigate('https://example.com');
      await agent.click('#btn1');
      await agent.click('#btn2');

      const history = agent.getHistory();
      expect(history).toHaveLength(3);
      expect(history[0].action.type).toBe(ActionTypes.NAVIGATE);
      expect(history[1].action.type).toBe(ActionTypes.CLICK);
      expect(history[2].action.type).toBe(ActionTypes.CLICK);
    });

    test('getHistory should return a copy', async () => {
      await agent.navigate('https://example.com');
      const history = agent.getHistory();
      history.push({ fake: 'entry' });
      
      expect(agent.history).toHaveLength(1);
    });
  });

  describe('Error Handling', () => {
    test('should record errors in history', async () => {
      try {
        await agent.click('#button'); // No navigation first
      } catch (error) {
        // Expected
      }

      expect(agent.errors).toHaveLength(1);
      expect(agent.errors[0].message).toContain('No page loaded');
      expect(agent.errors[0].step).toBe(1);
    });

    test('getErrors should return all errors', async () => {
      try {
        await agent.click('#button1');
      } catch (e) {}
      try {
        await agent.type('#input', 'text');
      } catch (e) {}

      const errors = agent.getErrors();
      expect(errors).toHaveLength(2);
    });

    test('getErrors should return a copy', async () => {
      try {
        await agent.click('#button');
      } catch (e) {}
      
      const errors = agent.getErrors();
      errors.push({ fake: 'error' });
      
      expect(agent.errors).toHaveLength(1);
    });

    test('should increment step count on error', async () => {
      try {
        await agent.click('#button');
      } catch (e) {}

      expect(agent.stepCount).toBe(1);
    });
  });

  describe('Step Limit', () => {
    test('should throw error when max steps exceeded', async () => {
      agent = new Agent({ maxSteps: 2, browser: mockBrowser });
      
      await agent.navigate('https://example.com');
      await agent.click('#btn');
      
      await expect(agent.click('#btn2')).rejects.toThrow('Maximum steps (2) exceeded');
    });

    test('canContinue should return true when under limit', () => {
      expect(agent.canContinue()).toBe(true);
    });

    test('canContinue should return false when over limit', async () => {
      agent = new Agent({ maxSteps: 2, browser: mockBrowser });
      await agent.navigate('https://example.com');
      await agent.click('#btn');
      
      expect(agent.canContinue()).toBe(false);
    });

    test('canContinue should return false after too many errors', async () => {
      for (let i = 0; i < 10; i++) {
        try {
          await agent.click('#button');
        } catch (e) {}
      }
      
      expect(agent.canContinue()).toBe(false);
    });
  });

  describe('Reset', () => {
    test('should reset all state', async () => {
      await agent.navigate('https://example.com');
      await agent.click('#button');
      
      agent.reset();

      expect(agent.currentUrl).toBeNull();
      expect(agent.history).toEqual([]);
      expect(agent.stepCount).toBe(0);
      expect(agent.errors).toEqual([]);
      expect(agent.lastAction).toBeNull();
      expect(agent.lastResult).toBeNull();
    });

    test('should preserve config after reset', async () => {
      agent.reset();
      
      expect(agent.name).toBe('TestAgent');
      expect(agent.maxSteps).toBe(10);
    });
  });

  describe('Action timestamps and IDs', () => {
    test('should assign unique IDs to actions', async () => {
      await agent.navigate('https://example.com');
      await agent.click('#btn1');

      const id1 = agent.history[0].action.id;
      const id2 = agent.history[1].action.id;
      
      expect(id1).toBeDefined();
      expect(id2).toBeDefined();
      expect(id1).not.toBe(id2);
    });

    test('should record timestamps', async () => {
      const before = Date.now();
      await agent.navigate('https://example.com');
      const after = Date.now();

      const timestamp = agent.history[0].action.timestamp;
      expect(timestamp).toBeGreaterThanOrEqual(before);
      expect(timestamp).toBeLessThanOrEqual(after);
    });
  });
});

describe('MockBrowser', () => {
  let browser;

  beforeEach(() => {
    browser = new MockBrowser();
  });

  test('should track current URL', async () => {
    await browser.navigate('https://test.com');
    expect(browser.currentUrl).toBe('https://test.com');
  });

  test('should throw error on click before navigation', async () => {
    await expect(browser.click('#btn')).rejects.toThrow('No page loaded');
  });

  test('should throw error on type before navigation', async () => {
    await expect(browser.type('#input', 'text')).rejects.toThrow('No page loaded');
  });

  test('should throw error on extract before navigation', async () => {
    await expect(browser.extract('#content')).rejects.toThrow('No page loaded');
  });
});
