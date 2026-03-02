/**
 * Core Agent class for Resilient Adaptive Agent
 * Provides basic web navigation capabilities with state tracking
 */

const { Action, ActionTypes, ActionFactory } = require('./Action');

/**
 * MockBrowser - Simulates browser operations for testing
 * In production, this would be replaced with actual Playwright/Puppeteer
 */
class MockBrowser {
  constructor() {
    this.currentUrl = null;
    this.pageContent = '';
    this.elements = new Map();
  }

  async navigate(url) {
    this.currentUrl = url;
    this.pageContent = `<html><body><h1>Mock page at ${url}</h1></body></html>`;
    return { success: true, url };
  }

  async click(selector) {
    if (!this.currentUrl) {
      throw new Error('No page loaded');
    }
    return { success: true, selector };
  }

  async type(selector, text) {
    if (!this.currentUrl) {
      throw new Error('No page loaded');
    }
    return { success: true, selector, text };
  }

  async extract(selector) {
    if (!this.currentUrl) {
      throw new Error('No page loaded');
    }
    return { success: true, selector, text: `Mock content for ${selector}` };
  }

  async getPageContent() {
    return this.pageContent;
  }
}

/**
 * Agent class - Core web navigation agent
 */
class Agent {
  /**
   * Create a new Agent instance
   * @param {Object} config - Agent configuration
   * @param {string} [config.name='Agent'] - Agent name
   * @param {number} [config.maxSteps=100] - Maximum steps before stopping
   * @param {Object} [config.browser] - Browser instance (defaults to MockBrowser)
   * @param {boolean} [config.verbose=false] - Enable verbose logging
   */
  constructor(config = {}) {
    this.name = config.name || 'Agent';
    this.maxSteps = config.maxSteps || 100;
    this.verbose = config.verbose || false;
    
    // Browser interface (mock by default)
    this.browser = config.browser || new MockBrowser();
    
    // State tracking
    this.currentUrl = null;
    this.history = [];
    this.stepCount = 0;
    this.isRunning = false;
    this.errors = [];
    
    // Action tracking
    this.lastAction = null;
    this.lastResult = null;
  }

  /**
   * Log a message if verbose mode is enabled
   * @private
   * @param {string} message - Message to log
   */
  _log(message) {
    if (this.verbose) {
      console.log(`[${this.name}] ${message}`);
    }
  }

  /**
   * Record an action in history
   * @private
   * @param {Action} action - The action performed
   * @param {*} result - The result of the action
   * @param {Error} [error] - Error if action failed
   */
  _recordAction(action, result, error = null) {
    this.stepCount++;
    this.lastAction = action;
    this.lastResult = result;
    
    const historyEntry = {
      step: this.stepCount,
      action: action.toJSON(),
      result: error ? null : result,
      error: error ? error.message : null,
      timestamp: Date.now()
    };
    
    this.history.push(historyEntry);
    
    if (error) {
      this.errors.push({
        step: this.stepCount,
        message: error.message,
        action: action.toString()
      });
    }
  }

  /**
   * Check if agent has exceeded max steps
   * @private
   * @returns {boolean}
   */
  _checkStepLimit() {
    if (this.stepCount >= this.maxSteps) {
      throw new Error(`Maximum steps (${this.maxSteps}) exceeded`);
    }
    return true;
  }

  /**
   * Navigate to a URL
   * @param {string} url - URL to navigate to
   * @returns {Promise<Object>} Navigation result
   * @throws {Error} If navigation fails or max steps exceeded
   */
  async navigate(url) {
    this._checkStepLimit();
    this._log(`Navigating to: ${url}`);
    
    const action = ActionFactory.navigate(url);
    
    try {
      const result = await this.browser.navigate(url);
      this.currentUrl = url;
      this._recordAction(action, result);
      this._log(`Successfully navigated to: ${url}`);
      return result;
    } catch (error) {
      this._recordAction(action, null, error);
      this._log(`Navigation failed: ${error.message}`);
      throw error;
    }
  }

  /**
   * Click an element on the page
   * @param {string} selector - CSS selector for the element
   * @returns {Promise<Object>} Click result
   * @throws {Error} If click fails or max steps exceeded
   */
  async click(selector) {
    this._checkStepLimit();
    this._log(`Clicking element: ${selector}`);
    
    const action = ActionFactory.click(selector);
    
    try {
      const result = await this.browser.click(selector);
      this._recordAction(action, result);
      this._log(`Successfully clicked: ${selector}`);
      return result;
    } catch (error) {
      this._recordAction(action, null, error);
      this._log(`Click failed: ${error.message}`);
      throw error;
    }
  }

  /**
   * Type text into an input element
   * @param {string} selector - CSS selector for the input element
   * @param {string} text - Text to type
   * @returns {Promise<Object>} Type result
   * @throws {Error} If typing fails or max steps exceeded
   */
  async type(selector, text) {
    this._checkStepLimit();
    this._log(`Typing into ${selector}: ${text}`);
    
    const action = ActionFactory.type(selector, text);
    
    try {
      const result = await this.browser.type(selector, text);
      this._recordAction(action, result);
      this._log(`Successfully typed into: ${selector}`);
      return result;
    } catch (error) {
      this._recordAction(action, null, error);
      this._log(`Type failed: ${error.message}`);
      throw error;
    }
  }

  /**
   * Extract text from an element
   * @param {string} selector - CSS selector for the element
   * @returns {Promise<Object>} Extraction result with text property
   * @throws {Error} If extraction fails or max steps exceeded
   */
  async extract(selector) {
    this._checkStepLimit();
    this._log(`Extracting from: ${selector}`);
    
    const action = ActionFactory.extract(selector);
    
    try {
      const result = await this.browser.extract(selector);
      this._recordAction(action, result);
      this._log(`Successfully extracted from: ${selector}`);
      return result;
    } catch (error) {
      this._recordAction(action, null, error);
      this._log(`Extract failed: ${error.message}`);
      throw error;
    }
  }

  /**
   * Get the current page content
   * @returns {Promise<string>} Page HTML content
   * @throws {Error} If content retrieval fails
   */
  async getPageContent() {
    this._log('Getting page content');
    
    try {
      const content = await this.browser.getPageContent();
      return content;
    } catch (error) {
      this._log(`Get page content failed: ${error.message}`);
      throw error;
    }
  }

  /**
   * Scroll the page
   * @param {Object} [options] - Scroll options
   * @param {string} [options.direction='down'] - Direction ('up', 'down', 'left', 'right')
   * @param {number} [options.amount=500] - Pixels to scroll
   * @returns {Promise<Object>} Scroll result
   * @throws {Error} If scroll fails or max steps exceeded
   */
  async scroll(options = {}) {
    this._checkStepLimit();
    const direction = options.direction || 'down';
    const amount = options.amount || 500;
    this._log(`Scrolling ${direction} by ${amount}px`);
    
    const action = ActionFactory.scroll(options);
    
    try {
      // Mock scroll implementation
      const result = { success: true, direction, amount };
      this._recordAction(action, result);
      this._log(`Successfully scrolled ${direction}`);
      return result;
    } catch (error) {
      this._recordAction(action, null, error);
      this._log(`Scroll failed: ${error.message}`);
      throw error;
    }
  }

  /**
   * Get current agent state
   * @returns {Object} Current state snapshot
   */
  getState() {
    return {
      name: this.name,
      currentUrl: this.currentUrl,
      stepCount: this.stepCount,
      maxSteps: this.maxSteps,
      isRunning: this.isRunning,
      lastAction: this.lastAction ? this.lastAction.toJSON() : null,
      lastResult: this.lastResult,
      errorCount: this.errors.length,
      historyLength: this.history.length
    };
  }

  /**
   * Get full action history
   * @returns {Array<Object>} Array of history entries
   */
  getHistory() {
    return [...this.history];
  }

  /**
   * Get all errors encountered
   * @returns {Array<Object>} Array of error entries
   */
  getErrors() {
    return [...this.errors];
  }

  /**
   * Reset agent state (keeps config)
   */
  reset() {
    this._log('Resetting agent state');
    this.currentUrl = null;
    this.history = [];
    this.stepCount = 0;
    this.isRunning = false;
    this.errors = [];
    this.lastAction = null;
    this.lastResult = null;
  }

  /**
   * Check if agent can continue (under max steps, no critical errors)
   * @returns {boolean}
   */
  canContinue() {
    return this.stepCount < this.maxSteps && this.errors.length < 10;
  }
}

module.exports = {
  Agent,
  MockBrowser
};
