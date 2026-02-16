/**
 * Action Types and Factory for Resilient Adaptive Agent
 * Defines all possible actions an agent can take
 */

/**
 * Action type constants
 * @readonly
 * @enum {string}
 */
const ActionTypes = {
  NAVIGATE: 'NAVIGATE',
  CLICK: 'CLICK',
  TYPE: 'TYPE',
  EXTRACT: 'EXTRACT',
  SCROLL: 'SCROLL'
};

/**
 * Action class representing a single action to be performed
 */
class Action {
  /**
   * Create an Action
   * @param {string} type - The action type (from ActionTypes)
   * @param {Object} params - Action-specific parameters
   * @param {number} [priority=0] - Action priority (higher = more urgent)
   */
  constructor(type, params = {}, priority = 0) {
    this.type = type;
    this.params = params;
    this.priority = priority;
    this.timestamp = Date.now();
    this.id = `action_${this.timestamp}_${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Validate the action has required parameters
   * @returns {{valid: boolean, error?: string}}
   */
  validate() {
    // Check type is valid
    if (!Object.values(ActionTypes).includes(this.type)) {
      return { valid: false, error: `Invalid action type: ${this.type}` };
    }

    // Type-specific validation
    switch (this.type) {
      case ActionTypes.NAVIGATE:
        if (!this.params.url) {
          return { valid: false, error: 'NAVIGATE action requires url parameter' };
        }
        break;
      case ActionTypes.CLICK:
        if (!this.params.selector) {
          return { valid: false, error: 'CLICK action requires selector parameter' };
        }
        break;
      case ActionTypes.TYPE:
        if (!this.params.selector || !this.params.text) {
          return { valid: false, error: 'TYPE action requires selector and text parameters' };
        }
        break;
      case ActionTypes.EXTRACT:
        if (!this.params.selector) {
          return { valid: false, error: 'EXTRACT action requires selector parameter' };
        }
        break;
      case ActionTypes.SCROLL:
        // SCROLL is optional, defaults to scrolling down
        break;
    }

    return { valid: true };
  }

  /**
   * Convert action to JSON-serializable object
   * @returns {Object}
   */
  toJSON() {
    return {
      id: this.id,
      type: this.type,
      params: this.params,
      priority: this.priority,
      timestamp: this.timestamp
    };
  }

  /**
   * Create a string representation of the action
   * @returns {string}
   */
  toString() {
    const paramStr = Object.entries(this.params)
      .map(([k, v]) => `${k}=${v}`)
      .join(', ');
    return `${this.type}(${paramStr})`;
  }
}

/**
 * Action factory for creating common actions
 */
const ActionFactory = {
  /**
   * Create a NAVIGATE action
   * @param {string} url - URL to navigate to
   * @param {number} [priority=0] - Action priority
   * @returns {Action}
   */
  navigate(url, priority = 0) {
    return new Action(ActionTypes.NAVIGATE, { url }, priority);
  },

  /**
   * Create a CLICK action
   * @param {string} selector - CSS selector for element to click
   * @param {number} [priority=0] - Action priority
   * @returns {Action}
   */
  click(selector, priority = 0) {
    return new Action(ActionTypes.CLICK, { selector }, priority);
  },

  /**
   * Create a TYPE action
   * @param {string} selector - CSS selector for input element
   * @param {string} text - Text to type
   * @param {number} [priority=0] - Action priority
   * @returns {Action}
   */
  type(selector, text, priority = 0) {
    return new Action(ActionTypes.TYPE, { selector, text }, priority);
  },

  /**
   * Create an EXTRACT action
   * @param {string} selector - CSS selector for element to extract text from
   * @param {number} [priority=0] - Action priority
   * @returns {Action}
   */
  extract(selector, priority = 0) {
    return new Action(ActionTypes.EXTRACT, { selector }, priority);
  },

  /**
   * Create a SCROLL action
   * @param {Object} [options] - Scroll options
   * @param {string} [options.direction='down'] - Direction to scroll ('up', 'down', 'left', 'right')
   * @param {number} [options.amount=500] - Pixels to scroll
   * @param {number} [priority=0] - Action priority
   * @returns {Action}
   */
  scroll(options = {}, priority = 0) {
    const params = {
      direction: options.direction || 'down',
      amount: options.amount || 500
    };
    return new Action(ActionTypes.SCROLL, params, priority);
  }
};

module.exports = {
  ActionTypes,
  Action,
  ActionFactory
};
