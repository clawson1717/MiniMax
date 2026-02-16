/**
 * UpdateTrigger Class
 * 
 * Events that cause belief revisions in agents.
 * Manages different trigger types and tracks triggered events.
 * 
 * Trigger Types:
 * - evidence: Direct observation or data that supports/refutes a belief
 * - inference: Logical derivation from existing beliefs
 * - persuasion: Communication from another agent attempting to influence beliefs
 * - observation: Noticing changes in the environment or other agents
 */

class UpdateTrigger {
  /**
   * Create a new UpdateTrigger
   * @param {string} type - Type of trigger (evidence, inference, persuasion, observation)
   * @param {string} proposition - The proposition being triggered
   * @param {Object} data - Additional trigger data
   */
  constructor(type, proposition, data = {}) {
    this.id = `${type}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    this.type = type;
    this.proposition = proposition;
    this.data = data;
    this.timestamp = Date.now();
    this.processed = false;
  }

  /**
   * Mark the trigger as processed
   */
  markProcessed() {
    this.processed = true;
  }

  /**
   * Get trigger details
   * @returns {Object} - Trigger details
   */
  toJSON() {
    return {
      id: this.id,
      type: this.type,
      proposition: this.proposition,
      data: this.data,
      timestamp: this.timestamp,
      processed: this.processed
    };
  }
}

/**
 * UpdateTriggerManager Class
 * 
 * Manages registration and triggering of events that cause belief revisions.
 */

class UpdateTriggerManager {
  /**
   * Create a new UpdateTriggerManager
   */
  constructor() {
    this.triggers = new Map(); // type -> Set of registered triggers
    this.triggeredHistory = []; // History of all triggered events
    this.listeners = new Map(); // proposition -> Array of callback functions
    
    // Initialize trigger type registries
    this.TRIGGER_TYPES = ['evidence', 'inference', 'persuasion', 'observation'];
    
    for (const type of this.TRIGGER_TYPES) {
      this.triggers.set(type, new Map()); // id -> UpdateTrigger
    }
  }

  /**
   * Register a trigger for a specific event type
   * @param {string} type - Trigger type (evidence, inference, persuasion, observation)
   * @param {string} proposition - Proposition this trigger applies to
   * @param {Object} [data={}] - Additional trigger configuration
   * @returns {UpdateTrigger} - The registered trigger
   * @throws {Error} - If trigger type is invalid
   */
  register(type, proposition, data = {}) {
    if (!this.TRIGGER_TYPES.includes(type)) {
      throw new Error(`Invalid trigger type: ${type}. Must be one of: ${this.TRIGGER_TYPES.join(', ')}`);
    }

    const trigger = new UpdateTrigger(type, proposition, data);
    this.triggers.get(type).set(trigger.id, trigger);
    
    return trigger;
  }

  /**
   * Trigger an event that causes belief revision
   * @param {string} type - Trigger type
   * @param {string} proposition - Proposition being triggered
   * @param {Object} [data={}] - Trigger data (confidence, justification, source, etc.)
   * @returns {UpdateTrigger} - The triggered event
   */
  trigger(type, proposition, data = {}) {
    if (!this.TRIGGER_TYPES.includes(type)) {
      throw new Error(`Invalid trigger type: ${type}`);
    }

    // Create a new triggered event
    const trigger = new UpdateTrigger(type, proposition, {
      ...data,
      triggeredAt: Date.now()
    });

    // Add to history
    this.triggeredHistory.push(trigger);

    // Notify listeners
    this._notifyListeners(proposition, trigger);

    return trigger;
  }

  /**
   * Get all triggered events matching criteria
   * @param {Object} [filters={}] - Filter criteria
   * @param {string} [filters.type] - Filter by trigger type
   * @param {string} [filters.proposition] - Filter by proposition
   * @param {boolean} [filters.unprocessedOnly=false] - Only return unprocessed triggers
   * @returns {Array<UpdateTrigger>} - Array of matching triggers
   */
  getTriggered(filters = {}) {
    let results = [...this.triggeredHistory];

    if (filters.type) {
      results = results.filter(t => t.type === filters.type);
    }

    if (filters.proposition) {
      results = results.filter(t => t.proposition === filters.proposition);
    }

    if (filters.unprocessedOnly) {
      results = results.filter(t => !t.processed);
    }

    return results;
  }

  /**
   * Register a listener for proposition updates
   * @param {string} proposition - Proposition to listen for
   * @param {Function} callback - Callback function (trigger) => void
   */
  addListener(proposition, callback) {
    if (!this.listeners.has(proposition)) {
      this.listeners.set(proposition, []);
    }
    this.listeners.get(proposition).push(callback);
  }

  /**
   * Remove a listener
   * @param {string} proposition - Proposition to stop listening to
   * @param {Function} callback - Callback to remove
   */
  removeListener(proposition, callback) {
    if (this.listeners.has(proposition)) {
      const callbacks = this.listeners.get(proposition);
      const index = callbacks.indexOf(callback);
      if (index > -1) {
        callbacks.splice(index, 1);
      }
    }
  }

  /**
   * Notify listeners of a triggered event
   * @param {string} proposition - Proposition that was triggered
   * @param {UpdateTrigger} trigger - The trigger event
   * @private
   */
  _notifyListeners(proposition, trigger) {
    if (this.listeners.has(proposition)) {
      for (const callback of this.listeners.get(proposition)) {
        try {
          callback(trigger);
        } catch (error) {
          console.error(`Error in trigger listener for "${proposition}":`, error);
        }
      }
    }
  }

  /**
   * Get registered triggers by type
   * @param {string} type - Trigger type
   * @returns {Array<UpdateTrigger>} - Array of registered triggers
   */
  getRegisteredTriggers(type) {
    if (!this.triggers.has(type)) {
      return [];
    }
    return Array.from(this.triggers.get(type).values());
  }

  /**
   * Get trigger statistics
   * @returns {Object} - Statistics about triggers
   */
  getStats() {
    const stats = {};
    for (const type of this.TRIGGER_TYPES) {
      stats[type] = {
        registered: this.triggers.get(type).size,
        triggered: this.triggeredHistory.filter(t => t.type === type).length
      };
    }
    return stats;
  }

  /**
   * Clear trigger history
   */
  clearHistory() {
    this.triggeredHistory = [];
  }

  /**
   * Export triggered history
   * @returns {Array} - History as JSON
   */
  exportHistory() {
    return this.triggeredHistory.map(t => t.toJSON());
  }
}

module.exports = { UpdateTrigger, UpdateTriggerManager };
