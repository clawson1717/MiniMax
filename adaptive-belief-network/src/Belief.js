/**
 * Belief Class
 * 
 * Represents an agent's belief with a proposition, confidence level, and justification.
 * 
 * Confidence is a value between 0 and 1, where:
 * - 0 means complete disbelief or maximum uncertainty
 * - 1 means complete certainty
 * - Values between represent varying degrees of confidence
 */

class Belief {
  /**
   * Create a new Belief
   * @param {string} proposition - The statement or proposition of the belief
   * @param {number} confidence - Confidence level between 0 and 1
   * @param {string} justification - Reason or evidence supporting the belief
   */
  constructor(proposition, confidence, justification) {
    this.proposition = proposition;
    this.confidence = this._validateConfidence(confidence);
    this.justification = justification;
    this.timestamp = Date.now();
    this.history = [{
      confidence: this.confidence,
      justification: this.justification,
      timestamp: this.timestamp
    }];
  }

  /**
   * Validate that confidence is between 0 and 1
   * @param {number} confidence - The confidence value to validate
   * @returns {number} - Validated confidence value
   * @throws {Error} - If confidence is not between 0 and 1
   */
  _validateConfidence(confidence) {
    if (typeof confidence !== 'number') {
      throw new Error('Confidence must be a number');
    }
    if (confidence < 0 || confidence > 1) {
      throw new Error('Confidence must be between 0 and 1');
    }
    return confidence;
  }

  /**
   * Update the belief with new confidence and/or justification
   * @param {number} [confidence] - New confidence level (0-1)
   * @param {string} [justification] - New or updated justification
   * @returns {Belief} - Returns this belief for chaining
   */
  update(confidence, justification) {
    if (confidence !== undefined) {
      this.confidence = this._validateConfidence(confidence);
    }
    if (justification !== undefined) {
      this.justification = justification;
    }
    
    this.timestamp = Date.now();
    this.history.push({
      confidence: this.confidence,
      justification: this.justification,
      timestamp: this.timestamp
    });
    
    return this;
  }

  /**
   * Get the history of belief updates
   * @returns {Array} - Array of history entries
   */
  getHistory() {
    return [...this.history];
  }

  /**
   * Get a summary of the belief
   * @returns {Object} - Summary object
   */
  toJSON() {
    return {
      proposition: this.proposition,
      confidence: this.confidence,
      justification: this.justification,
      timestamp: this.timestamp
    };
  }

  /**
   * Create a Belief from a JSON object
   * @param {Object} json - JSON object with belief data
   * @returns {Belief} - New Belief instance
   */
  static fromJSON(json) {
    const belief = new Belief(json.proposition, json.confidence, json.justification);
    belief.timestamp = json.timestamp || Date.now();
    return belief;
  }
}

module.exports = { Belief };
