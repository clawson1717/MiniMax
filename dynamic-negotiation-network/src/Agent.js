/**
 * Agent Class
 * 
 * Represents a negotiating agent with needs, offers, and beliefs.
 * Core data structure for all agents in the negotiation network.
 */

class Agent {
  /**
   * Create a new Agent
   * @param {string} id - Unique identifier for the agent
   * @param {Object} options - Agent configuration
   * @param {string[]} options.needs - What the agent needs
   * @param {string[]} options.offers - What the agent can offer
   * @param {Object} options.beliefs - Agent's beliefs about others (for reactive reasoning)
   */
  constructor(id, options = {}) {
    this.id = id;
    this.needs = options.needs || [];
    this.offers = options.offers || [];
    this.beliefs = options.beliefs || {};
    this.history = [];
    this.strategy = options.strategy || 'default';
    this.utility = options.utility || 1.0;
  }

  /**
   * Get the agent's current needs
   * @returns {string[]}
   */
  getNeeds() {
    return [...this.needs];
  }

  /**
   * Get the agent's current offers
   * @returns {string[]}
   */
  getOffers() {
    return [...this.offers];
  }

  /**
   * Update the agent's needs
   * @param {string[]} newNeeds
   */
  setNeeds(newNeeds) {
    this.needs = [...newNeeds];
  }

  /**
   * Update the agent's offers
   * @param {string[]} newOffers
   */
  setOffers(newOffers) {
    this.offers = [...newOffers];
  }

  /**
   * Add a need
   * @param {string} need
   */
  addNeed(need) {
    if (!this.needs.includes(need)) {
      this.needs.push(need);
    }
  }

  /**
   * Remove a need
   * @param {string} need
   */
  removeNeed(need) {
    this.needs = this.needs.filter(n => n !== need);
  }

  /**
   * Add an offer
   * @param {string} offer
   */
  addOffer(offer) {
    if (!this.offers.includes(offer)) {
      this.offers.push(offer);
    }
  }

  /**
   * Remove an offer
   * @param {string} offer
   */
  removeOffer(offer) {
    this.offers = this.offers.filter(o => o !== offer);
  }

  /**
   * Update belief about another agent
   * Based on Reactive Knowledge Representation principles
   * @param {string} agentId - Agent being observed
   * @param {Object} belief - New belief state
   */
  updateBelief(agentId, belief) {
    this.beliefs[agentId] = {
      ...this.beliefs[agentId],
      ...belief,
      timestamp: Date.now()
    };
  }

  /**
   * Get belief about another agent
   * @param {string} agentId
   * @returns {Object|null}
   */
  getBelief(agentId) {
    return this.beliefs[agentId] || null;
  }

  /**
   * Record a negotiation event in history
   * @param {Object} event
   */
  recordEvent(event) {
    this.history.push({
      ...event,
      timestamp: Date.now()
    });
  }

  /**
   * Get negotiation history
   * @returns {Object[]}
   */
  getHistory() {
    return [...this.history];
  }

  /**
   * Clear negotiation history
   */
  clearHistory() {
    this.history = [];
  }

  /**
   * Check if agent's needs are satisfied
   * @returns {boolean}
   */
  needsSatisfied() {
    return this.needs.length === 0;
  }

  /**
   * Get agent's strategy
   * @returns {string}
   */
  getStrategy() {
    return this.strategy;
  }

  /**
   * Set agent's negotiation strategy
   * @param {string} strategy
   */
  setStrategy(strategy) {
    this.strategy = strategy;
  }

  /**
   * Generate a response to an offer
   * @param {string} offer - The incoming offer
   * @returns {Object} - Response with action and counter-offer
   */
  respondToOffer(offer) {
    // Simple default strategy - accept if meets some need, else counter
    const matchedNeed = this.needs.find(need => 
      offer.toLowerCase().includes(need.toLowerCase())
    );

    if (matchedNeed) {
      return {
        action: 'accept',
        message: `Accepted offer: ${offer}`,
        matchedNeed
      };
    }

    // Generate counter-offer
    const counterOffer = this.offers[Math.floor(Math.random() * this.offers.length)];
    return {
      action: 'counter',
      message: `Counter-offer: ${counterOffer}`,
      counterOffer
    };
  }

  /**
   * Convert agent to plain object
   * @returns {Object}
   */
  toJSON() {
    return {
      id: this.id,
      needs: this.needs,
      offers: this.offers,
      beliefs: this.beliefs,
      history: this.history,
      strategy: this.strategy,
      utility: this.utility
    };
  }

  /**
   * Create Agent from plain object
   * @param {Object} data
   * @returns {Agent}
   */
  static fromJSON(data) {
    const agent = new Agent(data.id, {
      needs: data.needs,
      offers: data.offers,
      beliefs: data.beliefs,
      strategy: data.strategy,
      utility: data.utility
    });
    agent.history = data.history || [];
    return agent;
  }
}

module.exports = Agent;
