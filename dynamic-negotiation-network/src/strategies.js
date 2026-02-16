/**
 * Negotiation Strategies
 * 
 * Defines different strategies agents can use during negotiations:
 * - AcceptStrategy: Always accepts if offer meets threshold
 * - RejectStrategy: Always rejects
 * - CounterStrategy: Proposes counter-offers based on gap
 * - RandomStrategy: Random acceptance with configurable probability
 */

class AcceptStrategy {
  /**
   * Create an AcceptStrategy
   * @param {Object} options - Strategy configuration
   * @param {number} options.threshold - Minimum similarity threshold (0-1)
   */
  constructor(options = {}) {
    this.threshold = options.threshold || 0.3;
    this.name = 'accept';
  }

  /**
   * Evaluate an offer and decide action
   * @param {Object} context - Current negotiation context
   * @returns {Object} - Decision with action and details
   */
  evaluate(context) {
    const { offer, agent, senderOffers } = context;
    
    // Check if any agent needs match the offer
    const agentNeeds = agent.getNeeds();
    const matchedNeed = this._findMatchingNeed(offer, agentNeeds);
    
    if (matchedNeed && matchedNeed.score >= this.threshold) {
      return {
        action: 'accept',
        reason: `Offer meets threshold (${matchedNeed.score.toFixed(2)} >= ${this.threshold})`,
        matchedNeed: matchedNeed.need,
        confidence: matchedNeed.score
      };
    }
    
    // Check sender's offers against our needs
    if (senderOffers && senderOffers.length > 0) {
      for (const senderOffer of senderOffers) {
        const match = this._findMatchingNeed(senderOffer, agentNeeds);
        if (match && match.score >= this.threshold) {
          return {
            action: 'accept',
            reason: `Sender offer matches need (${match.score.toFixed(2)} >= ${this.threshold})`,
            matchedNeed: match.need,
            confidence: match.score
          };
        }
      }
    }
    
    // No match found - reject
    return {
      action: 'reject',
      reason: `No offer meets acceptance threshold (${this.threshold})`,
      confidence: 0
    };
  }

  /**
   * Find the best matching need for an offer
   * @private
   */
  _findMatchingNeed(offer, needs) {
    let bestMatch = null;
    
    for (const need of needs) {
      const score = this._calculateSimilarity(offer, need);
      if (!bestMatch || score > bestMatch.score) {
        bestMatch = { need, score };
      }
    }
    
    return bestMatch;
  }

  /**
   * Calculate simple similarity between two strings
   * @private
   */
  _calculateSimilarity(str1, str2) {
    const s1 = str1.toLowerCase();
    const s2 = str2.toLowerCase();
    
    // Exact match
    if (s1 === s2) return 1.0;
    
    // Contains match
    if (s1.includes(s2) || s2.includes(s1)) return 0.8;
    
    // Word overlap
    const words1 = new Set(s1.split(/[\s:,_-]+/));
    const words2 = new Set(s2.split(/[\s:,_-]+/));
    const intersection = new Set([...words1].filter(x => words2.has(x)));
    
    if (intersection.size === 0) return 0;
    
    const union = new Set([...words1, ...words2]);
    return intersection.size / union.size;
  }

  /**
   * Get strategy name
   * @returns {string}
   */
  getName() {
    return this.name;
  }

  /**
   * Get strategy description
   * @returns {string}
   */
  getDescription() {
    return `Accepts offers that meet a similarity threshold of ${this.threshold}`;
  }
}

class RejectStrategy {
  /**
   * Create a RejectStrategy
   * @param {Object} options - Strategy configuration
   * @param {number} options.probability - Base probability of rejection (0-1)
   */
  constructor(options = {}) {
    this.probability = options.probability || 1.0;
    this.name = 'reject';
  }

  /**
   * Evaluate an offer and decide action
   * @param {Object} context - Current negotiation context
   * @returns {Object} - Decision with action and details
   */
  evaluate(context) {
    // Always rejects based on probability
    if (Math.random() < this.probability) {
      return {
        action: 'reject',
        reason: 'Strategy configured to reject all offers',
        confidence: this.probability
      };
    }
    
    // Small chance to accept (unlikely with this strategy)
    return {
      action: 'accept',
      reason: 'Random acceptance despite reject strategy',
      confidence: 1 - this.probability
    };
  }

  /**
   * Get strategy name
   * @returns {string}
   */
  getName() {
    return this.name;
  }

  /**
   * Get strategy description
   * @returns {string}
   */
  getDescription() {
    return `Rejects offers with probability ${this.probability}`;
  }
}

class CounterStrategy {
  /**
   * Create a CounterStrategy
   * @param {Object} options - Strategy configuration
   * @param {number} options.counterThreshold - Threshold to trigger counter-offers
   * @param {number} options.concessionRate - How much to concede per round (0-1)
   */
  constructor(options = {}) {
    this.counterThreshold = options.counterThreshold || 0.2;
    this.concessionRate = options.concessionRate || 0.1;
    this.name = 'counter';
    this.roundCount = 0;
  }

  /**
   * Evaluate an offer and decide action
   * @param {Object} context - Current negotiation context
   * @returns {Object} - Decision with action and details
   */
  evaluate(context) {
    const { offer, agent, roundNumber = 0 } = context;
    this.roundCount = roundNumber;
    
    const agentNeeds = agent.getNeeds();
    const agentOffers = agent.getOffers();
    
    // Check if offer meets a need
    const matchedNeed = this._findMatchingNeed(offer, agentNeeds);
    
    if (matchedNeed && matchedNeed.score >= this.counterThreshold) {
      // Offer is good enough - accept
      return {
        action: 'accept',
        reason: `Offer meets counter threshold (${matchedNeed.score.toFixed(2)})`,
        matchedNeed: matchedNeed.need,
        confidence: matchedNeed.score
      };
    }
    
    // Need to counter - generate counter-offer
    const counterOffer = this._generateCounterOffer(agentOffers, offer);
    
    if (counterOffer) {
      return {
        action: 'counter',
        reason: `Counter-offer generated based on gap analysis`,
        counterOffer,
        originalOffer: offer,
        confidence: 0.5 - (this.roundCount * this.concessionRate)
      };
    }
    
    // No suitable counter-offer - reject
    return {
      action: 'reject',
      reason: 'No suitable counter-offer available',
      confidence: 0.3
    };
  }

  /**
   * Generate a counter-offer based on the gap
   * @private
   */
  _generateCounterOffer(agentOffers, incomingOffer) {
    if (!agentOffers || agentOffers.length === 0) {
      return null;
    }
    
    // Extract numeric values from offer (e.g., "$500" -> 500)
    const incomingValue = this._extractNumericValue(incomingOffer);
    const agentValue = this._extractNumericValue(agentOffers[0]);
    
    // If we can make a counter with adjusted value
    if (incomingValue !== null && agentValue !== null) {
      const gap = agentValue - incomingValue;
      const concession = gap * (1 - this.concessionRate * this.roundCount);
      const counterValue = Math.round(agentValue - concession);
      
      // Generate counter offer string
      const offerTemplate = agentOffers[0];
      return offerTemplate.replace(/\d+/, counterValue.toString());
    }
    
    // Fall back to random offer selection
    return agentOffers[Math.floor(Math.random() * agentOffers.length)];
  }

  /**
   * Extract numeric value from a string
   * @private
   */
  _extractNumericValue(str) {
    const match = str.match(/\d+/);
    return match ? parseInt(match[0], 10) : null;
  }

  /**
   * Find the best matching need for an offer
   * @private
   */
  _findMatchingNeed(offer, needs) {
    let bestMatch = null;
    
    for (const need of needs) {
      const score = this._calculateSimilarity(offer, need);
      if (!bestMatch || score > bestMatch.score) {
        bestMatch = { need, score };
      }
    }
    
    return bestMatch;
  }

  /**
   * Calculate simple similarity between two strings
   * @private
   */
  _calculateSimilarity(str1, str2) {
    const s1 = str1.toLowerCase();
    const s2 = str2.toLowerCase();
    
    if (s1 === s2) return 1.0;
    if (s1.includes(s2) || s2.includes(s1)) return 0.8;
    
    const words1 = new Set(s1.split(/[\s:,_-]+/));
    const words2 = new Set(s2.split(/[\s:,_-]+/));
    const intersection = new Set([...words1].filter(x => words2.has(x)));
    
    if (intersection.size === 0) return 0;
    
    const union = new Set([...words1, ...words2]);
    return intersection.size / union.size;
  }

  /**
   * Reset round counter
   */
  reset() {
    this.roundCount = 0;
  }

  /**
   * Get strategy name
   * @returns {string}
   */
  getName() {
    return this.name;
  }

  /**
   * Get strategy description
   * @returns {string}
   */
  getDescription() {
    return `Counter-offers based on gap analysis with ${(this.concessionRate * 100)}% concession rate`;
  }
}

class RandomStrategy {
  /**
   * Create a RandomStrategy
   * @param {Object} options - Strategy configuration
   * @param {number} options.acceptProbability - Probability of accepting (0-1)
   * @param {number} options.counterProbability - Probability of countering (0-1)
   */
  constructor(options = {}) {
    this.acceptProbability = options.acceptProbability || 0.3;
    this.counterProbability = options.counterProbability || 0.4;
    this.name = 'random';
  }

  /**
   * Evaluate an offer and decide action
   * @param {Object} context - Current negotiation context
   * @returns {Object} - Decision with action and details
   */
  evaluate(context) {
    const { offer, agent } = context;
    const agentOffers = agent.getOffers();
    
    const rand = Math.random();
    
    if (rand < this.acceptProbability) {
      return {
        action: 'accept',
        reason: 'Random acceptance',
        confidence: this.acceptProbability
      };
    } else if (rand < this.acceptProbability + this.counterProbability) {
      // Counter-offer
      const counterOffer = agentOffers.length > 0
        ? agentOffers[Math.floor(Math.random() * agentOffers.length)]
        : null;
      
      return {
        action: 'counter',
        reason: 'Random counter-offer',
        counterOffer: counterOffer || 'generic counter',
        confidence: this.counterProbability
      };
    } else {
      return {
        action: 'reject',
        reason: 'Random rejection',
        confidence: 1 - this.acceptProbability - this.counterProbability
      };
    }
  }

  /**
   * Get strategy name
   * @returns {string}
   */
  getName() {
    return this.name;
  }

  /**
   * Get strategy description
   * @returns {string}
   */
  getDescription() {
    return `Random decisions: ${(this.acceptProbability * 100)}% accept, ${(this.counterProbability * 100)}% counter, ${((1 - this.acceptProbability - this.counterProbability) * 100)}% reject`;
  }

  /**
   * Update accept probability
   * @param {number} probability
   */
  setAcceptProbability(probability) {
    this.acceptProbability = Math.max(0, Math.min(1, probability));
  }

  /**
   * Update counter probability
   * @param {number} probability
   */
  setCounterProbability(probability) {
    this.counterProbability = Math.max(0, Math.min(1, probability));
  }
}

/**
 * Strategy Factory
 * Creates strategy instances based on name and options
 */
class StrategyFactory {
  /**
   * Create a strategy by name
   * @param {string} strategyName - Name of the strategy
   * @param {Object} options - Strategy options
   * @returns {Object} - Strategy instance
   */
  static create(strategyName, options = {}) {
    switch (strategyName.toLowerCase()) {
      case 'accept':
        return new AcceptStrategy(options);
      case 'reject':
        return new RejectStrategy(options);
      case 'counter':
        return new CounterStrategy(options);
      case 'random':
        return new RandomStrategy(options);
      default:
        // Default to counter strategy
        return new CounterStrategy(options);
    }
  }

  /**
   * Get list of available strategies
   * @returns {string[]}
   */
  static getAvailableStrategies() {
    return ['accept', 'reject', 'counter', 'random'];
  }
}

module.exports = {
  AcceptStrategy,
  RejectStrategy,
  CounterStrategy,
  RandomStrategy,
  StrategyFactory
};
