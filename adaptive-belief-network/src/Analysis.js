/**
 * Analysis Module
 * 
 * Provides tools for analyzing and comparing belief network states.
 * Includes metrics for belief stability, agreement, disagreement,
 * and convergence rate analysis.
 * 
 * Methods:
 * - compareStates(): Compare two belief network states
 * - computeDivergence(): Compute divergence between belief states
 * - getConvergenceRate(): Calculate how quickly beliefs converge
 */

const { Belief } = require('./Belief');
const { BeliefNetwork } = require('./BeliefNetwork');

class Analysis {
  /**
   * Compare two belief network states
   * @param {BeliefNetwork} network1 - First network state
   * @param {BeliefNetwork} network2 - Second network state
   * @returns {Object} - Comparison results
   */
  static compareStates(network1, network2) {
    const beliefs1 = network1.getAllBeliefs();
    const beliefs2 = network2.getAllBeliefs();

    // Create maps for easy lookup
    const map1 = new Map(beliefs1.map(b => [b.proposition, b]));
    const map2 = new Map(beliefs2.map(b => [b.proposition, b]));

    // Find common, only-in-first, and only-in-second beliefs
    const commonPropositions = [];
    const onlyInFirst = [];
    const onlyInSecond = [];

    for (const prop of map1.keys()) {
      if (map2.has(prop)) {
        commonPropositions.push(prop);
      } else {
        onlyInFirst.push(prop);
      }
    }

    for (const prop of map2.keys()) {
      if (!map1.has(prop)) {
        onlyInSecond.push(prop);
      }
    }

    // Calculate confidence differences for common beliefs
    const confidenceDifferences = [];
    let totalDifference = 0;

    for (const prop of commonPropositions) {
      const b1 = map1.get(prop);
      const b2 = map2.get(prop);
      const diff = Math.abs(b1.confidence - b2.confidence);
      confidenceDifferences.push({
        proposition: prop,
        confidence1: b1.confidence,
        confidence2: b2.confidence,
        difference: diff
      });
      totalDifference += diff;
    }

    const avgDifference = commonPropositions.length > 0 
      ? totalDifference / commonPropositions.length 
      : 0;

    return {
      commonBeliefs: commonPropositions.length,
      onlyInFirst: onlyInFirst.length,
      onlyInSecond: onlyInSecond.length,
      commonPropositions,
      onlyInFirst,
      onlyInSecond,
      confidenceDifferences,
      averageDifference: avgDifference,
      maxDifference: confidenceDifferences.length > 0 
        ? Math.max(...confidenceDifferences.map(d => d.difference))
        : 0
    };
  }

  /**
   * Compute divergence between two belief states
   * Uses multiple metrics: Jensen-Shannon divergence, Euclidean distance, and Manhattan distance
   * @param {BeliefNetwork} network1 - First network state
   * @param {BeliefNetwork} network2 - Second network state
   * @returns {Object} - Divergence metrics
   */
  static computeDivergence(network1, network2) {
    const beliefs1 = network1.getAllBeliefs();
    const beliefs2 = network2.getAllBeliefs();

    // Create maps for common propositions
    const map1 = new Map(beliefs1.map(b => [b.proposition, b]));
    const map2 = new Map(beliefs2.map(b => [b.proposition, b]));

    // Find common propositions
    const commonPropositions = [];
    for (const prop of map1.keys()) {
      if (map2.has(prop)) {
        commonPropositions.push(prop);
      }
    }

    if (commonPropositions.length === 0) {
      return {
        jensenShannonDivergence: 0,
        euclideanDistance: 0,
        manhattanDistance: 0,
        cosineSimilarity: 0,
        commonBeliefs: 0
      };
    }

    // Extract confidence values
    const confidences1 = commonPropositions.map(p => map1.get(p).confidence);
    const confidences2 = commonPropositions.map(p => map2.get(p).confidence);

    // Jensen-Shannon Divergence
    const jsDivergence = Analysis._jensenShannon(confidences1, confidences2);

    // Euclidean Distance
    const euclideanDistance = Math.sqrt(
      confidences1.reduce((sum, c1, i) => sum + Math.pow(c1 - confidences2[i], 2), 0)
    );

    // Manhattan Distance
    const manhattanDistance = confidences1.reduce((sum, c1, i) => 
      sum + Math.abs(c1 - confidences2[i]), 0
    );

    // Cosine Similarity
    const cosineSimilarity = Analysis._cosineSimilarity(confidences1, confidences2);

    return {
      jensenShannonDivergence: jsDivergence,
      euclideanDistance,
      manhattanDistance,
      cosineSimilarity,
      commonBeliefs: commonPropositions.length
    };
  }

  /**
   * Calculate convergence rate based on belief history
   * @param {Belief} belief - The belief to analyze
   * @returns {Object} - Convergence metrics
   */
  static getConvergenceRate(belief) {
    const history = belief.getHistory();
    
    if (history.length < 2) {
      return {
        rate: 0,
        isConverging: false,
        volatility: 0,
        totalChange: 0,
        averageChange: 0
      };
    }

    // Calculate total change
    let totalChange = 0;
    const changes = [];
    
    for (let i = 1; i < history.length; i++) {
      const change = Math.abs(history[i].confidence - history[i-1].confidence);
      changes.push(change);
      totalChange += change;
    }

    const averageChange = totalChange / changes.length;

    // Calculate volatility (standard deviation of changes)
    const meanChange = averageChange;
    const variance = changes.reduce((sum, c) => sum + Math.pow(c - meanChange, 2), 0) / changes.length;
    const volatility = Math.sqrt(variance);

    // Determine if converging (decreasing change magnitude)
    const recentChanges = changes.slice(-Math.min(3, changes.length));
    const olderChanges = changes.slice(0, Math.min(3, changes.length - 1));
    
    let rate = 0;
    let isConverging = false;

    if (olderChanges.length > 0 && recentChanges.length > 0) {
      const avgRecent = recentChanges.reduce((a, b) => a + b, 0) / recentChanges.length;
      const avgOlder = olderChanges.reduce((a, b) => a + b, 0) / olderChanges.length;
      
      if (avgOlder > 0) {
        rate = (avgOlder - avgRecent) / avgOlder;
        isConverging = rate > 0.1; // Consider converging if rate improved by > 10%
      }
    }

    return {
      rate,
      isConverging,
      volatility,
      totalChange,
      averageChange,
      changeCount: changes.length,
      finalConfidence: history[history.length - 1].confidence,
      initialConfidence: history[0].confidence
    };
  }

  /**
   * Analyze belief stability across a network
   * @param {BeliefNetwork} network - The network to analyze
   * @returns {Object} - Stability metrics
   */
  static analyzeStability(network) {
    const beliefs = network.getAllBeliefs();
    const stabilityMetrics = beliefs.map(belief => {
      const history = belief.getHistory();
      const confidenceChanges = [];
      
      for (let i = 1; i < history.length; i++) {
        confidenceChanges.push(Math.abs(history[i].confidence - history[i-1].confidence));
      }

      const totalChange = confidenceChanges.reduce((a, b) => a + b, 0);
      const avgChange = confidenceChanges.length > 0 ? totalChange / confidenceChanges.length : 0;
      
      // Calculate stability score (1 = perfectly stable, 0 = highly volatile)
      const stabilityScore = Math.max(0, 1 - avgChange * 10);

      return {
        proposition: belief.proposition,
        currentConfidence: belief.confidence,
        totalChange,
        averageChange: avgChange,
        updateCount: history.length,
        stabilityScore
      };
    });

    const avgStability = stabilityMetrics.length > 0
      ? stabilityMetrics.reduce((sum, m) => sum + m.stabilityScore, 0) / stabilityMetrics.length
      : 1;

    return {
      averageStability: avgStability,
      mostStable: stabilityMetrics.sort((a, b) => b.stabilityScore - a.stabilityScore)[0] || null,
      mostVolatile: stabilityMetrics.sort((a, b) => a.stabilityScore - b.stabilityScore)[0] || null,
      beliefMetrics: stabilityMetrics
    };
  }

  /**
   * Analyze agreement between agents
   * @param {Agent[]} agents - Array of agents to compare
   * @returns {Object} - Agreement metrics
   */
  static analyzeAgreement(agents) {
    // Get all unique propositions
    const propositionMap = new Map();
    
    for (const agent of agents) {
      for (const belief of agent.getBeliefs()) {
        if (!propositionMap.has(belief.proposition)) {
          propositionMap.set(belief.proposition, []);
        }
        propositionMap.get(belief.proposition).push({
          agentId: agent.id,
          confidence: belief.confidence
        });
      }
    }

    const agreementMetrics = [];

    for (const [proposition, agentBeliefs] of propositionMap) {
      if (agentBeliefs.length < 2) continue;

      const confidences = agentBeliefs.map(ab => ab.confidence);
      const mean = confidences.reduce((a, b) => a + b, 0) / confidences.length;
      const variance = confidences.reduce((sum, c) => sum + Math.pow(c - mean, 2), 0) / confidences.length;
      const stdDev = Math.sqrt(variance);

      // Agreement score: 1 = perfect agreement, 0 = maximum disagreement
      const agreementScore = Math.max(0, 1 - stdDev * 2);

      agreementMetrics.push({
        proposition,
        agentCount: agentBeliefs.length,
        meanConfidence: mean,
        stdDev,
        agreementScore,
        agents: agentBeliefs
      });
    }

    const avgAgreement = agreementMetrics.length > 0
      ? agreementMetrics.reduce((sum, m) => sum + m.agreementScore, 0) / agreementMetrics.length
      : 1;

    return {
      averageAgreement: avgAgreement,
      highAgreement: agreementMetrics.filter(m => m.agreementScore > 0.7),
      lowAgreement: agreementMetrics.filter(m => m.agreementScore < 0.3),
      metrics: agreementMetrics
    };
  }

  /**
   * Analyze disagreement between agents
   * @param {Agent[]} agents - Array of agents to compare
   * @returns {Object} - Disagreement metrics
   */
  static analyzeDisagreement(agents) {
    const agreement = Analysis.analyzeAgreement(agents);
    
    const disagreementMetrics = agreement.metrics
      .filter(m => m.agreementScore < 0.5)
      .sort((a, b) => a.agreementScore - b.agreementScore);

    return {
      disagreementCount: disagreementMetrics.length,
      averageDisagreement: disagreementMetrics.length > 0
        ? 1 - disagreementMetrics.reduce((sum, m) => sum + m.agreementScore, 0) / disagreementMetrics.length
        : 0,
      mostDisputed: disagreementMetrics.slice(0, 5),
      metrics: disagreementMetrics
    };
  }

  /**
   * Get belief evolution over time
   * @param {Belief} belief - The belief to analyze
   * @returns {Object} - Evolution data
   */
  static getBeliefEvolution(belief) {
    const history = belief.getHistory();
    
    if (history.length === 0) {
      return {
        evolution: [],
        trend: 'stable',
        trendStrength: 0
      };
    }

    // Calculate trend using linear regression
    let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
    const n = history.length;

    for (let i = 0; i < n; i++) {
      sumX += i;
      sumY += history[i].confidence;
      sumXY += i * history[i].confidence;
      sumX2 += i * i;
    }

    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    
    let trend = 'stable';
    let trendStrength = Math.abs(slope);
    
    if (slope > 0.01) {
      trend = 'increasing';
    } else if (slope < -0.01) {
      trend = 'decreasing';
    }

    return {
      evolution: history.map((h, i) => ({
        index: i,
        confidence: h.confidence,
        justification: h.justification,
        timestamp: h.timestamp
      })),
      trend,
      trendStrength,
      slope,
      initialConfidence: history[0].confidence,
      finalConfidence: history[history.length - 1].confidence,
      totalChange: Math.abs(history[history.length - 1].confidence - history[0].confidence)
    };
  }

  /**
   * Get comprehensive network analysis
   * @param {BeliefNetwork} network - The network to analyze
   * @returns {Object} - Comprehensive analysis
   */
  static analyzeNetwork(network) {
    const beliefs = network.getAllBeliefs();
    const stats = network.getStats();
    const stability = Analysis.analyzeStability(network);

    // Calculate confidence distribution
    const confidenceDistribution = {
      veryLow: beliefs.filter(b => b.confidence < 0.2).length,
      low: beliefs.filter(b => b.confidence >= 0.2 && b.confidence < 0.4).length,
      medium: beliefs.filter(b => b.confidence >= 0.4 && b.confidence < 0.6).length,
      high: beliefs.filter(b => b.confidence >= 0.6 && b.confidence < 0.8).length,
      veryHigh: beliefs.filter(b => b.confidence >= 0.8).length
    };

    // Calculate network complexity
    let dependencyCount = 0;
    for (const deps of network.dependencies.values()) {
      dependencyCount += deps.size;
    }

    return {
      stats,
      stability,
      confidenceDistribution,
      networkComplexity: {
        beliefCount: beliefs.length,
        dependencyCount,
        averageDependencies: beliefs.length > 0 ? dependencyCount / beliefs.length : 0
      }
    };
  }

  // Helper: Jensen-Shannon Divergence
  static _jensenShannon(p, q) {
    // Convert to probability distributions
    const n = p.length;
    if (n === 0) return 0;

    const m = p.map((pi, i) => (pi + q[i]) / 2);
    
    let kl1 = 0, kl2 = 0;
    for (let i = 0; i < n; i++) {
      if (p[i] > 0) kl1 += p[i] * Math.log(p[i] / m[i]);
      if (q[i] > 0) kl2 += q[i] * Math.log(q[i] / m[i]);
    }

    return (kl1 + kl2) / 2;
  }

  // Helper: Cosine Similarity
  static _cosineSimilarity(a, b) {
    const dotProduct = a.reduce((sum, ai, i) => sum + ai * b[i], 0);
    const magnitudeA = Math.sqrt(a.reduce((sum, ai) => sum + ai * ai, 0));
    const magnitudeB = Math.sqrt(b.reduce((sum, bi) => sum + bi * bi, 0));
    
    if (magnitudeA === 0 || magnitudeB === 0) return 0;
    return dotProduct / (magnitudeA * magnitudeB);
  }
}

module.exports = { Analysis };
