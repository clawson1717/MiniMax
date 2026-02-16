/**
 * Evaluator
 * 
 * Evaluates negotiation outcomes and calculates metrics:
 * - Score negotiation outcomes
 * - Calculate payoff for each agent
 * - Check if agreement was reached
 * - Return precision, recall, F1 of negotiation success
 */

class Evaluator {
  /**
   * Create an Evaluator
   * @param {Object} options - Configuration options
   */
  constructor(options = {}) {
    this.options = options;
  }

  /**
   * Evaluate negotiation outcomes
   * @param {Object} negotiationResult - Result from NegotiationRunner
   * @returns {Object} - Evaluation with scores and metrics
   */
  evaluate(negotiationResult) {
    const {
      success,
      status,
      rounds,
      duration,
      statistics,
      agentResults
    } = negotiationResult;
    
    // Calculate individual agent scores
    const agentScores = agentResults && agentResults.map(agent => ({
      id: agent.id,
      needsSatisfied: agent.needsSatisfied,
      satisfactionRate: (agent.remainingNeeds && agent.remainingNeeds.length) === 0 ? 1.0 : 0.0,
      payoff: this.calculatePayoff(agent, negotiationResult)
    })) || [];
    
    // Determine if agreement was reached
    const agreement = this.isAgreement(negotiationResult);
    
    // Calculate metrics
    const metrics = this.getMetrics(negotiationResult);
    
    return {
      success,
      status,
      agreement,
      rounds,
      duration,
      statistics,
      agentScores,
      metrics,
      overallScore: this._calculateOverallScore(agentScores, agreement)
    };
  }

  /**
   * Calculate payoff for each agent
   * @param {Object} agentResult - Agent result from negotiation
   * @param {Object} negotiationResult - Full negotiation result
   * @returns {number} - Payoff value
   */
  calculatePayoff(agentResult, negotiationResult) {
    let payoff = 0;
    
    // Base payoff for participation
    payoff += 1.0;
    
    // Reward for needs satisfaction
    if (agentResult.needsSatisfied) {
      payoff += 10.0;
    } else {
      // Partial reward based on how many needs were satisfied
      const totalNeeds = agentResult.remainingNeeds.length + (agentResult.needsSatisfied ? 0 : 1);
      const satisfiedRatio = 1 - (agentResult.remainingNeeds.length / Math.max(totalNeeds, 1));
      payoff += 5.0 * satisfiedRatio;
    }
    
    // Bonus for quick agreement (less rounds = more bonus)
    const roundPenalty = Math.min(negotiationResult.rounds / 20, 1);
    payoff *= (1 - roundPenalty * 0.5);
    
    // Efficiency bonus for high acceptance rate
    const { totalAcceptances, totalOffers } = negotiationResult.statistics;
    if (totalOffers > 0) {
      const efficiency = totalAcceptances / totalOffers;
      payoff += efficiency * 3.0;
    }
    
    // Cooperation bonus (if agreement reached)
    if (this.isAgreement(negotiationResult)) {
      payoff += 2.0;
    }
    
    return Math.max(0, payoff);
  }

  /**
   * Check if agreement was reached
   * @param {Object} negotiationResult - Result from NegotiationRunner
   * @returns {boolean}
   */
  isAgreement(negotiationResult) {
    // Agreement is reached if:
    // 1. Status is 'completed' AND all agents have satisfied needs
    // 2. Status is 'completed' with high acceptance rate
    
    if (negotiationResult.status !== 'completed' && 
        negotiationResult.status !== 'running') {
      return false;
    }
    
    const { agentResults, statistics } = negotiationResult;
    
    // All agents satisfied
    const allSatisfied = agentResults.every(a => a.needsSatisfied);
    
    // High acceptance rate (> 50% of offers accepted)
    const { totalAcceptances, totalOffers } = statistics;
    const acceptanceRate = totalOffers > 0 ? totalAcceptances / totalOffers : 0;
    const highAcceptance = acceptanceRate > 0.5;
    
    return allSatisfied || highAcceptance;
  }

  /**
   * Get precision, recall, F1 of negotiation success
   * @param {Object} negotiationResult - Result from NegotiationRunner
   * @returns {Object} - Precision, recall, F1 scores
   */
  getMetrics(negotiationResult) {
    const { agentResults, statistics, status } = negotiationResult;
    
    // Handle undefined or empty agentResults
    if (!agentResults || !Array.isArray(agentResults) || agentResults.length === 0) {
      return {
        precision: 0,
        recall: 0,
        f1: 0,
        successRate: 0,
        agreement: false,
        efficiency: 0,
        conflictRate: 0,
        flexibility: 0,
        counts: {
          totalAgents: 0,
          totalOffers: 0,
          totalAcceptances: 0,
          totalRejections: 0,
          totalCounters: 0
        }
      };
    }
    
    // Define "relevant" as agents that should succeed (have needs)
    // Define "retrieved" as agents that actually succeeded
    
    // True Positives: Agents that needed something AND got it
    const truePositives = agentResults.filter(
      a => (a.remainingNeeds?.length || 0) === 0 && !a.needsSatisfied === false
    ).length;
    
    // False Positives: Agents that got what they needed but didn't need it
    const falsePositives = 0; // Can't have this in our model
    
    // False Negatives: Agents that needed something but didn't get it
    const falseNegatives = agentResults.filter(
      a => (a.remainingNeeds?.length || 0) > 0
    ).length;
    
    // True Negatives: Agents that didn't need anything AND didn't get anything
    const trueNegatives = agentResults.filter(
      a => (a.remainingNeeds?.length || 0) === 0 && a.needsSatisfied
    ).length;
    
    // Calculate precision
    const precision = (truePositives + trueNegatives) > 0
      ? (truePositives + trueNegatives) / agentResults.length
      : 0;
    
    // Calculate recall (based on needs satisfaction)
    const totalWithNeeds = agentResults.filter(a => a.remainingNeeds.length > 0).length;
    const recall = totalWithNeeds > 0
      ? (totalWithNeeds - falseNegatives) / totalWithNeeds
      : 1.0;
    
    // Calculate F1
    const f1 = (precision + recall) > 0
      ? (2 * precision * recall) / (precision + recall)
      : 0;
    
    // Alternative metrics based on negotiation success
    const agreement = this.isAgreement(negotiationResult);
    const successRate = agentResults.filter(a => a.needsSatisfied).length / Math.max(agentResults.length, 1);
    
    // Efficiency metrics
    const { totalOffers, totalAcceptances, totalRejections, totalCounters } = statistics;
    const efficiency = totalOffers > 0 ? totalAcceptances / totalOffers : 0;
    const conflictRate = totalOffers > 0 ? totalRejections / totalOffers : 0;
    const flexibility = totalOffers > 0 ? totalCounters / totalOffers : 0;
    
    return {
      // Core metrics
      precision: Math.min(1, precision),
      recall: Math.min(1, recall),
      f1: Math.min(1, f1),
      
      // Negotiation-specific metrics
      successRate,
      agreement,
      efficiency,
      conflictRate,
      flexibility,
      
      // Counts
      counts: {
        truePositives,
        falsePositives,
        falseNegatives,
        trueNegatives,
        totalAgents: agentResults.length,
        totalOffers,
        totalAcceptances,
        totalRejections,
        totalCounters
      }
    };
  }

  /**
   * Compare multiple negotiation results
   * @param {Object[]} results - Array of negotiation results
   * @returns {Object} - Comparison analysis
   */
  compare(results) {
    const evaluations = results.map(r => this.evaluate(r));
    
    // Find best result
    let bestIndex = 0;
    let bestScore = -Infinity;
    
    evaluations.forEach((eval_, index) => {
      if (eval_.overallScore > bestScore) {
        bestScore = eval_.overallScore;
        bestIndex = index;
      }
    });
    
    // Calculate averages
    const avgRounds = evaluations.reduce((sum, e) => sum + e.rounds, 0) / evaluations.length;
    const avgDuration = evaluations.reduce((sum, e) => sum + (e.duration || 0), 0) / evaluations.length;
    const avgScore = evaluations.reduce((sum, e) => sum + e.overallScore, 0) / evaluations.length;
    const avgF1 = evaluations.reduce((sum, e) => sum + e.metrics.f1, 0) / evaluations.length;
    
    return {
      evaluations,
      bestIndex,
      bestResult: evaluations[bestIndex],
      averages: {
        rounds: avgRounds,
        duration: avgDuration,
        score: avgScore,
        f1: avgF1
      },
      ranking: evaluations
        .map((e, i) => ({ index: i, score: e.overallScore }))
        .sort((a, b) => b.score - a.score)
    };
  }

  /**
   * Generate detailed report
   * @param {Object} negotiationResult - Result from NegotiationRunner
   * @returns {string} - Formatted report
   */
  generateReport(negotiationResult) {
    const evaluation = this.evaluate(negotiationResult);
    
    let report = '='.repeat(60) + '\n';
    report += 'NEGOTIATION EVALUATION REPORT\n';
    report += '='.repeat(60) + '\n\n';
    
    report += `Status: ${evaluation.status}\n`;
    report += `Agreement Reached: ${evaluation.agreement ? 'Yes' : 'No'}\n`;
    report += `Total Rounds:}\n`;
    report += `Duration: ${evaluation.duration}ms\n\n`;
    
    report += 'STATISTICS:\n';
    report += `- Total Offers: ${evaluation.statistics.totalOffers}\n`;
    report += `- Acceptances: ${evaluation.statistics.totalAcceptances}\n`;
    report += `- Rejections: ${evaluation.statistics.totalRejections}\n`;
    report += `- Counter-offers: ${evaluation.statistics.totalCounters}\n`;
    report += `- Agreement Rate: ${(evaluation.statistics.agreementRate * 100).toFixed(1)}%\n\n`;
    
    report += 'AGENT RESULTS:\n';
    for (const agent of evaluation.agentScores) {
      report += `\n${agent.id}:\n`;
      report += `  Needs Satisfied: ${agent.needsSatisfied ? 'Yes' : 'No'}\n`;
      report += `  Payoff: ${agent.payoff.toFixed(2)}\n`;
      report += `  Satisfaction Rate: ${(agent.satisfactionRate * 100).toFixed(1)}%\n`;
    }
    
    report += '\nMETRICS:\n';
    report += `- Precision: ${(evaluation.metrics.precision * 100).toFixed(1)}%\n`;
    report += `- Recall: ${(evaluation.metrics.recall * 100).toFixed(1)}%\n`;
    report += `- F1 Score: ${(evaluation.metrics.f1 * 100).toFixed(1)}%\n`;
    report += `- Success Rate: ${(evaluation.metrics.successRate * 100).toFixed(1)}%\n`;
    report += `- Efficiency: ${(evaluation.metrics.efficiency * 100).toFixed(1)}%\n`;
    report += `- Conflict Rate: ${(evaluation.metrics.conflictRate * 100).toFixed(1)}%\n`;
    
    report += '\n' + '='.repeat(60) + '\n';
    report += `OVERALL SCORE: ${evaluation.overallScore.toFixed(2)}\n`;
    report += '='.repeat(60) + '\n';
    
    return report;
  }

  /**
   * Calculate overall score from agent scores
   * @private
   */
  _calculateOverallScore(agentScores, agreement) {
    if (!agentScores || agentScores.length === 0) {
      return 0;
    }
    
    // Average payoff
    const avgPayoff = agentScores.reduce((sum, a) => sum + a.payoff, 0) / agentScores.length;
    
    // Agreement bonus
    const agreementBonus = agreement ? 5 : 0;
    
    // Satisfaction bonus
    const satisfactionBonus = agentScores.filter(a => a.needsSatisfied).length * 2;
    
    return avgPayoff + agreementBonus + satisfactionBonus;
  }
}

module.exports = Evaluator;
