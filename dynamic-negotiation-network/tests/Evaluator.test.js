/**
 * Unit tests for Evaluator class
 */

const Evaluator = require('../src/Evaluator');

describe('Evaluator', () => {
  describe('constructor', () => {
    it('should create evaluator with options', () => {
      const evaluator = new Evaluator({ customOption: true });
      
      expect(evaluator.options.customOption).toBe(true);
    });
  });

  describe('evaluate', () => {
    it('should evaluate negotiation result', () => {
      const evaluator = new Evaluator();
      
      const result = {
        success: true,
        status: 'completed',
        rounds: 5,
        duration: 1000,
        statistics: {
          totalOffers: 10,
          totalAcceptances: 5,
          totalRejections: 3,
          totalCounters: 2
        },
        agentResults: [
          { id: 'agent1', needsSatisfied: true, remainingNeeds: [] },
          { id: 'agent2', needsSatisfied: true, remainingNeeds: [] }
        ]
      };
      
      const evaluation = evaluator.evaluate(result);
      
      expect(evaluation.success).toBe(true);
      expect(evaluation.agreement).toBeDefined();
      expect(evaluation.metrics).toBeDefined();
      expect(evaluation.agentScores).toBeDefined();
    });

    it('should calculate agent scores', () => {
      const evaluator = new Evaluator();
      
      const result = {
        success: true,
        status: 'completed',
        rounds: 5,
        duration: 1000,
        statistics: {
          totalOffers: 10,
          totalAcceptances: 5,
          totalRejections: 3,
          totalCounters: 2
        },
        agentResults: [
          { id: 'agent1', needsSatisfied: true, remainingNeeds: [] }
        ]
      };
      
      const evaluation = evaluator.evaluate(result);
      
      expect(evaluation.agentScores.length).toBe(1);
      expect(evaluation.agentScores[0].id).toBe('agent1');
      expect(evaluation.agentScores[0].payoff).toBeDefined();
    });

    it('should handle undefined agentResults', () => {
      const evaluator = new Evaluator();
      
      const result = {
        success: true,
        status: 'completed',
        rounds: 5,
        duration: 1000,
        statistics: {
          totalOffers: 0,
          totalAcceptances: 0,
          totalRejections: 0,
          totalCounters: 0
        }
      };
      
      const evaluation = evaluator.evaluate(result);
      
      expect(evaluation.agentScores).toEqual([]);
    });
  });

  describe('calculatePayoff', () => {
    it('should calculate payoff for satisfied agent', () => {
      const evaluator = new Evaluator();
      
      const payoff = evaluator.calculatePayoff(
        { id: 'agent1', needsSatisfied: true, remainingNeeds: [] },
        { rounds: 5, statistics: { totalAcceptances: 5, totalOffers: 10 } }
      );
      
      expect(payoff).toBeGreaterThan(0);
    });

    it('should calculate payoff for unsatisfied agent', () => {
      const evaluator = new Evaluator();
      
      const payoff = evaluator.calculatePayoff(
        { id: 'agent1', needsSatisfied: false, remainingNeeds: ['x'] },
        { rounds: 10, statistics: { totalAcceptances: 1, totalOffers: 10 } }
      );
      
      expect(payoff).toBeGreaterThan(0);
    });

    it('should apply round penalty', () => {
      const evaluator = new Evaluator();
      
      const quickPayoff = evaluator.calculatePayoff(
        { id: 'agent1', needsSatisfied: true, remainingNeeds: [] },
        { rounds: 2, statistics: { totalAcceptances: 5, totalOffers: 10 } }
      );
      
      const slowPayoff = evaluator.calculatePayoff(
        { id: 'agent1', needsSatisfied: true, remainingNeeds: [] },
        { rounds: 20, statistics: { totalAcceptances: 5, totalOffers: 10 } }
      );
      
      expect(quickPayoff).toBeGreaterThan(slowPayoff);
    });

    it('should include efficiency bonus', () => {
      const evaluator = new Evaluator();
      
      const highEfficiency = evaluator.calculatePayoff(
        { id: 'agent1', needsSatisfied: true, remainingNeeds: [] },
        { rounds: 5, statistics: { totalAcceptances: 9, totalOffers: 10 } }
      );
      
      const lowEfficiency = evaluator.calculatePayoff(
        { id: 'agent1', needsSatisfied: true, remainingNeeds: [] },
        { rounds: 5, statistics: { totalAcceptances: 1, totalOffers: 10 } }
      );
      
      expect(highEfficiency).toBeGreaterThan(lowEfficiency);
    });
  });

  describe('isAgreement', () => {
    it('should return true when all agents satisfied', () => {
      const evaluator = new Evaluator();
      
      const result = {
        status: 'completed',
        agentResults: [
          { needsSatisfied: true, remainingNeeds: [] },
          { needsSatisfied: true, remainingNeeds: [] }
        ],
        statistics: { totalAcceptances: 5, totalOffers: 10 }
      };
      
      expect(evaluator.isAgreement(result)).toBe(true);
    });

    it('should return true with high acceptance rate', () => {
      const evaluator = new Evaluator();
      
      const result = {
        status: 'completed',
        agentResults: [
          { needsSatisfied: false, remainingNeeds: ['x'] }
        ],
        statistics: { totalAcceptances: 8, totalOffers: 10 }
      };
      
      expect(evaluator.isAgreement(result)).toBe(true);
    });

    it('should return false for timeout status', () => {
      const evaluator = new Evaluator();
      
      const result = {
        status: 'timeout',
        agentResults: [],
        statistics: { totalAcceptances: 0, totalOffers: 0 }
      };
      
      expect(evaluator.isAgreement(result)).toBe(false);
    });

    it('should return false for low acceptance rate', () => {
      const evaluator = new Evaluator();
      
      const result = {
        status: 'completed',
        agentResults: [
          { needsSatisfied: false, remainingNeeds: ['x'] }
        ],
        statistics: { totalAcceptances: 2, totalOffers: 10 }
      };
      
      expect(evaluator.isAgreement(result)).toBe(false);
    });
  });

  describe('getMetrics', () => {
    it('should calculate precision, recall, F1', () => {
      const evaluator = new Evaluator();
      
      const result = {
        agentResults: [
          { needsSatisfied: true, remainingNeeds: [] },
          { needsSatisfied: true, remainingNeeds: [] },
          { needsSatisfied: false, remainingNeeds: ['x'] }
        ],
        statistics: {
          totalOffers: 10,
          totalAcceptances: 5,
          totalRejections: 3,
          totalCounters: 2
        },
        status: 'completed'
      };
      
      const metrics = evaluator.getMetrics(result);
      
      expect(metrics.precision).toBeGreaterThanOrEqual(0);
      expect(metrics.recall).toBeGreaterThanOrEqual(0);
      expect(metrics.f1).toBeGreaterThanOrEqual(0);
    });

    it('should handle empty agentResults', () => {
      const evaluator = new Evaluator();
      
      const result = {
        agentResults: [],
        statistics: {
          totalOffers: 0,
          totalAcceptances: 0,
          totalRejections: 0,
          totalCounters: 0
        }
      };
      
      const metrics = evaluator.getMetrics(result);
      
      expect(metrics.precision).toBe(0);
      expect(metrics.recall).toBe(0);
      expect(metrics.f1).toBe(0);
    });

    it('should calculate efficiency metrics', () => {
      const evaluator = new Evaluator();
      
      const result = {
        agentResults: [
          { needsSatisfied: true, remainingNeeds: [] }
        ],
        statistics: {
          totalOffers: 10,
          totalAcceptances: 5,
          totalRejections: 3,
          totalCounters: 2
        },
        status: 'completed'
      };
      
      const metrics = evaluator.getMetrics(result);
      
      expect(metrics.efficiency).toBe(0.5); // 5/10
      expect(metrics.conflictRate).toBe(0.3); // 3/10
      expect(metrics.flexibility).toBe(0.2); // 2/10
    });

    it('should include success rate', () => {
      const evaluator = new Evaluator();
      
      const result = {
        agentResults: [
          { needsSatisfied: true, remainingNeeds: [] },
          { needsSatisfied: false, remainingNeeds: ['x'] }
        ],
        statistics: {
          totalOffers: 5,
          totalAcceptances: 3,
          totalRejections: 1,
          totalCounters: 1
        },
        status: 'completed'
      };
      
      const metrics = evaluator.getMetrics(result);
      
      expect(metrics.successRate).toBe(0.5); // 1/2 agents satisfied
    });
  });

  describe('compare', () => {
    it('should compare multiple results', () => {
      const evaluator = new Evaluator();
      
      const results = [
        {
          success: true,
          status: 'completed',
          rounds: 5,
          duration: 1000,
          statistics: { totalOffers: 10, totalAcceptances: 5, totalRejections: 3, totalCounters: 2 },
          agentResults: [{ id: 'a1', needsSatisfied: true, remainingNeeds: [] }]
        },
        {
          success: true,
          status: 'completed',
          rounds: 3,
          duration: 500,
          statistics: { totalOffers: 8, totalAcceptances: 6, totalRejections: 1, totalCounters: 1 },
          agentResults: [{ id: 'a1', needsSatisfied: true, remainingNeeds: [] }]
        }
      ];
      
      const comparison = evaluator.compare(results);
      
      expect(comparison.evaluations.length).toBe(2);
      expect(comparison.bestIndex).toBeDefined();
      expect(comparison.averages).toBeDefined();
      expect(comparison.ranking).toBeDefined();
    });

    it('should rank results by score', () => {
      const evaluator = new Evaluator();
      
      const results = [
        {
          success: true,
          status: 'completed',
          rounds: 10,
          duration: 2000,
          statistics: { totalOffers: 10, totalAcceptances: 2, totalRejections: 5, totalCounters: 3 },
          agentResults: [{ id: 'a1', needsSatisfied: true, remainingNeeds: [] }]
        },
        {
          success: true,
          status: 'completed',
          rounds: 2,
          duration: 200,
          statistics: { totalOffers: 10, totalAcceptances: 8, totalRejections: 1, totalCounters: 1 },
          agentResults: [{ id: 'a1', needsSatisfied: true, remainingNeeds: [] }]
        }
      ];
      
      const comparison = evaluator.compare(results);
      
      // Results should be ranked by score descending
      for (let i = 1; i < comparison.ranking.length; i++) {
        expect(comparison.ranking[i - 1].score).toBeGreaterThanOrEqual(comparison.ranking[i].score);
      }
    });
  });

  describe('generateReport', () => {
    it('should generate a formatted report', () => {
      const evaluator = new Evaluator();
      
      const result = {
        success: true,
        status: 'completed',
        rounds: 5,
        duration: 1000,
        statistics: {
          totalOffers: 10,
          totalAcceptances: 5,
          totalRejections: 3,
          totalCounters: 2,
          agreementRate: 0.5
        },
        agentResults: [
          { id: 'agent1', needsSatisfied: true, remainingNeeds: [] }
        ]
      };
      
      const report = evaluator.generateReport(result);
      
      expect(typeof report).toBe('string');
      expect(report).toContain('NEGOTIATION EVALUATION REPORT');
      expect(report).toContain('Status:');
      expect(report).toContain('OVERALL SCORE');
    });
  });

  describe('_calculateOverallScore', () => {
    it('should return 0 for empty agent scores', () => {
      const evaluator = new Evaluator();
      
      const score = evaluator._calculateOverallScore([], false);
      
      expect(score).toBe(0);
    });

    it('should calculate score from agent payoffs', () => {
      const evaluator = new Evaluator();
      
      const agentScores = [
        { id: 'a1', needsSatisfied: true, payoff: 10 },
        { id: 'a2', needsSatisfied: true, payoff: 10 }
      ];
      
      const score = evaluator._calculateOverallScore(agentScores, true);
      
      expect(score).toBeGreaterThan(0);
    });

    it('should add agreement bonus', () => {
      const evaluator = new Evaluator();
      
      const agentScores = [
        { id: 'a1', needsSatisfied: true, payoff: 10 }
      ];
      
      const withAgreement = evaluator._calculateOverallScore(agentScores, true);
      const withoutAgreement = evaluator._calculateOverallScore(agentScores, false);
      
      expect(withAgreement).toBeGreaterThan(withoutAgreement);
    });
  });

  describe('edge cases', () => {
    it('should handle zero offers', () => {
      const evaluator = new Evaluator();
      
      const result = {
        agentResults: [{ needsSatisfied: true, remainingNeeds: [] }],
        statistics: {
          totalOffers: 0,
          totalAcceptances: 0,
          totalRejections: 0,
          totalCounters: 0
        },
        status: 'completed'
      };
      
      const metrics = evaluator.getMetrics(result);
      
      expect(metrics.efficiency).toBe(0);
      expect(metrics.conflictRate).toBe(0);
    });

    it('should handle all agents satisfied', () => {
      const evaluator = new Evaluator();
      
      const result = {
        status: 'completed',
        agentResults: [
          { needsSatisfied: true, remainingNeeds: [] },
          { needsSatisfied: true, remainingNeeds: [] }
        ],
        statistics: { totalAcceptances: 5, totalOffers: 10 }
      };
      
      expect(evaluator.isAgreement(result)).toBe(true);
    });

    it('should handle single agent result', () => {
      const evaluator = new Evaluator();
      
      const result = {
        success: true,
        status: 'completed',
        rounds: 1,
        duration: 100,
        statistics: {
          totalOffers: 1,
          totalAcceptances: 1,
          totalRejections: 0,
          totalCounters: 0
        },
        agentResults: [
          { id: 'agent1', needsSatisfied: true, remainingNeeds: [] }
        ]
      };
      
      const evaluation = evaluator.evaluate(result);
      
      expect(evaluation.agentScores.length).toBe(1);
    });
  });
});
