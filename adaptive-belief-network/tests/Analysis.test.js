/**
 * Unit tests for Analysis module
 */

const { Agent, BeliefNetwork, Analysis, Belief } = require('../src/index');

describe('Analysis', () => {
  describe('compareStates', () => {
    test('should compare two belief networks', () => {
      const network1 = new BeliefNetwork('agent-1');
      const network2 = new BeliefNetwork('agent-2');
      
      network1.addBelief('A', 0.8, 'Reason');
      network1.addBelief('B', 0.7, 'Reason');
      
      network2.addBelief('A', 0.6, 'Reason');
      network2.addBelief('B', 0.9, 'Reason');
      network2.addBelief('C', 0.5, 'Reason');
      
      const result = Analysis.compareStates(network1, network2);
      
      expect(result.commonBeliefs).toBe(2);
      expect(result.onlyInFirst).toHaveLength(0);
      expect(result.onlyInSecond).toHaveLength(1);
      expect(result.averageDifference).toBeCloseTo(0.2);
    });

    test('should calculate confidence differences', () => {
      const network1 = new BeliefNetwork('agent-1');
      const network2 = new BeliefNetwork('agent-2');
      
      network1.addBelief('A', 0.8, 'Reason');
      network2.addBelief('A', 0.4, 'Reason');
      
      const result = Analysis.compareStates(network1, network2);
      
      expect(result.confidenceDifferences[0].difference).toBe(0.4);
    });
  });

  describe('computeDivergence', () => {
    test('should compute multiple divergence metrics', () => {
      const network1 = new BeliefNetwork('agent-1');
      const network2 = new BeliefNetwork('agent-2');
      
      network1.addBelief('A', 0.8, 'Reason');
      network1.addBelief('B', 0.7, 'Reason');
      
      network2.addBelief('A', 0.6, 'Reason');
      network2.addBelief('B', 0.9, 'Reason');
      
      const result = Analysis.computeDivergence(network1, network2);
      
      expect(result.euclideanDistance).toBeGreaterThan(0);
      expect(result.manhattanDistance).toBeGreaterThan(0);
      expect(result.cosineSimilarity).toBeDefined();
      expect(result.commonBeliefs).toBe(2);
    });

    test('should handle empty networks', () => {
      const network1 = new BeliefNetwork('agent-1');
      const network2 = new BeliefNetwork('agent-2');
      
      const result = Analysis.computeDivergence(network1, network2);
      
      expect(result.jensenShannonDivergence).toBe(0);
      expect(result.euclideanDistance).toBe(0);
    });
  });

  describe('getConvergenceRate', () => {
    test('should calculate convergence rate from history', () => {
      const belief = new Belief('Test', 0.5, 'Initial');
      belief.update(0.6, 'Update 1');
      belief.update(0.65, 'Update 2');
      belief.update(0.68, 'Update 3');
      
      const result = Analysis.getConvergenceRate(belief);
      
      expect(result.changeCount).toBe(3);
      expect(result.totalChange).toBeCloseTo(0.18);
      expect(result.averageChange).toBeCloseTo(0.06);
    });

    test('should detect converging beliefs', () => {
      const belief = new Belief('Test', 0.5, 'Initial');
      belief.update(0.8, 'Update 1');
      belief.update(0.85, 'Update 2');
      belief.update(0.86, 'Update 3');
      
      const result = Analysis.getConvergenceRate(belief);
      
      expect(result.isConverging).toBe(true);
      expect(result.rate).toBeGreaterThan(0);
    });

    test('should handle single-point history', () => {
      const belief = new Belief('Test', 0.5, 'Initial');
      
      const result = Analysis.getConvergenceRate(belief);
      
      expect(result.rate).toBe(0);
      expect(result.volatility).toBe(0);
    });
  });

  describe('analyzeStability', () => {
    test('should analyze network stability', () => {
      const network = new BeliefNetwork('agent-1');
      network.addBelief('A', 0.8, 'Reason');
      network.addBelief('B', 0.7, 'Reason');
      
      const result = Analysis.analyzeStability(network);
      
      expect(result.averageStability).toBeDefined();
      expect(result.beliefMetrics).toHaveLength(2);
    });

    test('should identify most stable and volatile beliefs', () => {
      const network = new BeliefNetwork('agent-1');
      network.addBelief('Stable', 0.8, 'Reason');
      network.addBelief('Volatile', 0.5, 'Reason');
      
      // Update stable belief once
      network.updateBelief('Stable', 0.81, 'Small change');
      
      // Update volatile belief multiple times
      network.updateBelief('Volatile', 0.9, 'Big change');
      network.updateBelief('Volatile', 0.2, 'Another big change');
      
      const result = Analysis.analyzeStability(network);
      
      expect(result.mostStable).toBeDefined();
      expect(result.mostVolatile).toBeDefined();
    });
  });

  describe('analyzeAgreement', () => {
    test('should analyze agreement between agents', () => {
      const agent1 = new Agent('agent-1', 'Agent One');
      const agent2 = new Agent('agent-2', 'Agent Two');
      
      agent1.addBelief('Test', 0.8, 'Reason');
      agent2.addBelief('Test', 0.85, 'Reason');
      
      const result = Analysis.analyzeAgreement([agent1, agent2]);
      
      expect(result.averageAgreement).toBeDefined();
      expect(result.metrics).toHaveLength(1);
    });

    test('should calculate agreement scores', () => {
      const agent1 = new Agent('agent-1', 'Agent One');
      const agent2 = new Agent('agent-2', 'Agent Two');
      
      agent1.addBelief('Test', 0.8, 'Reason');
      agent2.addBelief('Test', 0.8, 'Reason');
      
      const result = Analysis.analyzeAgreement([agent1, agent2]);
      
      expect(result.metrics[0].agreementScore).toBe(1); // Perfect agreement
    });
  });

  describe('analyzeDisagreement', () => {
    test('should identify disagreements', () => {
      const agent1 = new Agent('agent-1', 'Agent One');
      const agent2 = new Agent('agent-2', 'Agent Two');
      
      agent1.addBelief('Test', 0.9, 'Reason');
      agent2.addBelief('Test', 0.1, 'Reason');
      
      const result = Analysis.analyzeDisagreement([agent1, agent2]);
      
      expect(result.disagreementCount).toBe(1);
      expect(result.averageDisagreement).toBeGreaterThan(0);
    });
  });

  describe('getBeliefEvolution', () => {
    test('should track belief evolution', () => {
      const belief = new Belief('Test', 0.5, 'Initial');
      belief.update(0.6, 'Update 1');
      belief.update(0.7, 'Update 2');
      
      const result = Analysis.getBeliefEvolution(belief);
      
      expect(result.evolution).toHaveLength(3);
      expect(result.trend).toBeDefined();
      expect(result.initialConfidence).toBe(0.5);
      expect(result.finalConfidence).toBe(0.7);
    });

    test('should detect increasing trend', () => {
      const belief = new Belief('Test', 0.3, 'Initial');
      belief.update(0.5, 'Update 1');
      belief.update(0.7, 'Update 2');
      
      const result = Analysis.getBeliefEvolution(belief);
      
      expect(result.trend).toBe('increasing');
    });

    test('should detect decreasing trend', () => {
      const belief = new Belief('Test', 0.8, 'Initial');
      belief.update(0.6, 'Update 1');
      belief.update(0.4, 'Update 2');
      
      const result = Analysis.getBeliefEvolution(belief);
      
      expect(result.trend).toBe('decreasing');
    });
  });

  describe('analyzeNetwork', () => {
    test('should provide comprehensive network analysis', () => {
      const network = new BeliefNetwork('agent-1');
      network.addBelief('A', 0.8, 'Reason');
      network.addBelief('B', 0.7, 'Reason');
      
      const result = Analysis.analyzeNetwork(network);
      
      expect(result.stats).toBeDefined();
      expect(result.stability).toBeDefined();
      expect(result.confidenceDistribution).toBeDefined();
      expect(result.networkComplexity).toBeDefined();
    });
  });
});
