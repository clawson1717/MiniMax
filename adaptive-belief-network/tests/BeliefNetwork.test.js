/**
 * Unit tests for BeliefNetwork class
 */

const { Belief, BeliefNetwork } = require('../src/index');

describe('BeliefNetwork', () => {
  let network;

  beforeEach(() => {
    network = new BeliefNetwork('test-agent');
  });

  describe('constructor', () => {
    test('should create network with empty beliefs', () => {
      expect(network.agentId).toBe('test-agent');
      expect(network.beliefs.size).toBe(0);
      expect(network.dependencies.size).toBe(0);
    });
  });

  describe('addBelief', () => {
    test('should add a belief without dependencies', () => {
      const belief = network.addBelief('Sky is blue', 0.9, 'Observation');
      
      expect(belief).toBeInstanceOf(Belief);
      expect(network.beliefs.size).toBe(1);
      expect(network.getBelief('Sky is blue')).toBe(belief);
    });

    test('should add belief with dependencies', () => {
      network.addBelief('Sky is blue', 0.9, 'Observation');
      const belief = network.addBelief('Good weather', 0.8, 'Blue sky', ['Sky is blue']);
      
      expect(belief).toBeInstanceOf(Belief);
      expect(network.dependencies.get('Good weather')).toContain('Sky is blue');
    });

    test('should throw error for duplicate proposition', () => {
      network.addBelief('Test', 0.5, 'Reason');
      
      expect(() => {
        network.addBelief('Test', 0.7, 'Another reason');
      }).toThrow('already exists');
    });

    test('should throw error for non-existent dependency', () => {
      expect(() => {
        network.addBelief('Test', 0.5, 'Reason', ['NonExistent']);
      }).toThrow('does not exist');
    });

    test('should log update on add', () => {
      network.addBelief('Test', 0.5, 'Reason');
      
      const log = network.getUpdateLog();
      expect(log[0].type).toBe('ADD');
      expect(log[0].proposition).toBe('Test');
    });
  });

  describe('updateBelief', () => {
    test('should update belief confidence', () => {
      network.addBelief('Test', 0.5, 'Reason');
      const updated = network.updateBelief('Test', 0.8, 'New reason');
      
      expect(network.getBelief('Test').confidence).toBe(0.8);
      expect(updated).toHaveLength(1);
    });

    test('should propagate to dependent beliefs', () => {
      network.addBelief('Sky is blue', 0.9, 'Observation');
      network.addBelief('Good weather', 0.8, 'Blue sky', ['Sky is blue']);
      
      network.updateBelief('Sky is blue', 0.5, 'Clouds');
      
      expect(network.getBelief('Good weather').confidence).toBe(0.5);
    });

    test('should throw error for non-existent belief', () => {
      expect(() => {
        network.updateBelief('NonExistent', 0.5, 'Reason');
      }).toThrow('does not exist');
    });
  });

  describe('getBelief', () => {
    test('should return belief by proposition', () => {
      network.addBelief('Test', 0.5, 'Reason');
      
      const belief = network.getBelief('Test');
      expect(belief).toBeInstanceOf(Belief);
      expect(belief.confidence).toBe(0.5);
    });

    test('should return undefined for non-existent belief', () => {
      const belief = network.getBelief('NonExistent');
      expect(belief).toBeUndefined();
    });
  });

  describe('getDependent', () => {
    test('should return beliefs that depend on given proposition', () => {
      network.addBelief('A', 0.9, 'Reason');
      network.addBelief('B', 0.8, 'Depends on A', ['A']);
      network.addBelief('C', 0.7, 'Depends on A', ['A']);
      
      const dependents = network.getDependent('A');
      
      expect(dependents).toHaveLength(2);
      expect(dependents.map(b => b.proposition)).toContain('B');
      expect(dependents.map(b => b.proposition)).toContain('C');
    });
  });

  describe('getDependencies', () => {
    test('should return beliefs that given proposition depends on', () => {
      network.addBelief('A', 0.9, 'Reason');
      network.addBelief('B', 0.8, 'Depends on A', ['A']);
      
      const deps = network.getDependencies('B');
      
      expect(deps).toHaveLength(1);
      expect(deps[0].proposition).toBe('A');
    });
  });

  describe('removeBelief', () => {
    test('should remove belief from network', () => {
      network.addBelief('Test', 0.5, 'Reason');
      
      const result = network.removeBelief('Test');
      
      expect(result).toBe(true);
      expect(network.getBelief('Test')).toBeUndefined();
    });

    test('should return false for non-existent belief', () => {
      const result = network.removeBelief('NonExistent');
      expect(result).toBe(false);
    });

    test('should update dependency references', () => {
      network.addBelief('A', 0.9, 'Reason');
      network.addBelief('B', 0.8, 'Depends on A', ['A']);
      
      network.removeBelief('A');
      
      expect(network.getDependencies('B')).toHaveLength(0);
    });
  });

  describe('getAllBeliefs', () => {
    test('should return all beliefs', () => {
      network.addBelief('A', 0.9, 'Reason');
      network.addBelief('B', 0.8, 'Reason');
      
      const beliefs = network.getAllBeliefs();
      
      expect(beliefs).toHaveLength(2);
    });
  });

  describe('toJSON', () => {
    test('should serialize network to JSON', () => {
      network.addBelief('Test', 0.5, 'Reason');
      
      const json = network.toJSON();
      
      expect(json.agentId).toBe('test-agent');
      expect(json.beliefs).toHaveLength(1);
      expect(json.beliefs[0].proposition).toBe('Test');
    });
  });

  describe('getStats', () => {
    test('should return network statistics', () => {
      network.addBelief('A', 0.9, 'Reason');
      network.addBelief('B', 0.8, 'Reason');
      
      const stats = network.getStats();
      
      expect(stats.totalBeliefs).toBe(2);
      expect(stats.avgConfidence).toBeCloseTo(0.85);
    });
  });

  describe('clearUpdateLog', () => {
    test('should clear update log', () => {
      network.addBelief('Test', 0.5, 'Reason');
      network.updateBelief('Test', 0.8, 'New reason');
      
      network.clearUpdateLog();
      
      expect(network.getUpdateLog()).toHaveLength(0);
    });
  });
});
