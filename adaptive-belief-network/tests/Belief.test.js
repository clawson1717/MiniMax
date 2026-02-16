/**
 * Unit tests for Belief class
 */

const { Belief } = require('../src/index');

describe('Belief', () => {
  describe('constructor', () => {
    test('should create a belief with valid parameters', () => {
      const belief = new Belief('The sky is blue', 0.95, 'Direct observation');
      
      expect(belief.proposition).toBe('The sky is blue');
      expect(belief.confidence).toBe(0.95);
      expect(belief.justification).toBe('Direct observation');
      expect(belief.timestamp).toBeDefined();
      expect(belief.history).toHaveLength(1);
    });

    test('should throw error for invalid confidence (negative)', () => {
      expect(() => {
        new Belief('Test', -0.5, 'Justification');
      }).toThrow('Confidence must be between 0 and 1');
    });

    test('should throw error for invalid confidence (> 1)', () => {
      expect(() => {
        new Belief('Test', 1.5, 'Justification');
      }).toThrow('Confidence must be between 0 and 1');
    });

    test('should throw error for non-number confidence', () => {
      expect(() => {
        new Belief('Test', 'high', 'Justification');
      }).toThrow('Confidence must be a number');
    });
  });

  describe('update', () => {
    test('should update confidence', () => {
      const belief = new Belief('Test', 0.5, 'Initial');
      belief.update(0.8, 'Updated reason');
      
      expect(belief.confidence).toBe(0.8);
      expect(belief.justification).toBe('Updated reason');
      expect(belief.history).toHaveLength(2);
    });

    test('should allow chaining', () => {
      const belief = new Belief('Test', 0.5, 'Initial');
      const result = belief.update(0.8, 'Reason');
      
      expect(result).toBe(belief);
    });

    test('should preserve old history', () => {
      const belief = new Belief('Test', 0.5, 'Initial');
      belief.update(0.7, 'Second');
      belief.update(0.9, 'Third');
      
      expect(belief.history).toHaveLength(3);
      expect(belief.history[0].confidence).toBe(0.5);
      expect(belief.history[1].confidence).toBe(0.7);
      expect(belief.history[2].confidence).toBe(0.9);
    });
  });

  describe('getHistory', () => {
    test('should return copy of history', () => {
      const belief = new Belief('Test', 0.5, 'Initial');
      const history1 = belief.getHistory();
      const history2 = belief.getHistory();
      
      expect(history1).not.toBe(history2);
      expect(history1).toEqual(history2);
    });
  });

  describe('toJSON', () => {
    test('should serialize belief to JSON', () => {
      const belief = new Belief('Test', 0.8, 'Reason');
      const json = belief.toJSON();
      
      expect(json).toEqual({
        proposition: 'Test',
        confidence: 0.8,
        justification: 'Reason',
        timestamp: belief.timestamp
      });
    });
  });

  describe('fromJSON', () => {
    test('should deserialize belief from JSON', () => {
      const timestamp = Date.now();
      const json = {
        proposition: 'Test',
        confidence: 0.8,
        justification: 'Reason',
        timestamp
      };
      
      const belief = Belief.fromJSON(json);
      
      expect(belief.proposition).toBe('Test');
      expect(belief.confidence).toBe(0.8);
      expect(belief.justification).toBe('Reason');
      expect(belief.timestamp).toBe(timestamp);
    });
  });
});
