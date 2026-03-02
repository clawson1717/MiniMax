/**
 * CATTSAllocator Tests
 * 
 * Tests for the CATTS (Compute Allocator for Test-Time Scaling) module
 */

const { CATTSAllocator } = require('../src/CATTSAllocator');
const { UncertaintyTracker } = require('../src/UncertaintyTracker');

describe('CATTSAllocator', () => {
  describe('Default Configuration', () => {
    test('should create with default configuration', () => {
      const allocator = new CATTSAllocator();
      
      expect(allocator.minReasoningDepth).toBe(1);
      expect(allocator.maxReasoningDepth).toBe(5);
      expect(allocator.uncertaintyThreshold).toBe(0.5);
      expect(allocator.getReasoningDepth()).toBe(1);
    });

    test('should create with custom configuration', () => {
      const allocator = new CATTSAllocator({
        minReasoningDepth: 2,
        maxReasoningDepth: 10,
        uncertaintyThreshold: 0.7
      });
      
      expect(allocator.minReasoningDepth).toBe(2);
      expect(allocator.maxReasoningDepth).toBe(10);
      expect(allocator.uncertaintyThreshold).toBe(0.7);
      expect(allocator.getReasoningDepth()).toBe(2);
    });

    test('should throw error for invalid minReasoningDepth', () => {
      expect(() => {
        new CATTSAllocator({ minReasoningDepth: -1 });
      }).toThrow('minReasoningDepth must be non-negative');
    });

    test('should throw error when max < min', () => {
      expect(() => {
        new CATTSAllocator({ minReasoningDepth: 5, maxReasoningDepth: 3 });
      }).toThrow('maxReasoningDepth must be >= minReasoningDepth');
    });

    test('should throw error for invalid uncertaintyThreshold', () => {
      expect(() => {
        new CATTSAllocator({ uncertaintyThreshold: 1.5 });
      }).toThrow('uncertaintyThreshold must be between 0 and 1');

      expect(() => {
        new CATTSAllocator({ uncertaintyThreshold: -0.1 });
      }).toThrow('uncertaintyThreshold must be between 0 and 1');
    });
  });

  describe('getReasoningDepth', () => {
    test('should return current depth', () => {
      const allocator = new CATTSAllocator({ minReasoningDepth: 2 });
      expect(allocator.getReasoningDepth()).toBe(2);
    });

    test('should reflect changes after scaling', () => {
      const allocator = new CATTSAllocator({ minReasoningDepth: 1, maxReasoningDepth: 5 });
      
      allocator.scaleUp();
      expect(allocator.getReasoningDepth()).toBe(2);
      
      allocator.scaleDown();
      expect(allocator.getReasoningDepth()).toBe(1);
    });
  });

  describe('scaleUp', () => {
    test('should increase depth by 1', () => {
      const allocator = new CATTSAllocator({ minReasoningDepth: 1, maxReasoningDepth: 5 });
      
      const result = allocator.scaleUp();
      
      expect(result).toBe(2);
      expect(allocator.getReasoningDepth()).toBe(2);
    });

    test('should not exceed maxReasoningDepth', () => {
      const allocator = new CATTSAllocator({ minReasoningDepth: 1, maxReasoningDepth: 3 });
      
      allocator.scaleUp(); // 2
      allocator.scaleUp(); // 3
      allocator.scaleUp(); // should stay at 3
      allocator.scaleUp(); // should stay at 3
      
      expect(allocator.getReasoningDepth()).toBe(3);
    });

    test('should track scale up count', () => {
      const allocator = new CATTSAllocator({ minReasoningDepth: 1, maxReasoningDepth: 5 });
      
      allocator.scaleUp();
      allocator.scaleUp();
      
      const stats = allocator.getAllocationStats();
      expect(stats.scaleUpCount).toBe(2);
    });

    test('should not increment counter when at max', () => {
      const allocator = new CATTSAllocator({ minReasoningDepth: 1, maxReasoningDepth: 2 });
      
      allocator.scaleUp(); // 2, count = 1
      allocator.scaleUp(); // still 2, no increment
      
      const stats = allocator.getAllocationStats();
      expect(stats.scaleUpCount).toBe(1);
    });
  });

  describe('scaleDown', () => {
    test('should decrease depth by 1', () => {
      const allocator = new CATTSAllocator({ minReasoningDepth: 1, maxReasoningDepth: 5 });
      
      allocator.scaleUp(); // 2
      const result = allocator.scaleDown(); // 1
      
      expect(result).toBe(1);
      expect(allocator.getReasoningDepth()).toBe(1);
    });

    test('should not go below minReasoningDepth', () => {
      const allocator = new CATTSAllocator({ minReasoningDepth: 2, maxReasoningDepth: 5 });
      
      allocator.scaleDown();
      allocator.scaleDown();
      
      expect(allocator.getReasoningDepth()).toBe(2);
    });

    test('should track scale down count', () => {
      const allocator = new CATTSAllocator({ minReasoningDepth: 1, maxReasoningDepth: 5 });
      
      allocator.scaleUp();
      allocator.scaleUp();
      allocator.scaleDown();
      
      const stats = allocator.getAllocationStats();
      expect(stats.scaleDownCount).toBe(1);
    });

    test('should not increment counter when at min', () => {
      const allocator = new CATTSAllocator({ minReasoningDepth: 2, maxReasoningDepth: 5 });
      
      allocator.scaleDown(); // already at min
      
      const stats = allocator.getAllocationStats();
      expect(stats.scaleDownCount).toBe(0);
    });
  });

  describe('allocateCompute', () => {
    test('should throw error without uncertaintyTracker', () => {
      const allocator = new CATTSAllocator();
      
      expect(() => {
        allocator.allocateCompute();
      }).toThrow('uncertaintyTracker is required');

      expect(() => {
        allocator.allocateCompute(null);
      }).toThrow('uncertaintyTracker is required');
    });

    test('should scale up when entropy > threshold', () => {
      const allocator = new CATTSAllocator({
        minReasoningDepth: 1,
        maxReasoningDepth: 5,
        uncertaintyThreshold: 0.5
      });
      
      const tracker = new UncertaintyTracker();
      // Create high entropy by mixing successes and failures
      tracker.recordVote({ success: true, confidence: 0.5 });
      tracker.recordVote({ success: false, confidence: 0.5 });
      tracker.recordVote({ success: true, confidence: 0.5 });
      tracker.recordVote({ success: false, confidence: 0.5 });
      // Entropy should be high (close to 1 for 50/50 split)
      
      const depth = allocator.allocateCompute(tracker);
      
      expect(depth).toBeGreaterThan(1);
      expect(allocator.getAllocationStats().scaleUpCount).toBeGreaterThanOrEqual(1);
    });

    test('should scale down when entropy < threshold/2', () => {
      const allocator = new CATTSAllocator({
        minReasoningDepth: 1,
        maxReasoningDepth: 5,
        uncertaintyThreshold: 0.5
      });
      
      // First scale up to have room to scale down
      allocator.scaleUp();
      allocator.scaleUp();
      expect(allocator.getReasoningDepth()).toBe(3);
      
      const tracker = new UncertaintyTracker();
      // Create low entropy by having consistent results
      tracker.recordVote({ success: true, confidence: 0.9 });
      tracker.recordVote({ success: true, confidence: 0.9 });
      tracker.recordVote({ success: true, confidence: 0.9 });
      tracker.recordVote({ success: true, confidence: 0.9 });
      // Entropy should be low (close to 0 for all same outcomes)
      
      const depth = allocator.allocateCompute(tracker);
      
      expect(depth).toBeLessThan(3);
      expect(allocator.getAllocationStats().scaleDownCount).toBeGreaterThanOrEqual(1);
    });

    test('should maintain depth when entropy is in normal range', () => {
      const allocator = new CATTSAllocator({
        minReasoningDepth: 1,
        maxReasoningDepth: 5,
        uncertaintyThreshold: 0.5
      });
      
      allocator.scaleUp();
      const initialDepth = allocator.getReasoningDepth();
      
      const tracker = new UncertaintyTracker();
      // Create moderate entropy (between threshold/2=0.25 and threshold=0.5)
      // For binary outcomes, entropy in this range requires about 9:1 ratio
      // 9 successes, 1 failure gives entropy ~0.47 which is in [0.25, 0.5]
      tracker.recordVote({ success: true, confidence: 0.8 });
      tracker.recordVote({ success: true, confidence: 0.8 });
      tracker.recordVote({ success: true, confidence: 0.8 });
      tracker.recordVote({ success: true, confidence: 0.8 });
      tracker.recordVote({ success: true, confidence: 0.8 });
      tracker.recordVote({ success: true, confidence: 0.8 });
      tracker.recordVote({ success: true, confidence: 0.8 });
      tracker.recordVote({ success: true, confidence: 0.8 });
      tracker.recordVote({ success: true, confidence: 0.8 });
      tracker.recordVote({ success: false, confidence: 0.2 });
      
      // Verify entropy is in expected range
      const entropy = tracker.getEntropy();
      expect(entropy).toBeGreaterThanOrEqual(0.25);
      expect(entropy).toBeLessThanOrEqual(0.5);
      
      const depth = allocator.allocateCompute(tracker);
      
      // With entropy in [0.25, 0.5], depth should stay the same
      expect(depth).toBe(initialDepth);
    });

    test('should record allocation history', () => {
      const allocator = new CATTSAllocator();
      const tracker = new UncertaintyTracker();
      
      tracker.recordVote({ success: true, confidence: 0.5 });
      tracker.recordVote({ success: false, confidence: 0.5 });
      
      allocator.allocateCompute(tracker);
      
      const stats = allocator.getAllocationStats();
      expect(stats.totalAllocations).toBe(1);
      expect(stats.historyLength).toBe(1);
      expect(stats.recentAllocations).toHaveLength(1);
    });
  });

  describe('shouldRequery', () => {
    test('should return true when uncertainty is high and can scale', () => {
      const allocator = new CATTSAllocator({
        minReasoningDepth: 1,
        maxReasoningDepth: 5,
        uncertaintyThreshold: 0.5
      });
      
      // Current depth is 1, max is 5, so we can scale
      const result = allocator.shouldRequery(0.8);
      
      expect(result).toBe(true);
    });

    test('should return false when uncertainty is low', () => {
      const allocator = new CATTSAllocator({
        uncertaintyThreshold: 0.5
      });
      
      const result = allocator.shouldRequery(0.2);
      
      expect(result).toBe(false);
    });

    test('should return false when already at max depth', () => {
      const allocator = new CATTSAllocator({
        minReasoningDepth: 1,
        maxReasoningDepth: 3,
        uncertaintyThreshold: 0.5
      });
      
      // Scale to max
      allocator.scaleUp(); // 2
      allocator.scaleUp(); // 3
      
      const result = allocator.shouldRequery(0.8);
      
      expect(result).toBe(false);
    });

    test('should return false when uncertainty equals threshold', () => {
      const allocator = new CATTSAllocator({
        uncertaintyThreshold: 0.5
      });
      
      const result = allocator.shouldRequery(0.5);
      
      expect(result).toBe(false);
    });

    test('should handle edge case at boundary', () => {
      const allocator = new CATTSAllocator({
        uncertaintyThreshold: 0.5
      });
      
      // Just above threshold
      expect(allocator.shouldRequery(0.501)).toBe(true);
      
      // Just below threshold
      expect(allocator.shouldRequery(0.499)).toBe(false);
    });
  });

  describe('getAllocationStats', () => {
    test('should return complete stats object', () => {
      const allocator = new CATTSAllocator({
        minReasoningDepth: 1,
        maxReasoningDepth: 5,
        uncertaintyThreshold: 0.5
      });
      
      const stats = allocator.getAllocationStats();
      
      expect(stats).toHaveProperty('currentDepth', 1);
      expect(stats).toHaveProperty('config');
      expect(stats.config).toEqual({
        minReasoningDepth: 1,
        maxReasoningDepth: 5,
        uncertaintyThreshold: 0.5
      });
      expect(stats).toHaveProperty('totalAllocations', 0);
      expect(stats).toHaveProperty('scaleUpCount', 0);
      expect(stats).toHaveProperty('scaleDownCount', 0);
      expect(stats).toHaveProperty('averageDepth', 1);
      expect(stats).toHaveProperty('depthDistribution');
      expect(stats).toHaveProperty('recentAllocations');
      expect(stats).toHaveProperty('historyLength', 0);
    });

    test('should calculate correct average depth', () => {
      const allocator = new CATTSAllocator({
        minReasoningDepth: 1,
        maxReasoningDepth: 5
      });
      const tracker = new UncertaintyTracker();
      
      // Force different depths
      allocator.scaleUp(); // depth = 2
      allocator.scaleUp(); // depth = 3
      
      // Record with forced action
      tracker.recordVote({ success: true, confidence: 0.5 });
      tracker.recordVote({ success: false, confidence: 0.5 });
      allocator.allocateCompute(tracker); // This will scale up or down based on entropy
      
      const stats = allocator.getAllocationStats();
      expect(stats.averageDepth).toBeGreaterThan(1);
    });

    test('should track depth distribution', () => {
      const allocator = new CATTSAllocator({
        minReasoningDepth: 1,
        maxReasoningDepth: 3
      });
      const tracker = new UncertaintyTracker();
      
      // Record allocations at different depths
      tracker.recordVote({ success: true, confidence: 0.5 });
      tracker.recordVote({ success: true, confidence: 0.5 });
      allocator.allocateCompute(tracker); // depth 1
      
      allocator.scaleUp(); // depth 2
      tracker.recordVote({ success: true, confidence: 0.5 });
      tracker.recordVote({ success: true, confidence: 0.5 });
      allocator.allocateCompute(tracker); // depth 2
      
      allocator.scaleUp(); // depth 3
      tracker.recordVote({ success: true, confidence: 0.5 });
      tracker.recordVote({ success: true, confidence: 0.5 });
      allocator.allocateCompute(tracker); // depth 3
      
      const stats = allocator.getAllocationStats();
      expect(stats.depthDistribution).toHaveProperty('1');
      expect(stats.depthDistribution).toHaveProperty('2');
      expect(stats.depthDistribution).toHaveProperty('3');
      expect(stats.depthDistribution[1]).toBeGreaterThanOrEqual(0);
      expect(stats.depthDistribution[2]).toBeGreaterThanOrEqual(0);
      expect(stats.depthDistribution[3]).toBeGreaterThanOrEqual(0);
    });

    test('should limit recent allocations to last 10', () => {
      const allocator = new CATTSAllocator();
      const tracker = new UncertaintyTracker();
      
      tracker.recordVote({ success: true, confidence: 0.5 });
      tracker.recordVote({ success: false, confidence: 0.5 });
      
      // Make 15 allocations
      for (let i = 0; i < 15; i++) {
        allocator.allocateCompute(tracker);
      }
      
      const stats = allocator.getAllocationStats();
      expect(stats.recentAllocations).toHaveLength(10);
      expect(stats.totalAllocations).toBe(15);
    });
  });

  describe('reset', () => {
    test('should reset to default state', () => {
      const allocator = new CATTSAllocator({ minReasoningDepth: 2, maxReasoningDepth: 5 });
      const tracker = new UncertaintyTracker();
      
      // Modify state
      allocator.scaleUp();
      allocator.scaleUp();
      tracker.recordVote({ success: true, confidence: 0.5 });
      tracker.recordVote({ success: false, confidence: 0.5 });
      allocator.allocateCompute(tracker);
      
      // Reset
      allocator.reset();
      
      expect(allocator.getReasoningDepth()).toBe(2);
      expect(allocator.getAllocationStats().totalAllocations).toBe(0);
      expect(allocator.getAllocationStats().historyLength).toBe(0);
      expect(allocator.getAllocationStats().scaleUpCount).toBe(0);
      expect(allocator.getAllocationStats().scaleDownCount).toBe(0);
    });

    test('should keep configuration intact', () => {
      const allocator = new CATTSAllocator({
        minReasoningDepth: 2,
        maxReasoningDepth: 10,
        uncertaintyThreshold: 0.7
      });
      
      allocator.scaleUp();
      allocator.reset();
      
      expect(allocator.minReasoningDepth).toBe(2);
      expect(allocator.maxReasoningDepth).toBe(10);
      expect(allocator.uncertaintyThreshold).toBe(0.7);
    });
  });

  describe('Boundary Conditions', () => {
    test('should handle min depth of 0', () => {
      const allocator = new CATTSAllocator({
        minReasoningDepth: 0,
        maxReasoningDepth: 5
      });
      
      expect(allocator.getReasoningDepth()).toBe(0);
      
      // Should not go below 0
      allocator.scaleDown();
      expect(allocator.getReasoningDepth()).toBe(0);
    });

    test('should handle equal min and max depth', () => {
      const allocator = new CATTSAllocator({
        minReasoningDepth: 3,
        maxReasoningDepth: 3
      });
      
      expect(allocator.getReasoningDepth()).toBe(3);
      
      // Should stay at 3
      allocator.scaleUp();
      expect(allocator.getReasoningDepth()).toBe(3);
      
      allocator.scaleDown();
      expect(allocator.getReasoningDepth()).toBe(3);
    });

    test('should handle threshold at boundaries', () => {
      const allocator = new CATTSAllocator({
        uncertaintyThreshold: 0
      });
      
      expect(allocator.uncertaintyThreshold).toBe(0);
      
      const allocator2 = new CATTSAllocator({
        uncertaintyThreshold: 1
      });
      
      expect(allocator2.uncertaintyThreshold).toBe(1);
    });
  });

  describe('Integration with UncertaintyTracker', () => {
    test('should work with real UncertaintyTracker metrics', () => {
      const allocator = new CATTSAllocator({
        minReasoningDepth: 1,
        maxReasoningDepth: 5,
        uncertaintyThreshold: 0.5
      });
      
      const tracker = new UncertaintyTracker({ windowSize: 10 });
      
      // Simulate mixed results (high uncertainty)
      for (let i = 0; i < 5; i++) {
        tracker.recordVote({ success: i % 2 === 0, confidence: 0.5 });
      }
      
      const initialDepth = allocator.getReasoningDepth();
      const newDepth = allocator.allocateCompute(tracker);
      
      // High entropy should trigger scale up
      expect(tracker.getEntropy()).toBeGreaterThan(0);
      expect(allocator.getAllocationStats().totalAllocations).toBe(1);
    });

    test('should use tracker getStatistics method', () => {
      const allocator = new CATTSAllocator();
      const tracker = new UncertaintyTracker();
      
      tracker.recordVote({ success: true, confidence: 0.8 });
      tracker.recordVote({ success: false, confidence: 0.2 });
      
      // This should call tracker.getStatistics()
      const depth = allocator.allocateCompute(tracker);
      
      expect(depth).toBeDefined();
      expect(typeof depth).toBe('number');
    });

    test('should handle empty tracker', () => {
      const allocator = new CATTSAllocator();
      const tracker = new UncertaintyTracker();
      
      // Empty tracker should still work
      const depth = allocator.allocateCompute(tracker);
      
      expect(depth).toBe(allocator.minReasoningDepth);
    });

    test('should integrate with tracker thresholds', () => {
      const tracker = new UncertaintyTracker({
        scaleUpThreshold: 0.7,
        scaleDownThreshold: 0.3
      });
      
      const allocator = new CATTSAllocator({
        uncertaintyThreshold: 0.5
      });
      
      // Both should have consistent threshold semantics
      expect(allocator.uncertaintyThreshold).toBeLessThan(tracker.scaleUpThreshold);
      
      tracker.recordVote({ success: true, confidence: 0.5 });
      tracker.recordVote({ success: false, confidence: 0.5 });
      
      const depth = allocator.allocateCompute(tracker);
      expect(depth).toBeDefined();
    });
  });

  describe('Allocation History Details', () => {
    test('should record all relevant fields in history', () => {
      const allocator = new CATTSAllocator();
      const tracker = new UncertaintyTracker();
      
      tracker.recordVote({ success: true, confidence: 0.5 });
      tracker.recordVote({ success: false, confidence: 0.5 });
      
      allocator.allocateCompute(tracker);
      
      const stats = allocator.getAllocationStats();
      const record = stats.recentAllocations[0];
      
      expect(record).toHaveProperty('timestamp');
      expect(record).toHaveProperty('previousDepth');
      expect(record).toHaveProperty('newDepth');
      expect(record).toHaveProperty('entropy');
      expect(record).toHaveProperty('uncertainty');
      expect(record).toHaveProperty('successRate');
      expect(record).toHaveProperty('averageConfidence');
      expect(record).toHaveProperty('action');
      expect(record).toHaveProperty('reason');
      
      expect(typeof record.timestamp).toBe('number');
      expect(typeof record.previousDepth).toBe('number');
      expect(typeof record.newDepth).toBe('number');
      expect(typeof record.entropy).toBe('number');
    });

    test('should track action types correctly', () => {
      const allocator = new CATTSAllocator({
        minReasoningDepth: 1,
        maxReasoningDepth: 5,
        uncertaintyThreshold: 0.5
      });
      const tracker = new UncertaintyTracker();
      
      // High entropy - should scale up
      tracker.recordVote({ success: true, confidence: 0.5 });
      tracker.recordVote({ success: false, confidence: 0.5 });
      tracker.recordVote({ success: true, confidence: 0.5 });
      tracker.recordVote({ success: false, confidence: 0.5 });
      
      allocator.allocateCompute(tracker);
      
      let stats = allocator.getAllocationStats();
      expect(stats.recentAllocations[0].action).toBe('scaleUp');
      
      // Now create low entropy situation
      const allocator2 = new CATTSAllocator({
        minReasoningDepth: 1,
        maxReasoningDepth: 5,
        uncertaintyThreshold: 0.5
      });
      allocator2.scaleUp();
      allocator2.scaleUp(); // Start at depth 3
      
      const tracker2 = new UncertaintyTracker();
      // All same outcomes - low entropy
      tracker2.recordVote({ success: true, confidence: 0.9 });
      tracker2.recordVote({ success: true, confidence: 0.9 });
      tracker2.recordVote({ success: true, confidence: 0.9 });
      tracker2.recordVote({ success: true, confidence: 0.9 });
      
      allocator2.allocateCompute(tracker2);
      
      stats = allocator2.getAllocationStats();
      expect(stats.recentAllocations[0].action).toBe('scaleDown');
    });
  });
});
