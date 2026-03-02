/**
 * UncertaintyTracker Tests
 * Comprehensive unit tests for CATTS-style uncertainty tracking
 */

const { UncertaintyTracker } = require('../src/UncertaintyTracker');

describe('UncertaintyTracker', () => {
  let tracker;

  beforeEach(() => {
    tracker = new UncertaintyTracker();
  });

  describe('Constructor', () => {
    test('should create tracker with default config', () => {
      expect(tracker.windowSize).toBe(10);
      expect(tracker.scaleUpThreshold).toBe(0.7);
      expect(tracker.scaleDownThreshold).toBe(0.3);
      expect(tracker.voteHistory).toEqual([]);
      expect(tracker.confidenceHistory).toEqual([]);
      expect(tracker.successHistory).toEqual([]);
    });

    test('should create tracker with custom config', () => {
      const customTracker = new UncertaintyTracker({
        windowSize: 20,
        scaleUpThreshold: 0.8,
        scaleDownThreshold: 0.2
      });
      
      expect(customTracker.windowSize).toBe(20);
      expect(customTracker.scaleUpThreshold).toBe(0.8);
      expect(customTracker.scaleDownThreshold).toBe(0.2);
    });

    test('should handle partial config', () => {
      const customTracker = new UncertaintyTracker({ windowSize: 5 });
      
      expect(customTracker.windowSize).toBe(5);
      expect(customTracker.scaleUpThreshold).toBe(0.7);
      expect(customTracker.scaleDownThreshold).toBe(0.3);
    });
  });

  describe('recordVote', () => {
    test('should record a successful vote', () => {
      const vote = { success: true, confidence: 0.8 };
      const result = tracker.recordVote(vote);
      
      expect(result.success).toBe(true);
      expect(result.confidence).toBe(0.8);
      expect(result.timestamp).toBeDefined();
      expect(tracker.voteHistory).toHaveLength(1);
    });

    test('should record a failed vote', () => {
      const vote = { success: false, confidence: 0.3 };
      const result = tracker.recordVote(vote);
      
      expect(result.success).toBe(false);
      expect(result.confidence).toBe(0.3);
      expect(tracker.successHistory).toEqual([0]);
    });

    test('should use default confidence of 0.5', () => {
      tracker.recordVote({ success: true });
      
      expect(tracker.confidenceHistory[0]).toBe(0.5);
    });

    test('should clamp confidence to [0, 1]', () => {
      tracker.recordVote({ success: true, confidence: 1.5 });
      tracker.recordVote({ success: true, confidence: -0.5 });
      
      expect(tracker.confidenceHistory[0]).toBe(1);
      expect(tracker.confidenceHistory[1]).toBe(0);
    });

    test('should use provided timestamp', () => {
      const timestamp = 1234567890;
      const result = tracker.recordVote({ success: true, timestamp });
      
      expect(result.timestamp).toBe(timestamp);
    });

    test('should maintain sliding window', () => {
      tracker = new UncertaintyTracker({ windowSize: 3 });
      
      tracker.recordVote({ success: true, confidence: 0.1 });
      tracker.recordVote({ success: true, confidence: 0.2 });
      tracker.recordVote({ success: true, confidence: 0.3 });
      tracker.recordVote({ success: true, confidence: 0.4 });
      
      expect(tracker.voteHistory).toHaveLength(3);
      expect(tracker.confidenceHistory).toEqual([0.2, 0.3, 0.4]);
    });

    test('should accept boolean success values', () => {
      tracker.recordVote({ success: true });
      tracker.recordVote({ success: 1 });
      tracker.recordVote({ success: false });
      tracker.recordVote({ success: 0 });
      tracker.recordVote({ success: null });
      
      expect(tracker.successHistory).toEqual([1, 1, 0, 0, 0]);
    });
  });

  describe('getHistoryLength', () => {
    test('should return 0 for empty tracker', () => {
      expect(tracker.getHistoryLength()).toBe(0);
    });

    test('should return correct count', () => {
      tracker.recordVote({ success: true });
      tracker.recordVote({ success: true });
      
      expect(tracker.getHistoryLength()).toBe(2);
    });
  });

  describe('hasEnoughData', () => {
    test('should return false with 0 votes', () => {
      expect(tracker.hasEnoughData()).toBe(false);
    });

    test('should return false with 1 vote', () => {
      tracker.recordVote({ success: true });
      expect(tracker.hasEnoughData()).toBe(false);
    });

    test('should return true with 2 or more votes', () => {
      tracker.recordVote({ success: true });
      tracker.recordVote({ success: true });
      expect(tracker.hasEnoughData()).toBe(true);
    });
  });

  describe('getEntropy', () => {
    test('should return 0 for empty history', () => {
      expect(tracker.getEntropy()).toBe(0);
    });

    test('should return 0 for all same outcomes', () => {
      // All successes - no uncertainty
      tracker.recordVote({ success: true });
      tracker.recordVote({ success: true });
      tracker.recordVote({ success: true });
      
      expect(tracker.getEntropy()).toBe(0);
    });

    test('should return max entropy for 50/50 split', () => {
      // Equal successes and failures = maximum entropy for binary
      tracker.recordVote({ success: true });
      tracker.recordVote({ success: false });
      
      const entropy = tracker.getEntropy();
      expect(entropy).toBeCloseTo(1, 2); // log2(2) = 1
    });

    test('should calculate entropy correctly for biased distribution', () => {
      // 3 successes, 1 failure
      tracker.recordVote({ success: true });
      tracker.recordVote({ success: true });
      tracker.recordVote({ success: true });
      tracker.recordVote({ success: false });
      
      const entropy = tracker.getEntropy();
      // H = -0.75*log2(0.75) - 0.25*log2(0.25) ≈ 0.811
      expect(entropy).toBeCloseTo(0.811, 2);
    });
  });

  describe('getVariance', () => {
    test('should return 0 for empty history', () => {
      expect(tracker.getVariance()).toBe(0);
    });

    test('should return 0 for single vote', () => {
      tracker.recordVote({ success: true });
      expect(tracker.getVariance()).toBe(0);
    });

    test('should calculate variance for mixed outcomes', () => {
      // 2 successes, 2 failures: mean = 0.5
      // Variance = [(1-0.5)² + (1-0.5)² + (0-0.5)² + (0-0.5)²] / 4 = 0.25
      tracker.recordVote({ success: true });
      tracker.recordVote({ success: true });
      tracker.recordVote({ success: false });
      tracker.recordVote({ success: false });
      
      expect(tracker.getVariance()).toBe(0.25);
    });

    test('should return 0 for all same outcomes', () => {
      tracker.recordVote({ success: true });
      tracker.recordVote({ success: true });
      tracker.recordVote({ success: true });
      
      expect(tracker.getVariance()).toBe(0);
    });
  });

  describe('getSuccessRate', () => {
    test('should return 0 for empty history', () => {
      expect(tracker.getSuccessRate()).toBe(0);
    });

    test('should calculate success rate correctly', () => {
      tracker.recordVote({ success: true });
      tracker.recordVote({ success: true });
      tracker.recordVote({ success: false });
      
      expect(tracker.getSuccessRate()).toBeCloseTo(0.667, 2);
    });

    test('should return 1 for all successes', () => {
      tracker.recordVote({ success: true });
      tracker.recordVote({ success: true });
      
      expect(tracker.getSuccessRate()).toBe(1);
    });

    test('should return 0 for all failures', () => {
      tracker.recordVote({ success: false });
      tracker.recordVote({ success: false });
      
      expect(tracker.getSuccessRate()).toBe(0);
    });
  });

  describe('getAverageConfidence', () => {
    test('should return 0 for empty history', () => {
      expect(tracker.getAverageConfidence()).toBe(0);
    });

    test('should calculate average correctly', () => {
      tracker.recordVote({ success: true, confidence: 0.8 });
      tracker.recordVote({ success: true, confidence: 0.6 });
      
      expect(tracker.getAverageConfidence()).toBe(0.7);
    });
  });

  describe('getConfidenceVariance', () => {
    test('should return 0 for empty history', () => {
      expect(tracker.getConfidenceVariance()).toBe(0);
    });

    test('should return 0 for single vote', () => {
      tracker.recordVote({ success: true, confidence: 0.8 });
      expect(tracker.getConfidenceVariance()).toBe(0);
    });

    test('should calculate confidence variance correctly', () => {
      // Confidences: 0.9, 0.7 - mean = 0.8
      // Variance = [(0.9-0.8)² + (0.7-0.8)²] / 2 = 0.01
      tracker.recordVote({ success: true, confidence: 0.9 });
      tracker.recordVote({ success: true, confidence: 0.7 });
      
      expect(tracker.getConfidenceVariance()).toBeCloseTo(0.01, 3);
    });
  });

  describe('getUncertainty', () => {
    test('should return 1.0 when no data', () => {
      expect(tracker.getUncertainty()).toBe(1.0);
    });

    test('should return 1.0 with single vote (not enough data)', () => {
      tracker.recordVote({ success: true });
      expect(tracker.getUncertainty()).toBe(1.0);
    });

    test('should return lower uncertainty for consistent outcomes', () => {
      tracker.recordVote({ success: true });
      tracker.recordVote({ success: true });
      tracker.recordVote({ success: true });
      
      const uncertainty = tracker.getUncertainty();
      expect(uncertainty).toBeLessThan(0.5);
    });

    test('should return higher uncertainty for mixed outcomes', () => {
      tracker.recordVote({ success: true });
      tracker.recordVote({ success: false });
      tracker.recordVote({ success: true });
      tracker.recordVote({ success: false });
      
      const uncertainty = tracker.getUncertainty();
      expect(uncertainty).toBeGreaterThan(0.5);
    });

    test('should return uncertainty in [0, 1]', () => {
      tracker.recordVote({ success: true });
      tracker.recordVote({ success: true });
      
      const uncertainty = tracker.getUncertainty();
      expect(uncertainty).toBeGreaterThanOrEqual(0);
      expect(uncertainty).toBeLessThanOrEqual(1);
    });
  });

  describe('shouldScaleUp', () => {
    test('should return true with no data', () => {
      expect(tracker.shouldScaleUp()).toBe(true);
    });

    test('should return true with single vote', () => {
      tracker.recordVote({ success: true });
      expect(tracker.shouldScaleUp()).toBe(true);
    });

    test('should return true when entropy is high', () => {
      // Create high entropy with alternating successes/failures
      tracker.recordVote({ success: true });
      tracker.recordVote({ success: false });
      tracker.recordVote({ success: true });
      tracker.recordVote({ success: false });
      
      expect(tracker.shouldScaleUp()).toBe(true);
    });

    test('should return false when outcomes are consistent', () => {
      // Create low entropy with consistent outcomes
      for (let i = 0; i < 10; i++) {
        tracker.recordVote({ success: true });
      }
      
      expect(tracker.shouldScaleUp()).toBe(false);
    });

    test('should respect custom threshold', () => {
      tracker = new UncertaintyTracker({
        windowSize: 4,
        scaleUpThreshold: 0.99 // Very high threshold
      });
      
      // With max entropy (1.0), 1.0 > 0.99, so should scale up
      tracker.recordVote({ success: true });
      tracker.recordVote({ success: false });
      tracker.recordVote({ success: true });
      tracker.recordVote({ success: false });
      
      expect(tracker.shouldScaleUp()).toBe(true);
      
      // Now test with threshold at/below max entropy to prevent scaling
      tracker2 = new UncertaintyTracker({
        windowSize: 4,
        scaleUpThreshold: 1.0 // At max entropy - won't trigger
      });
      
      tracker2.recordVote({ success: true });
      tracker2.recordVote({ success: false });
      tracker2.recordVote({ success: true });
      tracker2.recordVote({ success: false });
      
      expect(tracker2.shouldScaleUp()).toBe(false);
    });
  });

  describe('shouldScaleDown', () => {
    test('should return false with no data', () => {
      expect(tracker.shouldScaleDown()).toBe(false);
    });

    test('should return false with single vote', () => {
      tracker.recordVote({ success: true });
      expect(tracker.shouldScaleDown()).toBe(false);
    });

    test('should return true when entropy is low', () => {
      // Create low entropy with consistent successes
      for (let i = 0; i < 5; i++) {
        tracker.recordVote({ success: true });
      }
      
      expect(tracker.shouldScaleDown()).toBe(true);
    });

    test('should return false when entropy is high', () => {
      // Create high entropy
      tracker.recordVote({ success: true });
      tracker.recordVote({ success: false });
      tracker.recordVote({ success: true });
      tracker.recordVote({ success: false });
      
      expect(tracker.shouldScaleDown()).toBe(false);
    });

    test('should respect custom threshold', () => {
      tracker = new UncertaintyTracker({
        windowSize: 4,
        scaleDownThreshold: 0.01 // Very low threshold
      });
      
      // With 0 entropy (all same), 0 < 0.01, so should scale down
      for (let i = 0; i < 4; i++) {
        tracker.recordVote({ success: true });
      }
      
      expect(tracker.shouldScaleDown()).toBe(true);
      
      // Now test with threshold at 0 to prevent scaling down
      tracker2 = new UncertaintyTracker({
        windowSize: 4,
        scaleDownThreshold: 0.0 // At min entropy - won't trigger
      });
      
      for (let i = 0; i < 4; i++) {
        tracker2.recordVote({ success: true });
      }
      
      expect(tracker2.shouldScaleDown()).toBe(false);
    });
  });

  describe('getStatistics', () => {
    test('should return complete statistics object', () => {
      tracker.recordVote({ success: true, confidence: 0.8 });
      tracker.recordVote({ success: true, confidence: 0.9 });
      
      const stats = tracker.getStatistics();
      
      expect(stats).toHaveProperty('historyLength');
      expect(stats).toHaveProperty('successCount');
      expect(stats).toHaveProperty('failureCount');
      expect(stats).toHaveProperty('successRate');
      expect(stats).toHaveProperty('averageConfidence');
      expect(stats).toHaveProperty('entropy');
      expect(stats).toHaveProperty('variance');
      expect(stats).toHaveProperty('confidenceVariance');
      expect(stats).toHaveProperty('uncertainty');
      expect(stats).toHaveProperty('shouldScaleUp');
      expect(stats).toHaveProperty('shouldScaleDown');
      expect(stats).toHaveProperty('windowSize');
      expect(stats).toHaveProperty('scaleUpThreshold');
      expect(stats).toHaveProperty('scaleDownThreshold');
      expect(stats).toHaveProperty('recentVotes');
    });

    test('should have correct counts', () => {
      tracker.recordVote({ success: true });
      tracker.recordVote({ success: false });
      tracker.recordVote({ success: true });
      
      const stats = tracker.getStatistics();
      
      expect(stats.historyLength).toBe(3);
      expect(stats.successCount).toBe(2);
      expect(stats.failureCount).toBe(1);
    });

    test('should have correct boolean flags', () => {
      // Consistent outcomes
      for (let i = 0; i < 5; i++) {
        tracker.recordVote({ success: true });
      }
      
      const stats = tracker.getStatistics();
      
      expect(stats.shouldScaleUp).toBe(false);
      expect(stats.shouldScaleDown).toBe(true);
    });

    test('should include recent votes', () => {
      tracker.recordVote({ success: true, confidence: 0.8 });
      tracker.recordVote({ success: false, confidence: 0.3 });
      
      const stats = tracker.getStatistics();
      
      expect(stats.recentVotes).toHaveLength(2);
      expect(stats.recentVotes[0]).toEqual({ success: true, confidence: 0.8 });
    });

    test('recentVotes should be limited to 5 entries', () => {
      for (let i = 0; i < 10; i++) {
        tracker.recordVote({ success: true, confidence: i / 10 });
      }
      
      const stats = tracker.getStatistics();
      
      expect(stats.recentVotes).toHaveLength(5);
    });
  });

  describe('reset', () => {
    test('should clear all history', () => {
      tracker.recordVote({ success: true });
      tracker.recordVote({ success: false });
      
      tracker.reset();
      
      expect(tracker.voteHistory).toEqual([]);
      expect(tracker.confidenceHistory).toEqual([]);
      expect(tracker.successHistory).toEqual([]);
      expect(tracker.getHistoryLength()).toBe(0);
    });

    test('should not affect configuration', () => {
      tracker = new UncertaintyTracker({ windowSize: 5 });
      tracker.recordVote({ success: true });
      
      tracker.reset();
      
      expect(tracker.windowSize).toBe(5);
    });
  });

  describe('getHistory', () => {
    test('should return empty array for new tracker', () => {
      expect(tracker.getHistory()).toEqual([]);
    });

    test('should return copy of history', () => {
      tracker.recordVote({ success: true, confidence: 0.8 });
      
      const history = tracker.getHistory();
      history[0].confidence = 0.9;
      
      expect(tracker.voteHistory[0].confidence).toBe(0.8);
    });

    test('should include all history entries', () => {
      tracker.recordVote({ success: true, confidence: 0.8 });
      tracker.recordVote({ success: false, confidence: 0.2 });
      
      const history = tracker.getHistory();
      
      expect(history).toHaveLength(2);
      expect(history[0].success).toBe(true);
      expect(history[1].success).toBe(false);
    });
  });

  describe('Edge Cases', () => {
    test('should handle rapid alternating votes', () => {
      for (let i = 0; i < 20; i++) {
        tracker.recordVote({ success: i % 2 === 0, confidence: 0.5 });
      }
      
      // Window should maintain only last 10
      expect(tracker.voteHistory).toHaveLength(10);
      
      // Should have high entropy
      expect(tracker.getEntropy()).toBeCloseTo(1, 1);
    });

    test('should handle all zero confidence', () => {
      tracker.recordVote({ success: true, confidence: 0 });
      tracker.recordVote({ success: true, confidence: 0 });
      
      expect(tracker.getAverageConfidence()).toBe(0);
      expect(tracker.getUncertainty()).toBeLessThan(1);
    });

    test('should handle all max confidence with failures', () => {
      tracker.recordVote({ success: false, confidence: 1 });
      tracker.recordVote({ success: false, confidence: 1 });
      
      expect(tracker.getAverageConfidence()).toBe(1);
      expect(tracker.getSuccessRate()).toBe(0);
    });

    test('should handle sliding window correctly with windowSize 1', () => {
      tracker = new UncertaintyTracker({ windowSize: 1 });
      
      tracker.recordVote({ success: true });
      tracker.recordVote({ success: false });
      
      expect(tracker.voteHistory).toHaveLength(1);
      expect(tracker.voteHistory[0].success).toBe(false);
    });

    test('entropy calculation with floating point precision', () => {
      // 7 successes, 3 failures - specific distribution
      for (let i = 0; i < 7; i++) tracker.recordVote({ success: true });
      for (let i = 0; i < 3; i++) tracker.recordVote({ success: false });
      
      const entropy = tracker.getEntropy();
      // H = -0.7*log2(0.7) - 0.3*log2(0.3) ≈ 0.881
      expect(entropy).toBeGreaterThan(0);
      expect(entropy).toBeLessThanOrEqual(1);
      expect(Number.isFinite(entropy)).toBe(true);
    });
  });

  describe('Integration with sliding window', () => {
    test('should maintain proper statistics as window slides', () => {
      tracker = new UncertaintyTracker({ windowSize: 5 });
      
      // First 5: all successes (low entropy)
      for (let i = 0; i < 5; i++) {
        tracker.recordVote({ success: true });
      }
      expect(tracker.getEntropy()).toBe(0);
      expect(tracker.shouldScaleDown()).toBe(true);
      
      // Add 5 alternating votes to get high entropy mix
      // This will replace all 5 successes with 5 alternating votes
      for (let i = 0; i < 5; i++) {
        tracker.recordVote({ success: i % 2 === 0 });
      }
      // Window now has alternating pattern -> high entropy
      expect(tracker.getEntropy()).toBeGreaterThan(0.9);
      expect(tracker.shouldScaleUp()).toBe(true);
      
      // Add 5 more failures: window now all failures (low entropy)
      for (let i = 0; i < 5; i++) {
        tracker.recordVote({ success: false });
      }
      expect(tracker.getEntropy()).toBe(0);
      expect(tracker.shouldScaleDown()).toBe(true);
    });

    test('window maintains correct vote order', () => {
      tracker = new UncertaintyTracker({ windowSize: 3 });
      
      tracker.recordVote({ success: true, confidence: 0.1 });
      tracker.recordVote({ success: true, confidence: 0.2 });
      tracker.recordVote({ success: true, confidence: 0.3 });
      tracker.recordVote({ success: true, confidence: 0.4 });
      
      const history = tracker.getHistory();
      expect(history[0].confidence).toBe(0.2);
      expect(history[1].confidence).toBe(0.3);
      expect(history[2].confidence).toBe(0.4);
    });
  });
});
