/**
 * UncertaintyTracker - CATTS-style uncertainty tracking for dynamic compute allocation
 * 
 * Based on CATTS (Confidence-Adaptive Tree of Thoughts) paper:
 * - Uncertainty is measured via vote distribution entropy
 * - High entropy = high uncertainty = need more compute
 * - Low entropy = low uncertainty = can use less compute
 * - Uses a sliding window for recent history
 */

/**
 * UncertaintyTracker class for tracking vote uncertainty and enabling
 * dynamic compute allocation decisions
 */
class UncertaintyTracker {
  /**
   * Create a new UncertaintyTracker
   * @param {Object} config - Configuration options
   * @param {number} [config.windowSize=10] - Size of sliding window for moving averages
   * @param {number} [config.scaleUpThreshold=0.7] - Entropy threshold above which to scale up
   * @param {number} [config.scaleDownThreshold=0.3] - Entropy threshold below which to scale down
   */
  constructor(config = {}) {
    this.windowSize = config.windowSize || 10;
    this.scaleUpThreshold = config.scaleUpThreshold !== undefined ? config.scaleUpThreshold : 0.7;
    this.scaleDownThreshold = config.scaleDownThreshold !== undefined ? config.scaleDownThreshold : 0.3;
    
    // History storage
    this.voteHistory = [];       // Array of vote objects { success, confidence, timestamp }
    this.confidenceHistory = []; // Array of confidence scores
    this.successHistory = [];    // Array of 1s (success) and 0s (failure)
  }

  /**
   * Record a vote/response from the agent
   * @param {Object} vote - Vote data
   * @param {boolean} vote.success - Whether the action was successful
   * @param {number} [vote.confidence=0.5] - Confidence score (0-1)
   * @param {number} [vote.timestamp=Date.now()] - Optional timestamp
   * @returns {Object} Recorded vote entry
   */
  recordVote(vote) {
    const entry = {
      success: Boolean(vote.success),
      confidence: vote.confidence !== undefined ? Math.max(0, Math.min(1, vote.confidence)) : 0.5,
      timestamp: vote.timestamp || Date.now()
    };

    this.voteHistory.push(entry);
    this.confidenceHistory.push(entry.confidence);
    this.successHistory.push(entry.success ? 1 : 0);

    // Maintain sliding window
    this._trimWindow();

    return entry;
  }

  /**
   * Trim all histories to maintain sliding window size
   * @private
   */
  _trimWindow() {
    if (this.voteHistory.length > this.windowSize) {
      const excess = this.voteHistory.length - this.windowSize;
      this.voteHistory.splice(0, excess);
      this.confidenceHistory.splice(0, excess);
      this.successHistory.splice(0, excess);
    }
  }

  /**
   * Get the current number of votes in history
   * @returns {number}
   */
  getHistoryLength() {
    return this.voteHistory.length;
  }

  /**
   * Check if we have enough data for uncertainty calculation
   * @returns {boolean}
   */
  hasEnoughData() {
    return this.voteHistory.length >= 2;
  }

  /**
   * Calculate current uncertainty based on recent votes
   * Returns normalized uncertainty score (0-1), where higher = more uncertain
   * @returns {number} Uncertainty score
   */
  getUncertainty() {
    if (!this.hasEnoughData()) {
      return 1.0; // Maximum uncertainty when no data
    }

    // Use entropy as the primary uncertainty metric
    const entropy = this.getEntropy();
    
    // Also consider variance for additional signal
    const variance = this.getVariance();
    
    // Combined uncertainty: entropy dominates, variance adds nuance
    // Entropy is already normalized to [0, 1] since we use log2 (max entropy for binary = 1)
    const normalizedVariance = variance * 4; // Variance max is 0.25 for binary, so scale by 4
    
    // Weighted combination (70% entropy, 30% variance)
    const uncertainty = 0.7 * entropy + 0.3 * Math.min(1, normalizedVariance);
    
    return Math.max(0, Math.min(1, uncertainty));
  }

  /**
   * Calculate entropy of recent vote distribution
   * Uses Shannon entropy: H(X) = -sum(p(x) * log2(p(x)))
   * For binary outcomes (success/failure), max entropy is log2(2) = 1
   * @returns {number} Entropy value
   */
  getEntropy() {
    if (this.voteHistory.length === 0) {
      return 0;
    }

    const successCount = this.successHistory.reduce((sum, val) => sum + val, 0);
    const failureCount = this.voteHistory.length - successCount;
    const total = this.voteHistory.length;

    // Calculate probabilities
    const pSuccess = successCount / total;
    const pFailure = failureCount / total;

    // Calculate entropy
    let entropy = 0;
    
    if (pSuccess > 0) {
      entropy -= pSuccess * Math.log2(pSuccess);
    }
    if (pFailure > 0) {
      entropy -= pFailure * Math.log2(pFailure);
    }

    return entropy;
  }

  /**
   * Calculate variance in success rates
   * Uses sample variance calculation
   * @returns {number} Variance value (0-0.25 for binary outcomes)
   */
  getVariance() {
    if (this.voteHistory.length === 0) {
      return 0;
    }

    if (this.voteHistory.length === 1) {
      return 0;
    }

    const mean = this.successHistory.reduce((sum, val) => sum + val, 0) / this.voteHistory.length;
    
    // Sample variance (divide by n-1 for sample, n for population)
    // Using population variance for sliding window
    const squaredDiffs = this.successHistory.map(val => Math.pow(val - mean, 2));
    const variance = squaredDiffs.reduce((sum, val) => sum + val, 0) / this.voteHistory.length;

    return variance;
  }

  /**
   * Calculate moving average of success rate
   * @returns {number} Success rate (0-1)
   */
  getSuccessRate() {
    if (this.voteHistory.length === 0) {
      return 0;
    }

    const sum = this.successHistory.reduce((acc, val) => acc + val, 0);
    return sum / this.voteHistory.length;
  }

  /**
   * Calculate moving average of confidence scores
   * @returns {number} Average confidence (0-1)
   */
  getAverageConfidence() {
    if (this.confidenceHistory.length === 0) {
      return 0;
    }

    const sum = this.confidenceHistory.reduce((acc, val) => acc + val, 0);
    return sum / this.confidenceHistory.length;
  }

  /**
   * Calculate confidence variance (how consistent are confidence scores)
   * @returns {number} Variance of confidence scores
   */
  getConfidenceVariance() {
    if (this.confidenceHistory.length < 2) {
      return 0;
    }

    const mean = this.getAverageConfidence();
    const squaredDiffs = this.confidenceHistory.map(val => Math.pow(val - mean, 2));
    return squaredDiffs.reduce((sum, val) => sum + val, 0) / this.confidenceHistory.length;
  }

  /**
   * Determine if we should scale up compute (high uncertainty)
   * @returns {boolean} True if more compute is needed
   */
  shouldScaleUp() {
    if (!this.hasEnoughData()) {
      return true; // Scale up when we don't have enough data
    }

    // Entropy is already normalized to [0, 1] since we use log2 (max entropy for binary = 1)
    const entropy = this.getEntropy();

    return entropy > this.scaleUpThreshold;
  }

  /**
   * Determine if we should scale down compute (low uncertainty)
   * @returns {boolean} True if less compute can be used
   */
  shouldScaleDown() {
    if (!this.hasEnoughData()) {
      return false; // Can't scale down without enough data
    }

    // Entropy is already normalized to [0, 1] since we use log2 (max entropy for binary = 1)
    const entropy = this.getEntropy();

    return entropy < this.scaleDownThreshold;
  }

  /**
   * Get all uncertainty metrics in one object
   * @returns {Object} Complete statistics
   */
  getStatistics() {
    const historyLength = this.voteHistory.length;
    const successRate = this.getSuccessRate();
    const avgConfidence = this.getAverageConfidence();
    const entropy = this.getEntropy();
    const variance = this.getVariance();
    const confidenceVariance = this.getConfidenceVariance();
    const uncertainty = this.getUncertainty();

    return {
      // Basic counts
      historyLength,
      successCount: this.successHistory.reduce((sum, val) => sum + val, 0),
      failureCount: historyLength - this.successHistory.reduce((sum, val) => sum + val, 0),
      
      // Moving averages
      successRate,
      averageConfidence: avgConfidence,
      
      // Uncertainty metrics
      entropy,
      variance,
      confidenceVariance,
      uncertainty,
      
      // Decision flags
      shouldScaleUp: this.shouldScaleUp(),
      shouldScaleDown: this.shouldScaleDown(),
      
      // Configuration
      windowSize: this.windowSize,
      scaleUpThreshold: this.scaleUpThreshold,
      scaleDownThreshold: this.scaleDownThreshold,
      
      // Recent history (last 5 entries)
      recentVotes: this.voteHistory.slice(-5).map(v => ({
        success: v.success,
        confidence: v.confidence
      }))
    };
  }

  /**
   * Reset all history
   */
  reset() {
    this.voteHistory = [];
    this.confidenceHistory = [];
    this.successHistory = [];
  }

  /**
   * Get raw vote history (for advanced analysis)
   * @returns {Array<Object>} Copy of vote history
   */
  getHistory() {
    return this.voteHistory.map(v => ({ ...v }));
  }
}

module.exports = { UncertaintyTracker };
