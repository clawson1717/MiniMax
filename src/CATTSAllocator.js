/**
 * CATTSAllocator - Compute Allocator for Test-Time Scaling
 * 
 * Based on CATTS (Confidence-Adaptive Test-Time Scaling) paper:
 * - Dynamically allocates compute based on uncertainty metrics
 * - Higher uncertainty → more reasoning steps/votes (increased depth)
 * - Lower uncertainty → fewer reasoning steps (save tokens)
 * - Uses UncertaintyTracker to get current uncertainty metrics
 */

/**
 * CATTSAllocator class for dynamic compute allocation
 * Manages reasoning depth based on uncertainty measurements
 */
class CATTSAllocator {
  /**
   * Create a new CATTSAllocator
   * @param {Object} config - Configuration options
   * @param {number} [config.minReasoningDepth=1] - Minimum reasoning depth
   * @param {number} [config.maxReasoningDepth=5] - Maximum reasoning depth
   * @param {number} [config.uncertaintyThreshold=0.5] - Uncertainty threshold for scaling decisions
   */
  constructor(config = {}) {
    this.minReasoningDepth = config.minReasoningDepth !== undefined ? config.minReasoningDepth : 1;
    this.maxReasoningDepth = config.maxReasoningDepth !== undefined ? config.maxReasoningDepth : 5;
    this.uncertaintyThreshold = config.uncertaintyThreshold !== undefined ? config.uncertaintyThreshold : 0.5;

    // Validate config
    if (this.minReasoningDepth < 0) {
      throw new Error('minReasoningDepth must be non-negative');
    }
    if (this.maxReasoningDepth < this.minReasoningDepth) {
      throw new Error('maxReasoningDepth must be >= minReasoningDepth');
    }
    if (this.uncertaintyThreshold < 0 || this.uncertaintyThreshold > 1) {
      throw new Error('uncertaintyThreshold must be between 0 and 1');
    }

    // Current state
    this.currentDepth = this.minReasoningDepth;
    
    // Allocation history for statistics
    this.allocationHistory = [];
    
    // Track allocations
    this.totalAllocations = 0;
    this.scaleUpCount = 0;
    this.scaleDownCount = 0;
  }

  /**
   * Allocate compute based on current uncertainty metrics
   * Returns the reasoning depth to use for the current query
   * @param {UncertaintyTracker} uncertaintyTracker - Uncertainty tracker instance
   * @returns {number} The reasoning depth to use
   */
  allocateCompute(uncertaintyTracker) {
    if (!uncertaintyTracker) {
      throw new Error('uncertaintyTracker is required');
    }

    // Get current uncertainty metrics
    const stats = uncertaintyTracker.getStatistics();
    const entropy = stats.entropy;

    // Record the pre-allocation state
    const allocationRecord = {
      timestamp: Date.now(),
      previousDepth: this.currentDepth,
      entropy: entropy,
      uncertainty: stats.uncertainty,
      successRate: stats.successRate,
      averageConfidence: stats.averageConfidence
    };

    // Apply scaling logic based on entropy
    // High uncertainty (entropy > threshold) → scale up
    // Low uncertainty (entropy < threshold/2) → scale down
    if (entropy > this.uncertaintyThreshold) {
      this.scaleUp();
      allocationRecord.action = 'scaleUp';
      allocationRecord.reason = `entropy (${entropy.toFixed(3)}) > threshold (${this.uncertaintyThreshold})`;
    } else if (entropy < this.uncertaintyThreshold / 2) {
      this.scaleDown();
      allocationRecord.action = 'scaleDown';
      allocationRecord.reason = `entropy (${entropy.toFixed(3)}) < threshold/2 (${(this.uncertaintyThreshold / 2).toFixed(3)})`;
    } else {
      allocationRecord.action = 'maintain';
      allocationRecord.reason = `entropy (${entropy.toFixed(3)}) within normal range`;
    }

    allocationRecord.newDepth = this.currentDepth;
    this.allocationHistory.push(allocationRecord);
    this.totalAllocations++;

    return this.currentDepth;
  }

  /**
   * Get the current reasoning depth
   * @returns {number} Current reasoning depth
   */
  getReasoningDepth() {
    return this.currentDepth;
  }

  /**
   * Scale up reasoning depth (increase compute allocation)
   * @returns {number} The new reasoning depth
   */
  scaleUp() {
    if (this.currentDepth < this.maxReasoningDepth) {
      this.currentDepth++;
      this.scaleUpCount++;
    }
    return this.currentDepth;
  }

  /**
   * Scale down reasoning depth (decrease compute allocation)
   * @returns {number} The new reasoning depth
   */
  scaleDown() {
    if (this.currentDepth > this.minReasoningDepth) {
      this.currentDepth--;
      this.scaleDownCount++;
    }
    return this.currentDepth;
  }

  /**
   * Determine if the agent should re-query with more compute
   * This is useful when uncertainty is too high after initial query
   * @param {number} uncertainty - Uncertainty score (0-1)
   * @returns {boolean} True if should re-query with more compute
   */
  shouldRequery(uncertainty) {
    // Should re-query if uncertainty is high and we haven't maxed out depth
    const canScaleMore = this.currentDepth < this.maxReasoningDepth;
    const highUncertainty = uncertainty > this.uncertaintyThreshold;
    
    return canScaleMore && highUncertainty;
  }

  /**
   * Get allocation history and statistics
   * @returns {Object} Allocation statistics
   */
  getAllocationStats() {
    const recentAllocations = this.allocationHistory.slice(-10);
    
    // Calculate average depth over history
    const avgDepth = this.allocationHistory.length > 0
      ? this.allocationHistory.reduce((sum, rec) => sum + rec.newDepth, 0) / this.allocationHistory.length
      : this.currentDepth;

    // Calculate depth distribution
    const depthDistribution = {};
    for (let d = this.minReasoningDepth; d <= this.maxReasoningDepth; d++) {
      depthDistribution[d] = this.allocationHistory.filter(rec => rec.newDepth === d).length;
    }

    return {
      // Current state
      currentDepth: this.currentDepth,
      
      // Configuration
      config: {
        minReasoningDepth: this.minReasoningDepth,
        maxReasoningDepth: this.maxReasoningDepth,
        uncertaintyThreshold: this.uncertaintyThreshold
      },
      
      // Statistics
      totalAllocations: this.totalAllocations,
      scaleUpCount: this.scaleUpCount,
      scaleDownCount: this.scaleDownCount,
      averageDepth: avgDepth,
      depthDistribution,
      
      // Recent history (last 10 allocations)
      recentAllocations,
      
      // Full history length
      historyLength: this.allocationHistory.length
    };
  }

  /**
   * Reset allocator to default state
   * Clears history and resets depth to minimum
   */
  reset() {
    this.currentDepth = this.minReasoningDepth;
    this.allocationHistory = [];
    this.totalAllocations = 0;
    this.scaleUpCount = 0;
    this.scaleDownCount = 0;
  }
}

module.exports = { CATTSAllocator };
