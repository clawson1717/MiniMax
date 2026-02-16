/**
 * BeliefNetwork Class
 * 
 * A graph-based network of beliefs with dependencies.
 * Supports reactive updates - only beliefs affected by new information are updated.
 * 
 * Inspired by Reactive Knowledge Representation research for efficient
 * real-time belief updates.
 */

const { Belief } = require('./Belief');

class BeliefNetwork {
  /**
   * Create a new BeliefNetwork
   * @param {string} agentId - Unique identifier for the agent owning this network
   */
  constructor(agentId) {
    this.agentId = agentId;
    this.beliefs = new Map(); // proposition -> Belief
    this.dependencies = new Map(); // proposition -> Set of dependent propositions
    this.reverseDependencies = new Map(); // proposition -> Set of dependency propositions
    this.updateLog = []; // Track all updates for debugging/analysis
  }

  /**
   * Add a belief to the network
   * @param {string} proposition - The proposition statement
   * @param {number} confidence - Confidence level (0-1)
   * @param {string} justification - Justification for the belief
   * @param {string[]} [dependencies=[]] - List of propositions this belief depends on
   * @returns {Belief} - The added belief
   */
  addBelief(proposition, confidence, justification, dependencies = []) {
    if (this.beliefs.has(proposition)) {
      throw new Error(`Belief with proposition "${proposition}" already exists`);
    }

    // Validate all dependencies exist
    for (const dep of dependencies) {
      if (!this.beliefs.has(dep)) {
        throw new Error(`Dependency "${dep}" does not exist in the network`);
      }
    }

    const belief = new Belief(proposition, confidence, justification);
    this.beliefs.set(proposition, belief);

    // Set up dependencies
    this.dependencies.set(proposition, new Set(dependencies));
    
    // Set up reverse dependencies (who depends on this)
    for (const dep of dependencies) {
      if (!this.reverseDependencies.has(dep)) {
        this.reverseDependencies.set(dep, new Set());
      }
      this.reverseDependencies.get(dep).add(proposition);
    }

    this._logUpdate('ADD', proposition, { confidence, justification, dependencies });

    return belief;
  }

  /**
   * Update a belief and trigger reactive updates to dependent beliefs
   * @param {string} proposition - The proposition to update
   * @param {number} confidence - New confidence level (0-1)
   * @param {string} justification - New justification
   * @returns {Array<Belief>} - Array of updated beliefs (including propagated updates)
   */
  updateBelief(proposition, confidence, justification) {
    const belief = this.beliefs.get(proposition);
    if (!belief) {
      throw new Error(`Belief with proposition "${proposition}" does not exist`);
    }

    const updatedBeliefs = [];
    belief.update(confidence, justification);
    updatedBeliefs.push(belief);

    this._logUpdate('UPDATE', proposition, { confidence, justification });

    // Trigger reactive updates - only update affected beliefs
    const propagatedUpdates = this._reactiveUpdate(proposition, confidence);
    updatedBeliefs.push(...propagatedUpdates);

    return updatedBeliefs;
  }

  /**
   * Reactive update mechanism - only update beliefs affected by the change
   * This is inspired by the Reactive Knowledge Representation paper
   * @param {string} changedProposition - The proposition that was updated
   * @param {number} newConfidence - The new confidence value
   * @returns {Array<Belief>} - Array of propagated belief updates
   */
  _reactiveUpdate(changedProposition, newConfidence) {
    const updatedBeliefs = [];
    const visited = new Set();
    const queue = [changedProposition];

    while (queue.length > 0) {
      const current = queue.shift();
      
      if (visited.has(current)) {
        continue;
      }
      visited.add(current);

      // Get all beliefs that depend on this one
      const dependents = this.reverseDependencies.get(current);
      
      if (!dependents) {
        continue;
      }

      for (const dependentProposition of dependents) {
        if (visited.has(dependentProposition)) {
          continue;
        }

        const dependentBelief = this.beliefs.get(dependentProposition);
        if (!dependentBelief) {
          continue;
        }

        // Calculate new confidence based on dependencies
        // Using a simple propagation model: confidence = average of dependency confidences
        // weighted by the number of dependencies
        const deps = this.dependencies.get(dependentProposition);
        let totalConfidence = 0;
        let validDeps = 0;

        for (const dep of deps) {
          const depBelief = this.beliefs.get(dep);
          if (depBelief) {
            totalConfidence += depBelief.confidence;
            validDeps++;
          }
        }

        if (validDeps > 0) {
          const propagatedConfidence = totalConfidence / validDeps;
          
          // Only update if the change is significant (> 0.01)
          if (Math.abs(dependentBelief.confidence - propagatedConfidence) > 0.01) {
            dependentBelief.update(
              propagatedConfidence,
              `Propagated from update to "${changedProposition}"`
            );
            updatedBeliefs.push(dependentBelief);
            
            this._logUpdate('PROPAGATE', dependentProposition, {
              from: changedProposition,
              confidence: propagatedConfidence
            });

            // Add to queue for further propagation
            queue.push(dependentProposition);
          }
        }
      }
    }

    return updatedBeliefs;
  }

  /**
   * Get a belief by its proposition
   * @param {string} proposition - The proposition to look up
   * @returns {Belief|undefined} - The belief or undefined if not found
   */
  getBelief(proposition) {
    return this.beliefs.get(proposition);
  }

  /**
   * Get all beliefs that depend on a given proposition
   * @param {string} proposition - The proposition to check
   * @returns {Array<Belief>} - Array of dependent beliefs
   */
  getDependent(proposition) {
    const dependents = this.reverseDependencies.get(proposition);
    if (!dependents) {
      return [];
    }

    return Array.from(dependents)
      .map(prop => this.beliefs.get(prop))
      .filter(belief => belief !== undefined);
  }

  /**
   * Get all beliefs that a given proposition depends on
   * @param {string} proposition - The proposition to check
   * @returns {Array<Belief>} - Array of dependency beliefs
   */
  getDependencies(proposition) {
    const deps = this.dependencies.get(proposition);
    if (!deps) {
      return [];
    }

    return Array.from(deps)
      .map(prop => this.beliefs.get(prop))
      .filter(belief => belief !== undefined);
  }

  /**
   * Remove a belief and its dependencies
   * @param {string} proposition - The proposition to remove
   * @returns {boolean} - True if removed, false if not found
   */
  removeBelief(proposition) {
    const belief = this.beliefs.get(proposition);
    if (!belief) {
      return false;
    }

    // Remove from reverse dependencies of all dependencies
    const deps = this.dependencies.get(proposition);
    if (deps) {
      for (const dep of deps) {
        const reverseDeps = this.reverseDependencies.get(dep);
        if (reverseDeps) {
          reverseDeps.delete(proposition);
        }
      }
    }

    // Remove from dependencies of all dependents
    const dependents = this.reverseDependencies.get(proposition);
    if (dependents) {
      for (const dependent of dependents) {
        const depSet = this.dependencies.get(dependent);
        if (depSet) {
          depSet.delete(proposition);
        }
      }
    }

    this.beliefs.delete(proposition);
    this.dependencies.delete(proposition);
    this.reverseDependencies.delete(proposition);

    this._logUpdate('REMOVE', proposition, {});

    return true;
  }

  /**
   * Get all beliefs in the network
   * @returns {Array<Belief>} - Array of all beliefs
   */
  getAllBeliefs() {
    return Array.from(this.beliefs.values());
  }

  /**
   * Get the network structure as JSON
   * @returns {Object} - Network structure
   */
  toJSON() {
    return {
      agentId: this.agentId,
      beliefs: Array.from(this.beliefs.values()).map(b => b.toJSON()),
      dependencies: Object.fromEntries(
        Array.from(this.dependencies.entries()).map(([k, v]) => [k, Array.from(v)])
      )
    };
  }

  /**
   * Log an update event
   * @param {string} type - Type of update
   * @param {string} proposition - Proposition affected
   * @param {Object} data - Additional data
   * @private
   */
  _logUpdate(type, proposition, data) {
    this.updateLog.push({
      type,
      proposition,
      data,
      timestamp: Date.now()
    });
  }

  /**
   * Get the update log
   * @returns {Array} - Array of update events
   */
  getUpdateLog() {
    return [...this.updateLog];
  }

  /**
   * Clear the update log
   */
  clearUpdateLog() {
    this.updateLog = [];
  }

  /**
   * Get network statistics
   * @returns {Object} - Statistics object
   */
  getStats() {
    return {
      totalBeliefs: this.beliefs.size,
      totalUpdates: this.updateLog.length,
      avgConfidence: this.getAllBeliefs().reduce((sum, b) => sum + b.confidence, 0) / (this.beliefs.size || 1)
    };
  }
}

module.exports = { BeliefNetwork };
