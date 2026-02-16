/**
 * Simulator Class
 * 
 * Orchestrates multi-agent simulation with time steps and belief updates.
 * Manages the simulation lifecycle including initialization, execution,
 * and history tracking.
 */

const { Agent } = require('./Agent');
const { UpdateTriggerManager } = require('./UpdateTrigger');

class Simulator {
  /**
   * Create a new Simulator
   * @param {Object} [options={}] - Simulation options
   * @param {number} [options.maxTimeSteps=100] - Maximum number of time steps
   * @param {number} [options.tickDelay=1000] - Delay between time steps (ms)
   */
  constructor(options = {}) {
    this.options = {
      maxTimeSteps: options.maxTimeSteps || 100,
      tickDelay: options.tickDelay || 1000
    };
    
    this.agents = new Map(); // id -> Agent
    this.triggerManager = new UpdateTriggerManager();
    this.currentTimeStep = 0;
    this.isRunning = false;
    this.history = []; // Array of simulation states at each time step
    this.listeners = new Map(); // event -> Array of callbacks
    
    this._boundTick = this._tick.bind(this);
    this._intervalId = null;
  }

  /**
   * Add an agent to the simulation
   * @param {Agent|string} agent - Agent instance or agent ID (if name provided)
   * @param {string} [name] - Agent name (required if agent is string ID)
   * @returns {Agent} - The added agent
   */
  addAgent(agent, name) {
    let addedAgent;
    
    if (typeof agent === 'string') {
      // Agent ID provided, create new agent
      if (!name) {
        throw new Error('Name is required when adding agent by ID');
      }
      addedAgent = new Agent(agent, name);
    } else {
      // Agent instance provided
      addedAgent = agent;
    }

    if (this.agents.has(addedAgent.id)) {
      throw new Error(`Agent with ID "${addedAgent.id}" already exists`);
    }

    this.agents.set(addedAgent.id, addedAgent);
    
    // Register listener for belief updates
    this.triggerManager.addListener('*', (trigger) => {
      this._handleTrigger(addedAgent, trigger);
    });

    this._emit('agentAdded', { agent: addedAgent });

    return addedAgent;
  }

  /**
   * Remove an agent from the simulation
   * @param {string} agentId - ID of the agent to remove
   * @returns {boolean} - True if removed, false if not found
   */
  removeAgent(agentId) {
    const agent = this.agents.get(agentId);
    if (!agent) {
      return false;
    }

    this.agents.delete(agentId);
    this._emit('agentRemoved', { agentId });

    return true;
  }

  /**
   * Get an agent by ID
   * @param {string} agentId - Agent ID
   * @returns {Agent|undefined} - The agent or undefined
   */
  getAgent(agentId) {
    return this.agents.get(agentId);
  }

  /**
   * Get all agents
   * @returns {Array<Agent>} - Array of all agents
   */
  getAllAgents() {
    return Array.from(this.agents.values());
  }

  /**
   * Run the simulation
   * @param {Object} [config={}] - Run configuration
   * @param {number} [config.timeSteps] - Number of time steps to run (default: maxTimeSteps)
   * @param {boolean} [config.async=false] - Run asynchronously (returns Promise)
   * @returns {Promise<Array>|Array} - History if async, void if sync
   */
  run(config = {}) {
    const timeSteps = config.timeSteps || this.options.maxTimeSteps;
    
    if (config.async) {
      return this._runAsync(timeSteps);
    }

    // Synchronous run
    this.isRunning = true;
    this._emit('simulationStart', { timeSteps });

    for (let i = 0; i < timeSteps && this.isRunning; i++) {
      this._tick();
      this.currentTimeStep++;
      
      // Record state
      this._recordState();
    }

    this.isRunning = false;
    this._emit('simulationComplete', { 
      totalTimeSteps: this.currentTimeStep,
      historyLength: this.history.length
    });

    return this.history;
  }

  /**
   * Run simulation asynchronously
   * @param {number} timeSteps - Number of time steps
   * @returns {Promise<Array>} - Promise resolving to history
   */
  async _runAsync(timeSteps) {
    this.isRunning = true;
    this._emit('simulationStart', { timeSteps });

    return new Promise((resolve) => {
      const runLoop = () => {
        if (!this.isRunning || this.currentTimeStep >= timeSteps) {
          this.isRunning = false;
          this._emit('simulationComplete', {
            totalTimeStep: this.currentTimeStep,
            historyLength: this.history.length
          });
          resolve(this.history);
          return;
        }

        this._tick();
        this.currentTimeStep++;
        this._recordState();

        this._intervalId = setTimeout(runLoop, this.options.tickDelay);
      };

      runLoop();
    });
  }

  /**
   * Pause the simulation (for async runs)
   */
  pause() {
    if (this._intervalId) {
      clearTimeout(this._intervalId);
      this._intervalId = null;
    }
    this.isRunning = false;
    this._emit('simulationPaused', { currentTimeStep: this.currentTimeStep });
  }

  /**
   * Resume a paused simulation
   */
  resume() {
    if (!this.isRunning && this.currentTimeStep < this.options.maxTimeSteps) {
      this._runAsync(this.options.maxTimeSteps);
    }
  }

  /**
   * Stop the simulation
   */
  stop() {
    this.pause();
    this.currentTimeStep = 0;
    this._emit('simulationStopped', {});
  }

  /**
   * Execute a single time step
   * @private
   */
  _tick() {
    this._emit('timeStepStart', { timeStep: this.currentTimeStep });

    // Process triggers for each agent
    for (const agent of this.agents.values()) {
      this._processAgentBeliefs(agent);
    }

    this._emit('timeStepEnd', { timeStep: this.currentTimeStep });
  }

  /**
   * Process agent beliefs for a time step
   * @param {Agent} agent - Agent to process
   * @private
   */
  _processAgentBeliefs(agent) {
    const beliefs = agent.getBeliefs();
    
    for (const belief of beliefs) {
      // Simulate natural belief decay or reinforcement
      if (Math.random() < 0.1) {
        const change = (Math.random() - 0.5) * 0.05;
        const newConfidence = Math.max(0, Math.min(1, belief.confidence + change));
        
        if (Math.abs(newConfidence - belief.confidence) > 0.01) {
          const trigger = this.triggerManager.trigger(
            'observation',
            belief.proposition,
            {
              agentId: agent.id,
              previousConfidence: belief.confidence,
              newConfidence,
              reason: 'natural_fluctuation'
            }
          );
          
          agent.updateBelief(
            belief.proposition,
            newConfidence,
            `Observed change at time step ${this.currentTimeStep}`
          );
        }
      }
    }
  }

  /**
   * Handle a trigger event for an agent
   * @param {Agent} agent - Target agent
   * @param {UpdateTrigger} trigger - Trigger event
   * @private
   */
  _handleTrigger(agent, trigger) {
    if (!trigger.data || !trigger.data.agentId) return;
    if (trigger.data.agentId !== agent.id) return;

    const { proposition, confidence, justification } = trigger.data;
    
    if (proposition && confidence !== undefined) {
      agent.updateBelief(proposition, confidence, justification);
    }
  }

  /**
   * Record current simulation state
   * @private
   */
  _recordState() {
    const state = {
      timeStep: this.currentTimeStep,
      timestamp: Date.now(),
      agents: {}
    };

    for (const agent of this.agents.values()) {
      state.agents[agent.id] = {
        name: agent.name,
        beliefs: agent.getBeliefs().map(b => b.toJSON())
      };
    }

    this.history.push(state);
  }

  /**
   * Get simulation history
   * @param {Object} [filters={}] - Optional filters
   * @param {number} [filters.from] - Start time step
   * @param {number} [filters.to] - End time step
   * @returns {Array} - History array
   */
  getHistory(filters = {}) {
    if (filters.from !== undefined || filters.to !== undefined) {
      const from = filters.from || 0;
      const to = filters.to || this.history.length;
      return this.history.slice(from, to);
    }
    return [...this.history];
  }

  /**
   * Get current belief states for all agents
   * @returns {Object} - Current belief states
   */
  getBeliefStates() {
    const states = {};
    
    for (const agent of this.agents.values()) {
      states[agent.id] = {
        name: agent.name,
        beliefs: agent.getBeliefs().map(b => ({
          proposition: b.proposition,
          confidence: b.confidence,
          justification: b.justification
        }))
      };
    }

    return states;
  }

  /**
   * Get belief states at a specific time step
   * @param {number} timeStep - Time step to query
   * @returns {Object|null} - Belief states at time step
   */
  getBeliefStatesAt(timeStep) {
    const state = this.history.find(s => s.timeStep === timeStep);
    return state ? state.agents : null;
  }

  /**
   * Get simulation statistics
   * @returns {Object} - Statistics
   */
  getStats() {
    return {
      currentTimeStep: this.currentTimeStep,
      agentCount: this.agents.size,
      historyLength: this.history.length,
      isRunning: this.isRunning,
      triggerStats: this.triggerManager.getStats()
    };
  }

  /**
   * Add event listener
   * @param {string} event - Event name
   * @param {Function} callback - Callback function
   */
  on(event, callback) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, []);
    }
    this.listeners.get(event).push(callback);
  }

  /**
   * Remove event listener
   * @param {string} event - Event name
   * @param {Function} callback - Callback to remove
   */
  off(event, callback) {
    if (this.listeners.has(event)) {
      const callbacks = this.listeners.get(event);
      const index = callbacks.indexOf(callback);
      if (index > -1) {
        callbacks.splice(index, 1);
      }
    }
  }

  /**
   * Emit an event
   * @param {string} event - Event name
   * @param {Object} data - Event data
   * @private
   */
  _emit(event, data) {
    if (this.listeners.has(event)) {
      for (const callback of this.listeners.get(event)) {
        try {
          callback(data);
        } catch (error) {
          console.error(`Error in ${event} listener:`, error);
        }
      }
    }
  }

  /**
   * Reset the simulation
   */
  reset() {
    this.stop();
    this.history = [];
    this.currentTimeStep = 0;
    this.triggerManager.clearHistory();
  }
}

module.exports = { Simulator };
