/**
 * Visualizer Class
 * 
 * Console output of belief states with color-coded confidence levels.
 * Provides visualization methods for simulation states and comparisons.
 */

class Visualizer {
  /**
   * Create a new Visualizer
   * @param {Object} [options={}] - Visualization options
   * @param {boolean} [options.useColors=true] - Enable color output
   * @param {boolean} [options.showTimestamps=true] - Show timestamps
   * @param {number} [options.indentSize=2] - Indentation size
   */
  constructor(options = {}) {
    this.options = {
      useColors: options.useColors !== false,
      showTimestamps: options.showTimestamps !== false,
      indentSize: options.indentSize || 2
    };
    
    // ANSI color codes
    this.colors = {
      reset: '\x1b[0m',
      bright: '\x1b[1m',
      dim: '\x1b[2m',
      
      // Confidence colors (green = high, yellow = medium, red = low)
      confidenceHigh: '\x1b[32m',   // Green - high confidence (0.7-1.0)
      confidenceMedium: '\x1b[33m', // Yellow - medium confidence (0.3-0.7)
      confidenceLow: '\x1b[31m',    // Red - low confidence (0.0-0.3)
      
      // Agent colors (cycling through)
      agent1: '\x1b[36m',   // Cyan
      agent2: '\x1b[35m',   // Magenta
      agent3: '\x1b[34m',   // Blue
      agent4: '\x1b[33m',   // Yellow
      agent5: '\x1b[32m',   // Green
      
      // Info colors
      info: '\x1b[36m',
      warning: '\x1b[33m',
      error: '\x1b[31m',
      success: '\x1b[32m',
      
      // Structure colors
      header: '\x1b[1;37m',
      subheader: '\x1b[36m',
      border: '\x1b[90m'
    };
    
    this.agentColors = [
      this.colors.agent1,
      this.colors.agent2,
      this.colors.agent3,
      this.colors.agent4,
      this.colors.agent5
    ];
  }

  /**
   * Get color for confidence level
   * @param {number} confidence - Confidence level (0-1)
   * @returns {string} - ANSI color code
   * @private
   */
  _getConfidenceColor(confidence) {
    if (!this.options.useColors) return '';
    
    if (confidence >= 0.7) return this.colors.confidenceHigh;
    if (confidence >= 0.3) return this.colors.confidenceMedium;
    return this.colors.confidenceLow;
  }

  /**
   * Get color for agent
   * @param {number} index - Agent index
   * @returns {string} - ANSI color code
   * @private
   */
  _getAgentColor(index) {
    if (!this.options.useColors) return '';
    return this.agentColors[index % this.agentColors.length];
  }

  /**
   * Create indentation string
   * @param {number} level - Indentation level
   * @returns {string} - Indentation string
   * @private
   */
  _indent(level = 0) {
    return ' '.repeat(level * this.options.indentSize);
  }

  /**
   * Format a confidence value with color
   * @param {number} confidence - Confidence value
   * @returns {string} - Formatted confidence string
   * @private
   */
  _formatConfidence(confidence) {
    const color = this._getConfidenceColor(confidence);
    const reset = this.options.useColors ? this.colors.reset : '';
    
    const percentage = (confidence * 100).toFixed(1) + '%';
    return `${color}${percentage}${reset}`;
  }

  /**
   * Visualize the current state of a simulator
   * @param {Simulator} simulator - The simulator to visualize
   * @param {Object} [options={}] - Visualization options
   * @param {number} [options.timeStep] - Specific time step to show (index in history)
   */
  visualize(simulator, options = {}) {
    const timeStep = options.timeStep;
    
    // If timeStep specified, try to get from history
    if (timeStep !== undefined) {
      const states = simulator.getBeliefStatesAt(timeStep);
      if (states) {
        this.printState(states, { title: `Time Step ${timeStep}` });
        return;
      }
      // Fall back to current state if not in history
    }
    
    // Default: show current belief states
    const states = simulator.getBeliefStates();
    this.printState(states, { title: `Current State (Step ${simulator.currentTimeStep})` });
  }

  /**
   * Print a belief state to console
   * @param {Object} states - Belief states
   * @param {Object} [options={}] - Print options
   * @param {string} [options.title] - Section title
   */
  printState(states, options = {}) {
    const { title = 'Belief States' } = options;
    
    this._printHeader(title);
    
    const agentIds = Object.keys(states);
    
    for (let i = 0; i < agentIds.length; i++) {
      const agentId = agentIds[i];
      const agentData = states[agentId];
      const agentColor = this._getAgentColor(i);
      const reset = this.options.useColors ? this.colors.reset : '';
      
      console.log(`${this._indent(1)}${agentColor}${agentData.name}${reset} (${agentId}):`);
      
      if (!agentData.beliefs || agentData.beliefs.length === 0) {
        console.log(`${this._indent(2)}No beliefs`);
        continue;
      }
      
      for (const belief of agentData.beliefs) {
        const confidenceStr = this._formatConfidence(belief.confidence);
        
        // Truncate long propositions and justifications
        const prop = belief.proposition.length > 40
          ? belief.proposition.substring(0, 37) + '...'
          : belief.proposition;
        
        console.log(`${this._indent(2)}• ${prop}`);
        console.log(`${this._indent(3)}Confidence: ${confidenceStr}`);
        
        if (this.options.useColors) {
          const just = belief.justification.length > 50
            ? belief.justification.substring(0, 47) + '...'
            : belief.justification;
          console.log(`${this._indent(3)}Justification: ${just}`);
        }
      }
      
      console.log('');
    }
  }

  /**
   * Print comparison between two time steps
   * @param {Simulator} simulator - The simulator
   * @param {number} fromTimeStep - Starting time step
   * @param {number} toTimeStep - Ending time step
   * @param {Object} [options={}] - Print options
   */
  printComparison(simulator, fromTimeStep, toTimeStep, options = {}) {
    const fromStates = simulator.getBeliefStatesAt(fromTimeStep);
    const toStates = simulator.getBeliefStatesAt(toTimeStep);
    
    if (!fromStates || !toStates) {
      this._printError('Invalid time step range');
      return;
    }

    const { title = 'Belief Comparison' } = options;
    
    this._printHeader(title);
    console.log(`${this._indent(1)}From Step ${fromTimeStep} → To Step ${toTimeStep}`);
    console.log('');

    // Compare each agent
    const allAgentIds = new Set([...Object.keys(fromStates), ...Object.keys(toStates)]);
    let agentIndex = 0;
    
    for (const agentId of allAgentIds) {
      const fromAgent = fromStates[agentId];
      const toAgent = toStates[agentId];
      
      const agentColor = this._getAgentColor(agentIndex);
      const reset = this.options.useColors ? this.colors.reset : '';
      
      console.log(`${this._indent(1)}${agentColor}${toAgent?.name || fromAgent?.name || agentId}${reset}:`);
      
      // Build belief maps for comparison
      const fromBeliefs = new Map(
        (fromAgent?.beliefs || []).map(b => [b.proposition, b])
      );
      const toBeliefs = new Map(
        (toAgent?.beliefs || []).map(b => [b.proposition, b])
      );
      
      // All unique propositions
      const allPropositions = new Set([...fromBeliefs.keys(), ...toBeliefs.keys()]);
      
      for (const proposition of allPropositions) {
        const fromBelief = fromBeliefs.get(proposition);
        const toBelief = toBeliefs.get(proposition);
        
        const fromConf = fromBelief?.confidence ?? 'N/A';
        const toConf = toBelief?.confidence ?? 'N/A';
        
        let changeIndicator = '';
        if (fromBelief && toBelief) {
          const diff = toBelief.confidence - fromBelief.confidence;
          if (Math.abs(diff) > 0.01) {
            const arrow = diff > 0 ? '↑' : '↓';
            const diffColor = diff > 0 ? this.colors.success : this.colors.error;
            const color = this.options.useColors ? diffColor : '';
            changeIndicator = ` ${color}${arrow} ${(diff * 100).toFixed(1)}%${reset}`;
          } else {
            changeIndicator = ' =';
          }
        } else if (!fromBelief && toBelief) {
          changeIndicator = this.options.useColors 
            ? ` ${this.colors.success}+${((toBelief.confidence) * 100).toFixed(1)}%`
            : ' +new';
        } else if (fromBelief && !toBelief) {
          changeIndicator = ' -gone';
        }
        
        const prop = proposition.length > 35
          ? proposition.substring(0, 32) + '...'
          : proposition;
        
        const fromStr = typeof fromConf === 'number' ? this._formatConfidence(fromConf) : fromConf;
        const toStr = typeof toConf === 'number' ? this._formatConfidence(toConf) : toConf;
        
        console.log(`${this._indent(2)}• ${prop}`);
        console.log(`${this._indent(3)}${fromStr} → ${toStr}${changeIndicator}`);
      }
      
      console.log('');
      agentIndex++;
    }
  }

  /**
   * Print simulation statistics
   * @param {Simulator} simulator - The simulator
   */
  printStats(simulator) {
    const stats = simulator.getStats();
    
    this._printHeader('Simulation Statistics');
    
    console.log(`${this._indent(1)}Current Time Step: ${stats.currentTimeStep}`);
    console.log(`${this._indent(1)}Agent Count: ${stats.agentCount}`);
    console.log(`${this._indent(1)}History Length: ${stats.historyLength}`);
    console.log(`${this._indent(1)}Is Running: ${stats.isRunning}`);
    
    console.log('');
    this._printSubheader('Trigger Statistics');
    
    const triggerStats = stats.triggerStats;
    for (const [type, data] of Object.entries(triggerStats)) {
      console.log(`${this._indent(1)}${type}:`);
      console.log(`${this._indent(2)}Registered: ${data.registered}`);
      console.log(`${this._indent(2)}Triggered: ${data.triggered}`);
    }
  }

  /**
   * Print a horizontal separator line
   * @param {string} [char='─'] - Character to use
   */
  printSeparator(char = '─') {
    const width = 60;
    const line = char.repeat(width);
    const color = this.options.useColors ? this.colors.border : '';
    const reset = this.options.useColors ? this.colors.reset : '';
    console.log(color + line + reset);
  }

  /**
   * Print a header
   * @param {string} text - Header text
   * @private
   */
  _printHeader(text) {
    this.printSeparator();
    const color = this.options.useColors ? this.colors.header : '';
    const reset = this.options.useColors ? this.colors.reset : '';
    console.log(`${color}${text}${reset}`);
    this.printSeparator();
  }

  /**
   * Print a subheader
   * @param {string} text - Subheader text
   * @private
   */
  _printSubheader(text) {
    const color = this.options.useColors ? this.colors.subheader : '';
    const reset = this.options.useColors ? this.colors.reset : '';
    console.log(`${color}${text}${reset}`);
  }

  /**
   * Print an error message
   * @param {string} message - Error message
   * @private
   */
  _printError(message) {
    const color = this.options.useColors ? this.colors.error : '';
    const reset = this.options.useColors ? this.colors.reset : '';
    console.error(`${color}Error: ${message}${reset}`);
  }

  /**
   * Print belief evolution over time for a specific proposition
   * @param {Simulator} simulator - The simulator
   * @param {string} proposition - Proposition to track
   * @param {string} [agentId] - Optional agent ID (shows all if not specified)
   */
  printEvolution(simulator, proposition, agentId) {
    const history = simulator.getHistory();
    
    this._printHeader(`Belief Evolution: "${proposition}"`);
    
    // Print header row
    console.log(`${this._indent(1)}Time Step | Agent | Confidence`);
    console.log(`${this._indent(1)}${'─'.repeat(8)} | ${'─'.repeat(6)} | ${'─'.repeat(10)}`);
    
    for (const state of history) {
      const agents = agentId 
        ? [{ id: agentId, ...state.agents[agentId] }].filter(a => a.beliefs)
        : Object.entries(state.agents).map(([id, data]) => ({ id, ...data }));
      
      for (const agent of agents) {
        const belief = agent.beliefs?.find(b => b.proposition === proposition);
        if (belief) {
          const agentColor = this._getAgentColor(Object.keys(state.agents).indexOf(agent.id));
          const confColor = this._getConfidenceColor(belief.confidence);
          const reset = this.options.useColors ? this.colors.reset : '';
          const agentReset = this.options.useColors ? this.colors.reset : '';
          
          const step = String(state.timeStep).padEnd(10);
          const name = (agent.name || agent.id).substring(0, 6).padEnd(6);
          
          if (this.options.useColors) {
            console.log(`${this._indent(1)}${step} | ${agentColor}${name}${agentReset} | ${confColor}${(belief.confidence * 100).toFixed(1)}%${reset}`);
          } else {
            console.log(`${this._indent(1)}${step} | ${name} | ${(belief.confidence * 100).toFixed(1)}%`);
          }
        }
      }
    }
  }

  /**
   * Create a simple text-based bar chart
   * @param {number} value - Value (0-1)
   * @param {number} width - Bar width
   * @returns {string} - Bar string
   */
  createBar(value, width = 20) {
    const filled = Math.round(value * width);
    const empty = width - filled;
    const color = this._getConfidenceColor(value);
    const reset = this.options.useColors ? this.colors.reset : '';
    
    if (this.options.useColors) {
      return `${color}${'█'.repeat(filled)}${'░'.repeat(empty)}${reset}`;
    }
    return `${'#'.repeat(filled)}${'-'.repeat(empty)}`;
  }
}

module.exports = { Visualizer };
