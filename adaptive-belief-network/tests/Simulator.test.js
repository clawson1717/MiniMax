/**
 * Unit tests for Simulator class
 */

const { Agent, Simulator } = require('../src/index');

describe('Simulator', () => {
  let simulator;

  beforeEach(() => {
    simulator = new Simulator({ maxTimeSteps: 10, tickDelay: 10 });
  });

  describe('constructor', () => {
    test('should create simulator with default options', () => {
      const defaultSim = new Simulator();
      
      expect(defaultSim.options.maxTimeSteps).toBe(100);
      expect(defaultSim.options.tickDelay).toBe(1000);
      expect(defaultSim.agents.size).toBe(0);
      expect(defaultSim.currentTimeStep).toBe(0);
      expect(defaultSim.isRunning).toBe(false);
      expect(defaultSim.history).toHaveLength(0);
    });

    test('should create simulator with custom options', () => {
      expect(simulator.options.maxTimeSteps).toBe(10);
      expect(simulator.options.tickDelay).toBe(10);
    });
  });

  describe('addAgent', () => {
    test('should add agent instance', () => {
      const agent = new Agent('agent-1', 'Test Agent');
      const addedAgent = simulator.addAgent(agent);
      
      expect(simulator.agents.size).toBe(1);
      expect(simulator.getAgent('agent-1')).toBe(agent);
      expect(addedAgent).toBe(agent);
    });

    test('should create agent from ID and name', () => {
      const addedAgent = simulator.addAgent('agent-1', 'Test Agent');
      
      expect(simulator.agents.size).toBe(1);
      expect(addedAgent.id).toBe('agent-1');
      expect(addedAgent.name).toBe('Test Agent');
    });

    test('should throw error when adding agent without name', () => {
      expect(() => {
        simulator.addAgent('agent-1');
      }).toThrow('Name is required');
    });

    test('should throw error for duplicate agent ID', () => {
      const agent = new Agent('agent-1', 'Test Agent');
      simulator.addAgent(agent);
      
      expect(() => {
        simulator.addAgent(agent);
      }).toThrow('already exists');
    });

    test('should emit agentAdded event', () => {
      const callback = jest.fn();
      simulator.on('agentAdded', callback);
      
      const agent = new Agent('agent-1', 'Test Agent');
      simulator.addAgent(agent);
      
      expect(callback).toHaveBeenCalledWith({ agent });
    });
  });

  describe('removeAgent', () => {
    test('should remove agent from simulation', () => {
      const agent = new Agent('agent-1', 'Test Agent');
      simulator.addAgent(agent);
      
      const result = simulator.removeAgent('agent-1');
      
      expect(result).toBe(true);
      expect(simulator.agents.size).toBe(0);
    });

    test('should return false for non-existent agent', () => {
      const result = simulator.removeAgent('non-existent');
      
      expect(result).toBe(false);
    });

    test('should emit agentRemoved event', () => {
      const agent = new Agent('agent-1', 'Test Agent');
      simulator.addAgent(agent);
      
      const callback = jest.fn();
      simulator.on('agentRemoved', callback);
      
      simulator.removeAgent('agent-1');
      
      expect(callback).toHaveBeenCalledWith({ agentId: 'agent-1' });
    });
  });

  describe('getAgent', () => {
    test('should return agent by ID', () => {
      const agent = new Agent('agent-1', 'Test Agent');
      simulator.addAgent(agent);
      
      const retrieved = simulator.getAgent('agent-1');
      
      expect(retrieved).toBe(agent);
    });

    test('should return undefined for non-existent agent', () => {
      const retrieved = simulator.getAgent('non-existent');
      
      expect(retrieved).toBeUndefined();
    });
  });

  describe('getAllAgents', () => {
    test('should return all agents', () => {
      const agent1 = new Agent('agent-1', 'Agent One');
      const agent2 = new Agent('agent-2', 'Agent Two');
      simulator.addAgent(agent1);
      simulator.addAgent(agent2);
      
      const agents = simulator.getAllAgents();
      
      expect(agents).toHaveLength(2);
      expect(agents).toContain(agent1);
      expect(agents).toContain(agent2);
    });
  });

  describe('run', () => {
    test('should run simulation for specified time steps', () => {
      const agent = new Agent('agent-1', 'Test Agent');
      simulator.addAgent(agent);
      
      const history = simulator.run({ timeSteps: 5 });
      
      expect(history).toHaveLength(5);
      expect(simulator.currentTimeStep).toBe(5);
    });

    test('should record agent states in history', () => {
      const agent = new Agent('agent-1', 'Test Agent');
      agent.addBelief('Test belief', 0.8, 'Initial');
      simulator.addAgent(agent);
      
      const history = simulator.run({ timeSteps: 3 });
      
      expect(history[0].agents['agent-1']).toBeDefined();
      expect(history[0].agents['agent-1'].beliefs).toHaveLength(1);
    });

    test('should emit simulationStart event', () => {
      const callback = jest.fn();
      simulator.on('simulationStart', callback);
      
      simulator.run({ timeSteps: 1 });
      
      expect(callback).toHaveBeenCalledWith({ timeSteps: 1 });
    });

    test('should emit simulationComplete event', () => {
      const callback = jest.fn();
      simulator.on('simulationComplete', callback);
      
      simulator.run({ timeSteps: 3 });
      
      expect(callback).toHaveBeenCalledWith({
        totalTimeSteps: 3,
        historyLength: 3
      });
    });

    test('should emit timeStep events', () => {
      const startCallback = jest.fn();
      const endCallback = jest.fn();
      simulator.on('timeStepStart', startCallback);
      simulator.on('timeStepEnd', endCallback);
      
      simulator.run({ timeSteps: 2 });
      
      expect(startCallback).toHaveBeenCalledTimes(2);
      expect(endCallback).toHaveBeenCalledTimes(2);
    });

    test('should respect maxTimeSteps as default when no timeSteps provided', () => {
      const limitedSim = new Simulator({ maxTimeSteps: 5, tickDelay: 0 });
      const agent = new Agent('agent-1', 'Test Agent');
      limitedSim.addAgent(agent);
      
      // Should only run 5 time steps (default) when no timeSteps specified
      const history = limitedSim.run();
      
      expect(history).toHaveLength(5);
    });

    test('should handle async run', async () => {
      const agent = new Agent('agent-1', 'Test Agent');
      simulator.addAgent(agent);
      
      const history = await simulator.run({ timeSteps: 3, async: true });
      
      expect(history).toHaveLength(3);
    });
  });

  describe('pause and resume', () => {
    test('should pause running simulation', () => {
      simulator.isRunning = true;
      
      simulator.pause();
      
      expect(simulator.isRunning).toBe(false);
    });

    test('should emit simulationPaused event', () => {
      simulator.isRunning = true;
      const callback = jest.fn();
      simulator.on('simulationPaused', callback);
      
      simulator.pause();
      
      expect(callback).toHaveBeenCalledWith({ currentTimeStep: 0 });
    });

    test('should stop simulation and reset time step', () => {
      simulator.currentTimeStep = 5;
      simulator.isRunning = true;
      
      simulator.stop();
      
      expect(simulator.currentTimeStep).toBe(0);
      expect(simulator.isRunning).toBe(false);
    });

    test('should emit simulationStopped event', () => {
      const callback = jest.fn();
      simulator.on('simulationStopped', callback);
      
      simulator.stop();
      
      expect(callback).toHaveBeenCalledWith({});
    });
  });

  describe('getHistory', () => {
    test('should return full history', () => {
      const agent = new Agent('agent-1', 'Test Agent');
      simulator.addAgent(agent);
      
      simulator.run({ timeSteps: 5 });
      
      const history = simulator.getHistory();
      
      expect(history).toHaveLength(5);
    });

    test('should filter history by time range', () => {
      const agent = new Agent('agent-1', 'Test Agent');
      simulator.addAgent(agent);
      
      simulator.run({ timeSteps: 10 });
      
      // Time steps recorded are 1-10 (history[0].timeStep = 1, etc.)
      // getHistory uses slice(from, to) on array indices
      // from: 2, to: 5 gives indices 2, 3, 4 (3 elements) with timeSteps 3, 4, 5
      const filtered = simulator.getHistory({ from: 2, to: 5 });
      
      expect(filtered).toHaveLength(3);
      expect(filtered[0].timeStep).toBe(3);
      expect(filtered[2].timeStep).toBe(5);
    });
  });

  describe('getBeliefStates', () => {
    test('should return current belief states for all agents', () => {
      const agent = new Agent('agent-1', 'Test Agent');
      agent.addBelief('Belief 1', 0.8, 'Reason');
      agent.addBelief('Belief 2', 0.6, 'Reason');
      simulator.addAgent(agent);
      
      simulator.run({ timeSteps: 1 });
      
      const states = simulator.getBeliefStates();
      
      expect(states['agent-1']).toBeDefined();
      expect(states['agent-1'].beliefs).toHaveLength(2);
    });
  });

  describe('getBeliefStatesAt', () => {
    test('should return belief states at specific time step', () => {
      const agent = new Agent('agent-1', 'Test Agent');
      agent.addBelief('Test', 0.8, 'Reason');
      simulator.addAgent(agent);
      
      simulator.run({ timeSteps: 5 });
      
      const states = simulator.getBeliefStatesAt(2);
      
      expect(states).toBeDefined();
      expect(states['agent-1']).toBeDefined();
    });

    test('should return null for non-existent time step', () => {
      const agent = new Agent('agent-1', 'Test Agent');
      simulator.addAgent(agent);
      
      simulator.run({ timeSteps: 3 });
      
      const states = simulator.getBeliefStatesAt(100);
      
      expect(states).toBeNull();
    });
  });

  describe('getStats', () => {
    test('should return simulation statistics', () => {
      const agent = new Agent('agent-1', 'Test Agent');
      simulator.addAgent(agent);
      
      simulator.run({ timeSteps: 5 });
      
      const stats = simulator.getStats();
      
      expect(stats.currentTimeStep).toBe(5);
      expect(stats.agentCount).toBe(1);
      expect(stats.historyLength).toBe(5);
      expect(stats.isRunning).toBe(false);
    });
  });

  describe('event listeners', () => {
    test('should add and remove event listeners', () => {
      const callback = jest.fn();
      
      simulator.on('testEvent', callback);
      simulator._emit('testEvent', { data: 'test' });
      
      expect(callback).toHaveBeenCalledWith({ data: 'test' });
      
      simulator.off('testEvent', callback);
      simulator._emit('testEvent', { data: 'test' });
      
      expect(callback).toHaveBeenCalledTimes(1);
    });

    test('should handle errors in event listeners gracefully', () => {
      const errorCallback = () => {
        throw new Error('Test error');
      };
      
      // Should not throw
      simulator.on('testEvent', errorCallback);
      expect(() => {
        simulator._emit('testEvent', {});
      }).not.toThrow();
    });
  });

  describe('reset', () => {
    test('should reset simulation state', () => {
      const agent = new Agent('agent-1', 'Test Agent');
      simulator.addAgent(agent);
      
      simulator.run({ timeSteps: 5 });
      simulator.reset();
      
      expect(simulator.currentTimeStep).toBe(0);
      expect(simulator.history).toHaveLength(0);
      expect(simulator.isRunning).toBe(false);
    });
  });

  describe('belief processing', () => {
    test('should process agent beliefs during time steps', () => {
      const agent = new Agent('agent-1', 'Test Agent');
      agent.addBelief('Stable belief', 0.8, 'Strong reason');
      simulator.addAgent(agent);
      
      // Run simulation - some beliefs may fluctuate slightly due to random processing
      const history = simulator.run({ timeSteps: 3 });
      
      // History should be recorded
      expect(history).toHaveLength(3);
      expect(history[0].agents['agent-1']).toBeDefined();
    });
  });
});

describe('Simulator Integration', () => {
  test('should run multi-agent simulation', () => {
    const simulator = new Simulator({ maxTimeSteps: 5, tickDelay: 0 });
    
    const agent1 = new Agent('agent-1', 'Agent One');
    const agent2 = new Agent('agent-2', 'Agent Two');
    
    agent1.addBelief('Weather is good', 0.9, 'Sunny forecast');
    agent2.addBelief('Weather is good', 0.7, 'Clear sky');
    
    simulator.addAgent(agent1);
    simulator.addAgent(agent2);
    
    const history = simulator.run();
    
    expect(history).toHaveLength(5);
    expect(simulator.getStats().agentCount).toBe(2);
  });

  test('should handle agent removal during simulation', () => {
    const simulator = new Simulator({ maxTimeSteps: 5, tickDelay: 0 });
    
    const agent1 = new Agent('agent-1', 'Agent One');
    const agent2 = new Agent('agent-2', 'Agent Two');
    
    simulator.addAgent(agent1);
    simulator.addAgent(agent2);
    
    simulator.run({ timeSteps: 2 });
    
    simulator.removeAgent('agent-2');
    
    const history = simulator.run({ timeSteps: 3 });
    
    expect(simulator.getStats().agentCount).toBe(1);
  });
});
