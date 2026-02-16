/**
 * Unit tests for Persistence module
 */

const fs = require('fs');
const path = require('path');
const { Agent, BeliefNetwork, TopologyManager, Persistence } = require('../src/index');

describe('Persistence', () => {
  const testDir = path.join(__dirname, 'test-output');
  const testFile = path.join(testDir, 'test-network.json');

  beforeAll(() => {
    // Create test directory if it doesn't exist
    if (!fs.existsSync(testDir)) {
      fs.mkdirSync(testDir, { recursive: true });
    }
  });

  afterAll(() => {
    // Clean up test files
    if (fs.existsSync(testDir)) {
      fs.rmSync(testDir, { recursive: true, force: true });
    }
  });

  describe('BeliefNetwork persistence', () => {
    test('should save belief network to file', async () => {
      const network = new BeliefNetwork('test-agent');
      network.addBelief('Test belief', 0.8, 'Test justification');
      
      const result = await Persistence.save(network, testFile);
      
      expect(result).toBe(true);
      expect(fs.existsSync(testFile)).toBe(true);
    });

    test('should load belief network from file', async () => {
      const network = new BeliefNetwork('test-agent');
      network.addBelief('Test belief', 0.8, 'Test justification');
      
      await Persistence.save(network, testFile);
      const loaded = await Persistence.load(testFile, 'new-agent-id');
      
      expect(loaded.getBelief('Test belief')).toBeDefined();
      expect(loaded.getBelief('Test belief').confidence).toBe(0.8);
    });

    test('should export network to JSON string', () => {
      const network = new BeliefNetwork('test-agent');
      network.addBelief('Test belief', 0.8, 'Justification');
      
      const json = Persistence.exportNetwork(network);
      
      expect(typeof json).toBe('string');
      expect(json).toContain('test-agent');
      expect(json).toContain('Test belief');
    });

    test('should import network from JSON string', () => {
      const network = new BeliefNetwork('test-agent');
      network.addBelief('Test belief', 0.8, 'Justification');
      
      const json = Persistence.exportNetwork(network);
      const imported = Persistence.importNetwork(json, 'imported-agent');
      
      expect(imported.getBelief('Test belief')).toBeDefined();
    });
  });

  describe('Agent persistence', () => {
    test('should save agent to file', async () => {
      const agent = new Agent('agent-1', 'Test Agent');
      agent.addBelief('Test belief', 0.8, 'Justification');
      
      const agentFile = path.join(testDir, 'test-agent.json');
      const result = await Persistence.saveAgent(agent, agentFile);
      
      expect(result).toBe(true);
      expect(fs.existsSync(agentFile)).toBe(true);
    });

    test('should load agent from file', async () => {
      const agent = new Agent('agent-1', 'Test Agent');
      agent.addBelief('Test belief', 0.8, 'Justification');
      agent.subscribe('weather');
      
      const agentFile = path.join(testDir, 'test-agent2.json');
      await Persistence.saveAgent(agent, agentFile);
      
      const loaded = await Persistence.loadAgent(agentFile);
      
      expect(loaded.id).toBe('agent-1');
      expect(loaded.name).toBe('Test Agent');
      expect(loaded.getBelief('Test belief')).toBeDefined();
      expect(loaded.isSubscribed('weather')).toBe(true);
    });

    test('should export agent to JSON string', () => {
      const agent = new Agent('agent-1', 'Test Agent');
      agent.addBelief('Test belief', 0.8, 'Justification');
      
      const json = Persistence.exportAgent(agent);
      
      expect(typeof json).toBe('string');
      expect(json).toContain('agent-1');
    });

    test('should import agent from JSON', () => {
      const agent = new Agent('agent-1', 'Test Agent');
      agent.addBelief('Test belief', 0.8, 'Justification');
      
      const json = Persistence.exportAgent(agent);
      const imported = Persistence.importAgent(json);
      
      expect(imported.id).toBe('agent-1');
      expect(imported.getBelief('Test belief')).toBeDefined();
    });
  });

  describe('TopologyManager persistence', () => {
    test('should save topology to file', async () => {
      const topology = new TopologyManager();
      const agent = new Agent('agent-1', 'Test Agent');
      topology.registerAgent(agent);
      
      const topoFile = path.join(testDir, 'test-topology.json');
      const result = await Persistence.saveTopology(topology, topoFile);
      
      expect(result).toBe(true);
    });

    test('should load topology from file', async () => {
      const agent = new Agent('agent-1', 'Test Agent');
      const topology = new TopologyManager();
      topology.registerAgent(agent);
      topology.setRelevanceRules('agent-1', ['Test']);
      
      const topoFile = path.join(testDir, 'test-topology2.json');
      await Persistence.saveTopology(topology, topoFile);
      
      const agent2 = new Agent('agent-1', 'Test Agent');
      const loaded = await Persistence.loadTopology(topoFile, [agent2]);
      
      expect(loaded.agents.size).toBe(1);
    });
  });

  describe('Simulation persistence', () => {
    test('should save complete simulation state', async () => {
      const agent1 = new Agent('agent-1', 'Agent One');
      const agent2 = new Agent('agent-2', 'Agent Two');
      const topology = new TopologyManager();
      
      topology.registerAgent(agent1);
      topology.registerAgent(agent2);
      agent1.addBelief('Test', 0.8, 'Justification');
      
      const simFile = path.join(testDir, 'test-simulation.json');
      const result = await Persistence.saveSimulation([agent1, agent2], topology, simFile);
      
      expect(result).toBe(true);
    });

    test('should load complete simulation state', async () => {
      const agent1 = new Agent('agent-1', 'Agent One');
      const agent2 = new Agent('agent-2', 'Agent Two');
      const topology = new TopologyManager();
      
      topology.registerAgent(agent1);
      topology.registerAgent(agent2);
      agent1.addBelief('Test', 0.8, 'Justification');
      
      const simFile = path.join(testDir, 'test-simulation2.json');
      await Persistence.saveSimulation([agent1, agent2], topology, simFile);
      
      const { agents, topologyManager } = await Persistence.loadSimulation(simFile);
      
      expect(agents).toHaveLength(2);
      expect(agents[0].getBelief('Test')).toBeDefined();
    });

    test('should export simulation to JSON', () => {
      const agent = new Agent('agent-1', 'Test Agent');
      const topology = new TopologyManager();
      topology.registerAgent(agent);
      
      const json = Persistence.exportSimulation([agent], topology);
      
      expect(typeof json).toBe('string');
      expect(json).toContain('agent-1');
    });
  });
});
