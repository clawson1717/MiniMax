/**
 * Unit tests for TopologyManager class
 */

const { Agent, TopologyManager } = require('../src/index');

describe('TopologyManager', () => {
  let topologyManager;
  let agent1;
  let agent2;
  let agent3;

  beforeEach(() => {
    topologyManager = new TopologyManager();
    agent1 = new Agent('agent-1', 'Agent One');
    agent2 = new Agent('agent-2', 'Agent Two');
    agent3 = new Agent('agent-3', 'Agent Three');
  });

  describe('constructor', () => {
    test('should create empty topology manager', () => {
      expect(topologyManager.agents.size).toBe(0);
      expect(topologyManager.topology.size).toBe(0);
    });
  });

  describe('registerAgent', () => {
    test('should register agent', () => {
      topologyManager.registerAgent(agent1);
      
      expect(topologyManager.agents.size).toBe(1);
      expect(topologyManager.getAgent('agent-1')).toBe(agent1);
    });

    test('should initialize topology for agent', () => {
      topologyManager.registerAgent(agent1);
      
      expect(topologyManager.getNeighbors('agent-1')).toHaveLength(0);
    });
  });

  describe('unregisterAgent', () => {
    test('should unregister agent', () => {
      topologyManager.registerAgent(agent1);
      topologyManager.unregisterAgent('agent-1');
      
      expect(topologyManager.agents.size).toBe(0);
    });

    test('should remove from neighbor sets', () => {
      topologyManager.registerAgent(agent1);
      topologyManager.registerAgent(agent2);
      topologyManager.topology.get('agent-1').add('agent-2');
      topologyManager.topology.get('agent-2').add('agent-1');
      
      topologyManager.unregisterAgent('agent-1');
      
      expect(topologyManager.getNeighbors('agent-2')).toHaveLength(0);
    });
  });

  describe('rebuildTopology', () => {
    test('should build topology based on relevance rules', () => {
      topologyManager.registerAgent(agent1);
      topologyManager.registerAgent(agent2);
      
      agent1.addBelief('Weather is good', 0.9, 'Sunny');
      agent2.addBelief('Weather is good', 0.8, 'Clear sky');
      
      topologyManager.setRelevanceRules('agent-1', ['Weather is good']);
      topologyManager.setRelevanceRules('agent-2', ['Weather is good']);
      
      topologyManager.rebuildTopology();
      
      expect(topologyManager.getNeighbors('agent-1')).toContain('agent-2');
      expect(topologyManager.getNeighbors('agent-2')).toContain('agent-1');
    });

    test('should not connect agents without matching relevance', () => {
      topologyManager.registerAgent(agent1);
      topologyManager.registerAgent(agent2);
      
      agent1.addBelief('Weather', 0.9, 'Sunny');
      agent2.addBelief('Sports', 0.8, 'Football');
      
      topologyManager.setRelevanceRules('agent-1', ['Weather']);
      topologyManager.setRelevanceRules('agent-2', ['Sports']);
      
      topologyManager.rebuildTopology();
      
      expect(topologyManager.getNeighbors('agent-1')).toHaveLength(0);
      expect(topologyManager.getNeighbors('agent-2')).toHaveLength(0);
    });
  });

  describe('setRelevanceRules', () => {
    test('should set relevance rules', () => {
      topologyManager.registerAgent(agent1);
      topologyManager.setRelevanceRules('agent-1', ['Weather', 'Sports']);
      
      topologyManager.rebuildTopology();
      
      // Should not throw, rules should be set
      expect(topologyManager.relevanceRules.get('agent-1').size).toBe(2);
    });
  });

  describe('addRelevanceRule', () => {
    test('should add single relevance rule', () => {
      topologyManager.registerAgent(agent1);
      topologyManager.addRelevanceRule('agent-1', 'Weather');
      
      expect(topologyManager.relevanceRules.get('agent-1').has('Weather')).toBe(true);
    });
  });

  describe('getNeighbors', () => {
    test('should return neighbors of agent', () => {
      topologyManager.registerAgent(agent1);
      topologyManager.registerAgent(agent2);
      
      topologyManager.topology.get('agent-1').add('agent-2');
      
      const neighbors = topologyManager.getNeighbors('agent-1');
      
      expect(neighbors).toContain('agent-2');
    });
  });

  describe('getNeighborAgents', () => {
    test('should return neighbor agent objects', () => {
      topologyManager.registerAgent(agent1);
      topologyManager.registerAgent(agent2);
      
      topologyManager.topology.get('agent-1').add('agent-2');
      
      const neighbors = topologyManager.getNeighborAgents('agent-1');
      
      expect(neighbors).toHaveLength(1);
      expect(neighbors[0].id).toBe('agent-2');
    });
  });

  describe('sendTo', () => {
    test('should send message to connected agent', () => {
      topologyManager.registerAgent(agent1);
      topologyManager.registerAgent(agent2);
      topologyManager.topology.get('agent-1').add('agent-2');
      
      const message = {
        type: 'belief_update',
        payload: { proposition: 'Test', confidence: 0.5, justification: 'Reason' }
      };
      
      const result = topologyManager.sendTo('agent-1', 'agent-2', message);
      
      expect(result).toBe(true);
    });

    test('should fail for non-existent agent', () => {
      topologyManager.registerAgent(agent1);
      
      const message = { type: 'test', payload: {} };
      const result = topologyManager.sendTo('agent-1', 'non-existent', message);
      
      expect(result).toBe(false);
    });

    test('should fail for disconnected agents', () => {
      topologyManager.registerAgent(agent1);
      topologyManager.registerAgent(agent2);
      
      const message = { type: 'test', payload: {} };
      const result = topologyManager.sendTo('agent-1', 'agent-2', message);
      
      expect(result).toBe(false);
    });
  });

  describe('broadcast', () => {
    test('should broadcast to all neighbors', () => {
      topologyManager.registerAgent(agent1);
      topologyManager.registerAgent(agent2);
      topologyManager.registerAgent(agent3);
      
      topologyManager.topology.get('agent-1').add('agent-2');
      topologyManager.topology.get('agent-1').add('agent-3');
      
      const message = {
        type: 'belief_update',
        payload: { proposition: 'Test', confidence: 0.5, justification: 'Reason' }
      };
      
      const count = topologyManager.broadcast('agent-1', message);
      
      expect(count).toBe(2);
    });
  });

  describe('broadcastAll', () => {
    test('should broadcast to all agents', () => {
      topologyManager.registerAgent(agent1);
      topologyManager.registerAgent(agent2);
      
      const message = {
        type: 'belief_update',
        payload: { proposition: 'Test', confidence: 0.5, justification: 'Reason' }
      };
      
      const count = topologyManager.broadcastAll('agent-1', message);
      
      expect(count).toBe(1); // agent-2 receives it
    });
  });

  describe('getRelevantAgents', () => {
    test('should find agents with matching relevance', () => {
      topologyManager.registerAgent(agent1);
      topologyManager.registerAgent(agent2);
      
      topologyManager.setRelevanceRules('agent-1', ['Weather']);
      topologyManager.setRelevanceRules('agent-2', ['Weather']);
      
      const agents = topologyManager.getRelevantAgents('Weather');
      
      expect(agents).toHaveLength(2);
      expect(agents).toContain('agent-1');
      expect(agents).toContain('agent-2');
    });
  });

  describe('getStats', () => {
    test('should return topology statistics', () => {
      topologyManager.registerAgent(agent1);
      topologyManager.registerAgent(agent2);
      
      const stats = topologyManager.getStats();
      
      expect(stats.agentCount).toBe(2);
      expect(stats.totalConnections).toBe(0);
    });
  });

  describe('clearTopology', () => {
    test('should clear all connections', () => {
      topologyManager.registerAgent(agent1);
      topologyManager.registerAgent(agent2);
      topologyManager.topology.get('agent-1').add('agent-2');
      topologyManager.topology.get('agent-2').add('agent-1');
      
      topologyManager.clearTopology();
      
      expect(topologyManager.getNeighbors('agent-1')).toHaveLength(0);
    });
  });

  describe('toJSON', () => {
    test('should serialize topology to JSON', () => {
      topologyManager.registerAgent(agent1);
      topologyManager.registerAgent(agent2);
      topologyManager.topology.get('agent-1').add('agent-2');
      
      const json = topologyManager.toJSON();
      
      expect(json.agents).toHaveLength(2);
      expect(json.topology['agent-1']).toContain('agent-2');
    });
  });
});
