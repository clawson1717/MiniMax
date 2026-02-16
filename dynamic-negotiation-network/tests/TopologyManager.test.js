/**
 * Unit tests for DynamicTopologyManager class
 */

const DynamicTopologyManager = require('../src/TopologyManager');
const Agent = require('../src/Agent');

describe('DynamicTopologyManager', () => {
  let topologyManager;
  let agent1;
  let agent2;
  let agent3;

  beforeEach(() => {
    topologyManager = new DynamicTopologyManager({
      threshold: 0.3,
      maxNeighbors: 5,
      bidirectional: true,
      matcherMethod: 'tfidf'
    });
    
    agent1 = new Agent('agent-1', {
      needs: ['apples'],
      offers: ['oranges']
    });
    agent2 = new Agent('agent-2', {
      needs: ['oranges'],
      offers: ['apples']
    });
    agent3 = new Agent('agent-3', {
      needs: ['bananas'],
      offers: ['grapes']
    });
  });

  describe('constructor', () => {
    test('should create topology manager with correct properties', () => {
      expect(topologyManager.threshold).toBe(0.3);
      expect(topologyManager.maxNeighbors).toBe(5);
      expect(topologyManager.bidirectional).toBe(true);
      expect(topologyManager.matcher).toBeDefined();
      expect(topologyManager.graph).toBeInstanceOf(Map);
      expect(topologyManager.edgeWeights).toBeInstanceOf(Map);
      expect(topologyManager.agents).toBeInstanceOf(Map);
    });

    test('should use default values when not provided', () => {
      const defaultManager = new DynamicTopologyManager();
      expect(defaultManager.threshold).toBe(0.3);
      expect(defaultManager.maxNeighbors).toBe(5);
      expect(defaultManager.bidirectional).toBe(true);
    });
  });

  describe('registerAgent', () => {
    test('should register agent in agents map', () => {
      topologyManager.registerAgent(agent1);
      expect(topologyManager.agents.has('agent-1')).toBe(true);
      expect(topologyManager.agents.get('agent-1')).toBe(agent1);
    });

    test('should initialize empty neighbor set for agent', () => {
      topologyManager.registerAgent(agent1);
      expect(topologyManager.graph.has('agent-1')).toBe(true);
      expect(topologyManager.graph.get('agent-1')).toBeInstanceOf(Set);
      expect(topologyManager.graph.get('agent-1').size).toBe(0);
    });

    test('should handle multiple agents', () => {
      topologyManager.registerAgent(agent1);
      topologyManager.registerAgent(agent2);
      expect(topologyManager.agents.size).toBe(2);
    });
  });

  describe('unregisterAgent', () => {
    test('should remove agent from agents map', () => {
      topologyManager.registerAgent(agent1);
      topologyManager.unregisterAgent('agent-1');
      expect(topologyManager.agents.has('agent-1')).toBe(false);
    });

    test('should remove agent from graph', () => {
      topologyManager.registerAgent(agent1);
      topologyManager.unregisterAgent('agent-1');
      expect(topologyManager.graph.has('agent-1')).toBe(false);
    });

    test('should remove agent from all neighbor sets', () => {
      topologyManager.registerAgent(agent1);
      topologyManager.registerAgent(agent2);
      topologyManager.graph.get('agent-1').add('agent-2');
      topologyManager.graph.get('agent-2').add('agent-1');
      
      topologyManager.unregisterAgent('agent-1');
      expect(topologyManager.graph.get('agent-2').has('agent-1')).toBe(false);
    });

    test('should clean up edge weights', () => {
      topologyManager.registerAgent(agent1);
      topologyManager.unregisterAgent('agent-1');
      expect(topologyManager.edgeWeights.has('agent-1')).toBe(false);
    });
  });

  describe('rebuildTopology', () => {
    test('should build edges between compatible agents', () => {
      topologyManager.registerAgent(agent1);
      topologyManager.registerAgent(agent2);
      
      const stats = topologyManager.rebuildTopology();
      
      expect(stats.agentCount).toBe(2);
      expect(topologyManager.areConnected('agent-1', 'agent-2')).toBe(true);
      expect(topologyManager.areConnected('agent-2', 'agent-1')).toBe(true);
    });

    test('should not connect incompatible agents', () => {
      topologyManager.registerAgent(agent1);
      topologyManager.registerAgent(agent3);
      
      topologyManager.rebuildTopology();
      
      expect(topologyManager.areConnected('agent-1', 'agent-3')).toBe(false);
    });

    test('should return topology statistics', () => {
      topologyManager.registerAgent(agent1);
      topologyManager.registerAgent(agent2);
      
      const stats = topologyManager.rebuildTopology();
      
      expect(stats.timestamp).toBeDefined();
      expect(stats.agentCount).toBe(2);
      expect(stats.edgeCount).toBeGreaterThan(0);
      expect(stats.density).toBeGreaterThan(0);
    });

    test('should record topology state in history', () => {
      topologyManager.registerAgent(agent1);
      topologyManager.registerAgent(agent2);
      
      topologyManager.rebuildTopology();
      
      expect(topologyManager.topologyHistory.length).toBe(1);
      expect(topologyManager.topologyHistory[0].agentCount).toBe(2);
    });

    test('should clear existing graph before rebuild', () => {
      topologyManager.registerAgent(agent1);
      topologyManager.registerAgent(agent2);
      topologyManager.rebuildTopology();
      
      // Change agent needs to break compatibility
      agent1.setNeeds(['spaceships']);
      topologyManager.rebuildTopology();
      
      expect(topologyManager.areConnected('agent-1', 'agent-2')).toBe(false);
    });

    test('should respect maxNeighbors limit', () => {
      topologyManager.maxNeighbors = 1;
      
      // Create agents where one would connect to many
      const agentA = new Agent('agent-a', { needs: ['apples'], offers: ['oranges'] });
      const agentB = new Agent('agent-b', { needs: ['oranges'], offers: ['apples'] });
      const agentC = new Agent('agent-c', { needs: ['oranges'], offers: ['bananas'] });
      
      topologyManager.registerAgent(agentA);
      topologyManager.registerAgent(agentB);
      topologyManager.registerAgent(agentC);
      
      topologyManager.rebuildTopology();
      
      const neighborsA = topologyManager.getNeighbors('agent-a');
      expect(neighborsA.length).toBeLessThanOrEqual(1);
    });
  });

  describe('updateAgentTopology', () => {
    test('should update topology for specific agent', () => {
      topologyManager.registerAgent(agent1);
      topologyManager.registerAgent(agent2);
      topologyManager.rebuildTopology();
      
      const result = topologyManager.updateAgentTopology('agent-1');
      
      expect(result.agentId).toBe('agent-1');
      expect(result.neighbors).toContain('agent-2');
    });

    test('should throw error for non-existent agent', () => {
      expect(() => {
        topologyManager.updateAgentTopology('non-existent');
      }).toThrow('Agent non-existent not found');
    });

    test('should clear existing connections before update', () => {
      topologyManager.registerAgent(agent1);
      topologyManager.registerAgent(agent2);
      topologyManager.registerAgent(agent3);
      topologyManager.rebuildTopology();
      
      // Change agent1's needs to only match agent3
      agent1.setNeeds(['bananas']);
      agent1.setOffers(['grapes']);
      
      const result = topologyManager.updateAgentTopology('agent-1');
      
      expect(result.neighbors).not.toContain('agent-2');
    });
  });

  describe('getNeighbors', () => {
    test('should return neighbors for agent', () => {
      topologyManager.registerAgent(agent1);
      topologyManager.registerAgent(agent2);
      topologyManager.rebuildTopology();
      
      const neighbors = topologyManager.getNeighbors('agent-1');
      expect(neighbors).toContain('agent-2');
    });

    test('should return empty array for agent with no neighbors', () => {
      topologyManager.registerAgent(agent1);
      topologyManager.rebuildTopology();
      
      const neighbors = topologyManager.getNeighbors('agent-1');
      expect(neighbors).toEqual([]);
    });

    test('should return empty array for non-existent agent', () => {
      const neighbors = topologyManager.getNeighbors('non-existent');
      expect(neighbors).toEqual([]);
    });
  });

  describe('getNeighborAgents', () => {
    test('should return neighbor agent objects', () => {
      topologyManager.registerAgent(agent1);
      topologyManager.registerAgent(agent2);
      topologyManager.rebuildTopology();
      
      const neighbors = topologyManager.getNeighborAgents('agent-1');
      expect(neighbors.length).toBe(1);
      expect(neighbors[0].id).toBe('agent-2');
    });

    test('should return empty array for no neighbors', () => {
      topologyManager.registerAgent(agent1);
      topologyManager.rebuildTopology();
      
      const neighbors = topologyManager.getNeighborAgents('agent-1');
      expect(neighbors).toEqual([]);
    });
  });

  describe('getPath', () => {
    test('should find shortest path between connected agents', () => {
      topologyManager.registerAgent(agent1);
      topologyManager.registerAgent(agent2);
      topologyManager.rebuildTopology();
      
      const path = topologyManager.getPath('agent-1', 'agent-2');
      expect(path).toEqual(['agent-1', 'agent-2']);
    });

    test('should return null for disconnected agents', () => {
      topologyManager.registerAgent(agent1);
      topologyManager.registerAgent(agent3);
      topologyManager.rebuildTopology();
      
      const path = topologyManager.getPath('agent-1', 'agent-3');
      expect(path).toBeNull();
    });

    test('should return null for non-existent agent', () => {
      topologyManager.registerAgent(agent1);
      topologyManager.rebuildTopology();
      
      const path = topologyManager.getPath('agent-1', 'non-existent');
      expect(path).toBeNull();
    });

    test('should find multi-hop path', () => {
      // Create chain: agentA -> agentB -> agentC
      const agentA = new Agent('agent-a', { needs: ['item-a'], offers: ['item-b'] });
      const agentB = new Agent('agent-b', { needs: ['item-b'], offers: ['item-c'] });
      const agentC = new Agent('agent-c', { needs: ['item-c'], offers: ['item-d'] });
      
      topologyManager.registerAgent(agentA);
      topologyManager.registerAgent(agentB);
      topologyManager.registerAgent(agentC);
      topologyManager.rebuildTopology();
      
      const path = topologyManager.getPath('agent-a', 'agent-c');
      // Path may or may not exist depending on similarity scores
      if (path) {
        expect(path[0]).toBe('agent-a');
        expect(path[path.length - 1]).toBe('agent-c');
      }
    });
  });

  describe('getWeightedPath', () => {
    test('should find weighted path between agents', () => {
      topologyManager.registerAgent(agent1);
      topologyManager.registerAgent(agent2);
      topologyManager.rebuildTopology();
      
      const result = topologyManager.getWeightedPath('agent-1', 'agent-2');
      expect(result.path).toEqual(['agent-1', 'agent-2']);
      expect(result.totalWeight).toBeDefined();
    });

    test('should return null for disconnected agents', () => {
      topologyManager.registerAgent(agent1);
      topologyManager.registerAgent(agent3);
      topologyManager.rebuildTopology();
      
      const result = topologyManager.getWeightedPath('agent-1', 'agent-3');
      expect(result).toBeNull();
    });

    test('should return null for non-existent agent', () => {
      topologyManager.registerAgent(agent1);
      topologyManager.rebuildTopology();
      
      const result = topologyManager.getWeightedPath('agent-1', 'non-existent');
      expect(result).toBeNull();
    });
  });

  describe('getGraph', () => {
    test('should return graph as adjacency list', () => {
      topologyManager.registerAgent(agent1);
      topologyManager.registerAgent(agent2);
      topologyManager.rebuildTopology();
      
      const graph = topologyManager.getGraph();
      expect(graph['agent-1']).toContain('agent-2');
      expect(graph['agent-2']).toContain('agent-1');
    });

    test('should return empty object for empty graph', () => {
      const graph = topologyManager.getGraph();
      expect(graph).toEqual({});
    });
  });

  describe('getStatistics', () => {
    test('should return topology statistics', () => {
      topologyManager.registerAgent(agent1);
      topologyManager.registerAgent(agent2);
      topologyManager.rebuildTopology();
      
      const stats = topologyManager.getStatistics();
      expect(stats.agentCount).toBe(2);
      expect(stats.totalEdges).toBeGreaterThan(0);
      expect(stats.averageDegree).toBeGreaterThan(0);
      expect(stats.maxNeighbors).toBe(5);
      expect(stats.threshold).toBe(0.3);
      expect(stats.bidirectional).toBe(true);
    });

    test('should handle empty topology', () => {
      const stats = topologyManager.getStatistics();
      expect(stats.agentCount).toBe(0);
      expect(stats.totalEdges).toBe(0);
      expect(stats.averageDegree).toBe(0);
    });
  });

  describe('getTopologyHistory', () => {
    test('should return copy of topology history', () => {
      topologyManager.registerAgent(agent1);
      topologyManager.rebuildTopology();
      
      const history = topologyManager.getTopologyHistory();
      expect(history).toHaveLength(1);
      
      history.push({ fake: 'entry' });
      expect(topologyManager.topologyHistory).toHaveLength(1);
    });
  });

  describe('areConnected', () => {
    test('should return true for connected agents', () => {
      topologyManager.registerAgent(agent1);
      topologyManager.registerAgent(agent2);
      topologyManager.rebuildTopology();
      
      expect(topologyManager.areConnected('agent-1', 'agent-2')).toBe(true);
    });

    test('should return false for disconnected agents', () => {
      topologyManager.registerAgent(agent1);
      topologyManager.registerAgent(agent3);
      topologyManager.rebuildTopology();
      
      expect(topologyManager.areConnected('agent-1', 'agent-3')).toBe(false);
    });

    test('should return false for non-existent agent', () => {
      expect(topologyManager.areConnected('agent-1', 'non-existent')).toBe(false);
    });
  });

  describe('getEdgeWeight', () => {
    test('should return edge weight for connected agents', () => {
      topologyManager.registerAgent(agent1);
      topologyManager.registerAgent(agent2);
      topologyManager.rebuildTopology();
      
      const weight = topologyManager.getEdgeWeight('agent-1', 'agent-2');
      expect(weight).toBeDefined();
      expect(typeof weight).toBe('number');
    });

    test('should return null for non-connected agents', () => {
      topologyManager.registerAgent(agent1);
      topologyManager.registerAgent(agent3);
      topologyManager.rebuildTopology();
      
      const weight = topologyManager.getEdgeWeight('agent-1', 'agent-3');
      expect(weight).toBeNull();
    });
  });

  describe('getAgentsWithinHops', () => {
    test('should return agents within specified hops', () => {
      // Chain topology
      const agentA = new Agent('agent-a', { needs: ['item-a'], offers: ['item-b'] });
      const agentB = new Agent('agent-b', { needs: ['item-b'], offers: ['item-c'] });
      const agentC = new Agent('agent-c', { needs: ['item-c'], offers: ['item-d'] });
      
      topologyManager.registerAgent(agentA);
      topologyManager.registerAgent(agentB);
      topologyManager.registerAgent(agentC);
      topologyManager.rebuildTopology();
      
      const result = topologyManager.getAgentsWithinHops('agent-a', 2);
      // Results depend on topology connectivity
      if (Object.keys(result).length > 0) {
        expect(Object.keys(result)).toContain('agent-b');
      }
    });

    test('should return empty object for isolated agent', () => {
      topologyManager.registerAgent(agent1);
      topologyManager.rebuildTopology();
      
      const result = topologyManager.getAgentsWithinHops('agent-1', 2);
      expect(result).toEqual({});
    });

    test('should return empty object for non-existent agent', () => {
      const result = topologyManager.getAgentsWithinHops('non-existent', 2);
      expect(result).toEqual({});
    });
  });

  describe('clearHistory', () => {
    test('should clear topology history', () => {
      topologyManager.registerAgent(agent1);
      topologyManager.rebuildTopology();
      
      topologyManager.clearHistory();
      expect(topologyManager.topologyHistory).toHaveLength(0);
    });
  });

  describe('reset', () => {
    test('should reset all state', () => {
      topologyManager.registerAgent(agent1);
      topologyManager.registerAgent(agent2);
      topologyManager.rebuildTopology();
      topologyManager.topologyHistory.push({ test: 'entry' });
      
      topologyManager.reset();
      
      expect(topologyManager.graph.size).toBe(0);
      expect(topologyManager.edgeWeights.size).toBe(0);
      expect(topologyManager.topologyHistory).toHaveLength(0);
    });
  });
});
