/**
 * Unit tests for DynamicTopologyManager class
 */

const DynamicTopologyManager = require('../src/TopologyManager');
const Agent = require('../src/Agent');

describe('DynamicTopologyManager', () => {
  describe('constructor', () => {
    it('should create with default options', () => {
      const manager = new DynamicTopologyManager();
      
      expect(manager.threshold).toBe(0.3);
      expect(manager.maxNeighbors).toBe(5);
      expect(manager.bidirectional).toBe(true);
    });

    it('should create with custom options', () => {
      const manager = new DynamicTopologyManager({
        matcherMethod: 'hash',
        threshold: 0.5,
        maxNeighbors: 10,
        bidirectional: false
      });
      
      expect(manager.threshold).toBe(0.5);
      expect(manager.maxNeighbors).toBe(10);
      expect(manager.bidirectional).toBe(false);
    });
  });

  describe('registerAgent and unregisterAgent', () => {
    it('should register an agent', () => {
      const manager = new DynamicTopologyManager();
      const agent = new Agent('agent1');
      
      manager.registerAgent(agent);
      
      expect(manager.agents.has('agent1')).toBe(true);
      expect(manager.graph.has('agent1')).toBe(true);
    });

    it('should unregister an agent', () => {
      const manager = new DynamicTopologyManager();
      const agent = new Agent('agent1');
      
      manager.registerAgent(agent);
      manager.unregisterAgent('agent1');
      
      expect(manager.agents.has('agent1')).toBe(false);
      expect(manager.graph.has('agent1')).toBe(false);
    });

    it('should remove agent from all neighbor sets', () => {
      const manager = new DynamicTopologyManager({ threshold: 0.1 });
      
      const agent1 = new Agent('agent1', {
        needs: ['coffee'],
        offers: ['tea']
      });
      const agent2 = new Agent('agent2', {
        needs: ['tea'],
        offers: ['coffee']
      });
      
      manager.registerAgent(agent1);
      manager.registerAgent(agent2);
      manager.rebuildTopology();
      
      // Verify they were connected
      expect(manager.graph.get('agent1').has('agent2')).toBe(true);
      
      // Unregister agent2
      manager.unregisterAgent('agent2');
      
      // Verify agent1 no longer has agent2 as neighbor
      expect(manager.graph.get('agent1').has('agent2')).toBe(false);
    });
  });

  describe('rebuildTopology', () => {
    it('should build topology for compatible agents', () => {
      const manager = new DynamicTopologyManager({ threshold: 0.1 });
      
      const agent1 = new Agent('agent1', {
        needs: ['coffee'],
        offers: ['tea']
      });
      const agent2 = new Agent('agent2', {
        needs: ['tea'],
        offers: ['coffee']
      });
      
      manager.registerAgent(agent1);
      manager.registerAgent(agent2);
      
      const stats = manager.rebuildTopology();
      
      expect(stats.agentCount).toBe(2);
      expect(stats.edgeCount).toBeGreaterThan(0);
    });

    it('should not connect incompatible agents', () => {
      const manager = new DynamicTopologyManager({ threshold: 0.9 });
      
      const agent1 = new Agent('agent1', {
        needs: ['coffee'],
        offers: ['tea']
      });
      const agent2 = new Agent('agent2', {
        needs: ['automobile'],
        offers: ['spaceship']
      });
      
      manager.registerAgent(agent1);
      manager.registerAgent(agent2);
      
      const stats = manager.rebuildTopology();
      
      expect(stats.edgeCount).toBe(0);
    });

    it('should respect maxNeighbors limit', () => {
      const manager = new DynamicTopologyManager({
        threshold: 0.1,
        maxNeighbors: 2
      });
      
      // Create one agent that could connect to many
      const mainAgent = new Agent('main', {
        needs: ['item'],
        offers: ['item']
      });
      manager.registerAgent(mainAgent);
      
      // Add many compatible agents
      for (let i = 0; i < 5; i++) {
        manager.registerAgent(new Agent(`agent${i}`, {
          needs: ['item'],
          offers: ['item']
        }));
      }
      
      manager.rebuildTopology();
      
      const neighbors = manager.getNeighbors('main');
      expect(neighbors.length).toBeLessThanOrEqual(2);
    });

    it('should record topology history', () => {
      const manager = new DynamicTopologyManager();
      
      manager.registerAgent(new Agent('agent1'));
      manager.registerAgent(new Agent('agent2'));
      
      manager.rebuildTopology();
      
      const history = manager.getTopologyHistory();
      expect(history.length).toBe(1);
      expect(history[0].agentCount).toBe(2);
    });

    it('should return density calculation', () => {
      const manager = new DynamicTopologyManager({ threshold: 0.1 });
      
      manager.registerAgent(new Agent('agent1', { needs: ['x'], offers: ['y'] }));
      manager.registerAgent(new Agent('agent2', { needs: ['y'], offers: ['x'] }));
      
      const stats = manager.rebuildTopology();
      
      expect(stats.density).toBeGreaterThanOrEqual(0);
      expect(stats.density).toBeLessThanOrEqual(1);
    });
  });

  describe('updateAgentTopology', () => {
    it('should update topology for specific agent', () => {
      const manager = new DynamicTopologyManager({ threshold: 0.1 });
      
      manager.registerAgent(new Agent('agent1', { needs: ['x'], offers: ['y'] }));
      manager.registerAgent(new Agent('agent2', { needs: ['y'], offers: ['x'] }));
      manager.rebuildTopology();
      
      // Update agent1's needs
      const agent1 = manager.agents.get('agent1');
      agent1.setNeeds(['newNeed']);
      
      const result = manager.updateAgentTopology('agent1');
      
      expect(result.agentId).toBe('agent1');
      expect(typeof result.neighborCount).toBe('number');
    });

    it('should throw error for non-existent agent', () => {
      const manager = new DynamicTopologyManager();
      
      expect(() => manager.updateAgentTopology('nonexistent')).toThrow('not found');
    });
  });

  describe('getNeighbors and getNeighborAgents', () => {
    it('should return neighbor IDs', () => {
      const manager = new DynamicTopologyManager({ threshold: 0.1 });
      
      manager.registerAgent(new Agent('agent1', { needs: ['x'], offers: ['y'] }));
      manager.registerAgent(new Agent('agent2', { needs: ['y'], offers: ['x'] }));
      manager.rebuildTopology();
      
      const neighbors = manager.getNeighbors('agent1');
      
      expect(Array.isArray(neighbors)).toBe(true);
    });

    it('should return empty array for unknown agent', () => {
      const manager = new DynamicTopologyManager();
      
      const neighbors = manager.getNeighbors('unknown');
      
      expect(neighbors).toEqual([]);
    });

    it('should return neighbor agents', () => {
      const manager = new DynamicTopologyManager({ threshold: 0.1 });
      
      const agent1 = new Agent('agent1', { needs: ['x'], offers: ['y'] });
      const agent2 = new Agent('agent2', { needs: ['y'], offers: ['x'] });
      
      manager.registerAgent(agent1);
      manager.registerAgent(agent2);
      manager.rebuildTopology();
      
      const neighbors = manager.getNeighborAgents('agent1');
      
      expect(Array.isArray(neighbors)).toBe(true);
    });
  });

  describe('pathfinding', () => {
    it('should find direct path between connected agents', () => {
      const manager = new DynamicTopologyManager({ threshold: 0.1 });
      
      const agent1 = new Agent('agent1', { needs: ['x'], offers: ['y'] });
      const agent2 = new Agent('agent2', { needs: ['y'], offers: ['x'] });
      
      manager.registerAgent(agent1);
      manager.registerAgent(agent2);
      manager.rebuildTopology();
      
      const path = manager.getPath('agent1', 'agent2');
      
      if (path) {
        expect(path[0]).toBe('agent1');
        expect(path[path.length - 1]).toBe('agent2');
      }
    });

    it('should return null for no path', () => {
      const manager = new DynamicTopologyManager({ threshold: 0.9 });
      
      manager.registerAgent(new Agent('agent1', { needs: ['x'], offers: ['y'] }));
      manager.registerAgent(new Agent('agent2', { needs: ['z'], offers: ['w'] }));
      manager.rebuildTopology();
      
      const path = manager.getPath('agent1', 'agent2');
      
      expect(path).toBeNull();
    });

    it('should return null for unknown agents', () => {
      const manager = new DynamicTopologyManager();
      
      const path = manager.getPath('unknown1', 'unknown2');
      
      expect(path).toBeNull();
    });

    it('should find weighted path', () => {
      const manager = new DynamicTopologyManager({ threshold: 0.1 });
      
      manager.registerAgent(new Agent('agent1', { needs: ['x'], offers: ['y'] }));
      manager.registerAgent(new Agent('agent2', { needs: ['y'], offers: ['x'] }));
      manager.rebuildTopology();
      
      const path = manager.getWeightedPath('agent1', 'agent2');
      
      if (path) {
        expect(path.path).toBeDefined();
        expect(path.totalWeight).toBeDefined();
      }
    });
  });

  describe('areConnected', () => {
    it('should return true for connected agents', () => {
      const manager = new DynamicTopologyManager({ threshold: 0.1 });
      
      manager.registerAgent(new Agent('agent1', { needs: ['x'], offers: ['y'] }));
      manager.registerAgent(new Agent('agent2', { needs: ['y'], offers: ['x'] }));
      manager.rebuildTopology();
      
      const connected = manager.areConnected('agent1', 'agent2');
      
      expect(connected).toBe(true);
    });

    it('should return false for non-connected agents', () => {
      const manager = new DynamicTopologyManager({ threshold: 0.9 });
      
      manager.registerAgent(new Agent('agent1', { needs: ['x'], offers: ['y'] }));
      manager.registerAgent(new Agent('agent2', { needs: ['z'], offers: ['w'] }));
      manager.rebuildTopology();
      
      const connected = manager.areConnected('agent1', 'agent2');
      
      expect(connected).toBe(false);
    });
  });

  describe('getEdgeWeight', () => {
    it('should return edge weight for connected agents', () => {
      const manager = new DynamicTopologyManager({ threshold: 0.1 });
      
      manager.registerAgent(new Agent('agent1', { needs: ['x'], offers: ['y'] }));
      manager.registerAgent(new Agent('agent2', { needs: ['y'], offers: ['x'] }));
      manager.rebuildTopology();
      
      const weight = manager.getEdgeWeight('agent1', 'agent2');
      
      if (weight !== null) {
        expect(weight).toBeGreaterThanOrEqual(0);
        expect(weight).toBeLessThanOrEqual(1);
      }
    });

    it('should return null for non-connected agents', () => {
      const manager = new DynamicTopologyManager();
      
      const weight = manager.getEdgeWeight('agent1', 'agent2');
      
      expect(weight).toBeNull();
    });
  });

  describe('getAgentsWithinHops', () => {
    it('should find agents within N hops', () => {
      const manager = new DynamicTopologyManager({ threshold: 0.1, maxNeighbors: 5 });
      
      manager.registerAgent(new Agent('agent1', { needs: ['a'], offers: ['b'] }));
      manager.registerAgent(new Agent('agent2', { needs: ['b'], offers: ['c'] }));
      manager.registerAgent(new Agent('agent3', { needs: ['c'], offers: ['a'] }));
      manager.rebuildTopology();
      
      const withinHops = manager.getAgentsWithinHops('agent1', 2);
      
      expect(typeof withinHops).toBe('object');
    });

    it('should return empty object for unknown agent', () => {
      const manager = new DynamicTopologyManager();
      
      const withinHops = manager.getAgentsWithinHops('unknown', 2);
      
      expect(withinHops).toEqual({});
    });
  });

  describe('getStatistics', () => {
    it('should return topology statistics', () => {
      const manager = new DynamicTopologyManager();
      
      manager.registerAgent(new Agent('agent1'));
      manager.registerAgent(new Agent('agent2'));
      
      const stats = manager.getStatistics();
      
      expect(stats.agentCount).toBe(2);
      expect(stats.maxNeighbors).toBeDefined();
      expect(stats.threshold).toBeDefined();
      expect(stats.bidirectional).toBeDefined();
    });
  });

  describe('getGraph', () => {
    it('should return adjacency list', () => {
      const manager = new DynamicTopologyManager();
      
      manager.registerAgent(new Agent('agent1'));
      manager.registerAgent(new Agent('agent2'));
      manager.rebuildTopology();
      
      const graph = manager.getGraph();
      
      expect(typeof graph).toBe('object');
      expect(graph['agent1']).toBeDefined();
      expect(graph['agent2']).toBeDefined();
    });
  });

  describe('clearHistory and reset', () => {
    it('should clear topology history', () => {
      const manager = new DynamicTopologyManager();
      
      manager.registerAgent(new Agent('agent1'));
      manager.rebuildTopology();
      
      manager.clearHistory();
      
      expect(manager.topologyHistory.length).toBe(0);
    });

    it('should reset entire topology', () => {
      const manager = new DynamicTopologyManager();
      
      manager.registerAgent(new Agent('agent1'));
      manager.rebuildTopology();
      
      manager.reset();
      
      expect(manager.agents.size).toBe(0);
      expect(manager.graph.size).toBe(0);
      expect(manager.topologyHistory.length).toBe(0);
    });
  });
});
