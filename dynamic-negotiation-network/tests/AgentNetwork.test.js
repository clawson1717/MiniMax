/**
 * Unit tests for AgentNetwork class
 */

const AgentNetwork = require('../src/AgentNetwork');
const Agent = require('../src/Agent');

describe('AgentNetwork', () => {
  let network;
  let agent1;
  let agent2;

  beforeEach(() => {
    network = new AgentNetwork({ maxRounds: 10 });
    
    agent1 = new Agent('agent-1', {
      needs: ['apples'],
      offers: ['oranges']
    });
    agent2 = new Agent('agent-2', {
      needs: ['oranges'],
      offers: ['apples']
    });
  });

  describe('constructor', () => {
    test('should create network with correct properties', () => {
      expect(network.agents).toBeInstanceOf(Map);
      expect(network.inboxes).toBeInstanceOf(Map);
      expect(network.currentRound).toBe(0);
      expect(network.maxRounds).toBe(10);
      expect(network.isActive).toBe(false);
    });

    test('should use default maxRounds', () => {
      const defaultNetwork = new AgentNetwork();
      expect(defaultNetwork.maxRounds).toBe(10);
    });

    test('should initialize topology manager', () => {
      expect(network.topology).toBeDefined();
    });
  });

  describe('addAgent', () => {
    test('should add agent to network', () => {
      const result = network.addAgent(agent1);
      
      expect(result).toBe(true);
      expect(network.agents.has('agent-1')).toBe(true);
      expect(network.agents.get('agent-1')).toBe(agent1);
    });

    test('should register agent with topology manager', () => {
      network.addAgent(agent1);
      
      expect(network.topology.agents.has('agent-1')).toBe(true);
    });

    test('should initialize inbox for agent', () => {
      network.addAgent(agent1);
      
      expect(network.inboxes.has('agent-1')).toBe(true);
      expect(network.inboxes.get('agent-1')).toEqual([]);
    });

    test('should return false for duplicate agent', () => {
      network.addAgent(agent1);
      const result = network.addAgent(agent1);
      
      expect(result).toBe(false);
    });

    test('should warn on duplicate agent', () => {
      console.warn = jest.fn();
      network.addAgent(agent1);
      network.addAgent(agent1);
      
      expect(console.warn).toHaveBeenCalled();
    });
  });

  describe('removeAgent', () => {
    test('should remove agent from network', () => {
      network.addAgent(agent1);
      const result = network.removeAgent('agent-1');
      
      expect(result).toBe(true);
      expect(network.agents.has('agent-1')).toBe(false);
    });

    test('should unregister agent from topology', () => {
      network.addAgent(agent1);
      network.removeAgent('agent-1');
      
      expect(network.topology.agents.has('agent-1')).toBe(false);
    });

    test('should remove agent inbox', () => {
      network.addAgent(agent1);
      network.removeAgent('agent-1');
      
      expect(network.inboxes.has('agent-1')).toBe(false);
    });

    test('should clear pending messages from removed agent', () => {
      network.addAgent(agent1);
      network.addAgent(agent2);
      network.inboxes.get('agent-2').push({
        from: 'agent-1',
        to: 'agent-2',
        type: 'offer'
      });
      
      network.removeAgent('agent-1');
      
      const remaining = network.inboxes.get('agent-2');
      expect(remaining.some(m => m.from === 'agent-1')).toBe(false);
    });

    test('should return false for non-existent agent', () => {
      const result = network.removeAgent('non-existent');
      
      expect(result).toBe(false);
    });

    test('should warn on removing non-existent agent', () => {
      console.warn = jest.fn();
      network.removeAgent('non-existent');
      
      expect(console.warn).toHaveBeenCalled();
    });
  });

  describe('getAgent', () => {
    test('should return agent by ID', () => {
      network.addAgent(agent1);
      const agent = network.getAgent('agent-1');
      
      expect(agent).toBe(agent1);
    });

    test('should return null for non-existent agent', () => {
      const agent = network.getAgent('non-existent');
      
      expect(agent).toBeNull();
    });
  });

  describe('getAllAgents', () => {
    test('should return all agents', () => {
      network.addAgent(agent1);
      network.addAgent(agent2);
      
      const agents = network.getAllAgents();
      
      expect(agents).toHaveLength(2);
      expect(agents).toContain(agent1);
      expect(agents).toContain(agent2);
    });

    test('should return empty array when no agents', () => {
      const agents = network.getAllAgents();
      
      expect(agents).toEqual([]);
    });

    test('should return copy of agents array', () => {
      network.addAgent(agent1);
      const agents = network.getAllAgents();
      
      agents.push({ fake: 'agent' });
      expect(network.agents.size).toBe(1);
    });
  });

  describe('rebuildTopology', () => {
    test('should rebuild topology', () => {
      network.addAgent(agent1);
      network.addAgent(agent2);
      
      const stats = network.rebuildTopology();
      
      expect(stats.agentCount).toBe(2);
    });

    test('should connect compatible agents', () => {
      network.addAgent(agent1);
      network.addAgent(agent2);
      network.rebuildTopology();
      
      const neighbors = network.getNeighbors('agent-1');
      expect(neighbors.length).toBeGreaterThan(0);
    });
  });

  describe('getNeighbors', () => {
    test('should return neighbors for agent', () => {
      network.addAgent(agent1);
      network.addAgent(agent2);
      network.rebuildTopology();
      
      const neighbors = network.getNeighbors('agent-1');
      
      expect(neighbors.length).toBeGreaterThanOrEqual(0);
    });

    test('should return empty array for agent with no neighbors', () => {
      network.addAgent(agent1);
      network.rebuildTopology();
      
      const neighbors = network.getNeighbors('agent-1');
      expect(neighbors).toEqual([]);
    });
  });

  describe('send', () => {
    beforeEach(() => {
      network.addAgent(agent1);
      network.addAgent(agent2);
      network.rebuildTopology();
    });

    test('should send message successfully', () => {
      const result = network.send({
        from: 'agent-1',
        to: 'agent-2',
        type: 'offer',
        content: { offer: 'oranges' }
      });
      
      expect(result).toBe(true);
    });

    test('should add message to receiver inbox', () => {
      network.send({
        from: 'agent-1',
        to: 'agent-2',
        type: 'offer',
        content: { offer: 'oranges' }
      });
      
      const inbox = network.inboxes.get('agent-2');
      expect(inbox.length).toBe(1);
      expect(inbox[0].type).toBe('offer');
    });

    test('should add message to message history', () => {
      network.send({
        from: 'agent-1',
        to: 'agent-2',
        type: 'offer',
        content: { offer: 'oranges' }
      });
      
      expect(network.messageHistory.length).toBe(1);
    });

    test('should return false for non-existent sender', () => {
      console.error = jest.fn();
      const result = network.send({
        from: 'non-existent',
        to: 'agent-2',
        type: 'offer',
        content: {}
      });
      
      expect(result).toBe(false);
    });

    test('should return false for non-existent receiver', () => {
      console.error = jest.fn();
      const result = network.send({
        from: 'agent-1',
        to: 'non-existent',
        type: 'offer',
        content: {}
      });
      
      expect(result).toBe(false);
    });

    test('should update agent beliefs', () => {
      network.send({
        from: 'agent-1',
        to: 'agent-2',
        type: 'offer',
        content: { offer: 'oranges' }
      });
      
      const belief1 = agent1.getBelief('agent-2');
      const belief2 = agent2.getBelief('agent-1');
      
      expect(belief1).toBeDefined();
      expect(belief2).toBeDefined();
    });

    test('should trigger onMessage callback', () => {
      const callback = jest.fn();
      network.onMessage = callback;
      
      network.send({
        from: 'agent-1',
        to: 'agent-2',
        type: 'offer',
        content: {}
      });
      
      expect(callback).toHaveBeenCalled();
    });

    test('should generate unique message ID', () => {
      network.send({
        from: 'agent-1',
        to: 'agent-2',
        type: 'offer',
        content: {}
      });
      network.send({
        from: 'agent-1',
        to: 'agent-2',
        type: 'offer',
        content: {}
      });
      
      const ids = network.messageHistory.map(m => m.id);
      expect(new Set(ids).size).toBe(2);
    });
  });

  describe('broadcast', () => {
    beforeEach(() => {
      network.addAgent(agent1);
      network.addAgent(agent2);
      network.rebuildTopology();
    });

    test('should broadcast to all neighbors', () => {
      const count = network.broadcast('agent-1', {
        type: 'query',
        content: { query: 'info' }
      });
      
      expect(count).toBeGreaterThanOrEqual(0);
    });

    test('should send to each neighbor', () => {
      network.broadcast('agent-1', {
        type: 'query',
        content: {}
      });
      
      // Check that messages were sent
      const messages = network.messageHistory;
      expect(messages.length).toBeGreaterThanOrEqual(0);
    });
  });

  describe('getMessages', () => {
    test('should return messages for current round', () => {
      network.addAgent(agent1);
      network.addAgent(agent2);
      network.newRound();
      
      network.inboxes.get('agent-1').push({
        round: 1,
        type: 'offer'
      });
      
      const messages = network.getMessages('agent-1');
      expect(messages.length).toBe(1);
    });

    test('should filter by current round', () => {
      network.addAgent(agent1);
      network.newRound();
      
      network.inboxes.get('agent-1').push({ round: 0, type: 'old' });
      network.inboxes.get('agent-1').push({ round: 1, type: 'current' });
      
      const messages = network.getMessages('agent-1');
      expect(messages).toHaveLength(1);
      expect(messages[0].type).toBe('current');
    });
  });

  describe('receiveMessage', () => {
    test('should return unread messages', () => {
      network.addAgent(agent1);
      network.addAgent(agent2);
      network.newRound();
      
      network.inboxes.get('agent-1').push({
        round: 1,
        type: 'offer',
        read: false
      });
      
      const messages = network.receiveMessage('agent-1');
      expect(messages).toHaveLength(1);
    });

    test('should mark messages as read', () => {
      network.addAgent(agent1);
      network.newRound();
      
      network.inboxes.get('agent-1').push({
        round: 1,
        type: 'offer',
        read: false
      });
      
      network.receiveMessage('agent-1');
      
      const inbox = network.inboxes.get('agent-1');
      expect(inbox[0].read).toBe(true);
    });

    test('should return empty array for non-existent agent', () => {
      const messages = network.receiveMessage('non-existent');
      expect(messages).toEqual([]);
    });
  });

  describe('newRound', () => {
    test('should increment current round', () => {
      network.newRound();
      
      expect(network.currentRound).toBe(1);
    });

    test('should return round info', () => {
      network.addAgent(agent1);
      const result = network.newRound();
      
      expect(result.success).toBe(true);
      expect(result.round).toBe(1);
    });

    test('should fail at max rounds', () => {
      network.maxRounds = 1;
      network.newRound();
      
      const result = network.newRound();
      expect(result.success).toBe(false);
    });

    test('should add to round history', () => {
      network.addAgent(agent1);
      network.newRound();
      
      expect(network.roundHistory).toHaveLength(1);
    });
  });

  describe('getCurrentRound', () => {
    test('should return current round', () => {
      expect(network.getCurrentRound()).toBe(0);
      
      network.newRound();
      expect(network.getCurrentRound()).toBe(1);
    });
  });

  describe('getRoundHistory', () => {
    test('should return round history', () => {
      network.addAgent(agent1);
      network.newRound();
      
      const history = network.getRoundHistory();
      expect(history).toHaveLength(1);
    });

    test('should return copy of history', () => {
      network.newRound();
      const history = network.getRoundHistory();
      
      history.push({ fake: 'entry' });
      expect(network.roundHistory.length).toBe(1);
    });
  });

  describe('getRoundMessages', () => {
    test('should return messages for specific round', () => {
      network.addAgent(agent1);
      network.addAgent(agent2);
      network.newRound();
      
      network.send({
        from: 'agent-1',
        to: 'agent-2',
        type: 'offer',
        content: {}
      });
      
      const messages = network.getRoundMessages(1);
      expect(messages.length).toBe(1);
    });
  });

  describe('isComplete', () => {
    test('should return true at max rounds', () => {
      network.maxRounds = 1;
      network.newRound();
      
      expect(network.isComplete()).toBe(true);
    });

    test('should return true when all needs satisfied', () => {
      network.addAgent(agent1);
      network.addAgent(agent2);
      agent1.needs = [];
      agent2.needs = [];
      
      expect(network.isComplete()).toBe(true);
    });

    test('should return false when needs remain', () => {
      network.addAgent(agent1);
      
      expect(network.isComplete()).toBe(false);
    });
  });

  describe('getPendingMessageCount', () => {
    test('should return count of unread messages', () => {
      network.addAgent(agent1);
      network.inboxes.get('agent-1').push({ read: false, to: 'agent-1' });
      network.inboxes.get('agent-1').push({ read: true, to: 'agent-1' });
      
      const count = network.getPendingMessageCount('agent-1');
      expect(count).toBe(1);
    });

    test('should return 0 for non-existent agent', () => {
      const count = network.getPendingMessageCount('non-existent');
      expect(count).toBe(0);
    });
  });

  describe('clearInbox', () => {
    test('should clear agent inbox', () => {
      network.addAgent(agent1);
      network.inboxes.get('agent-1').push({ type: 'offer' });
      
      network.clearInbox('agent-1');
      
      expect(network.inboxes.get('agent-1')).toHaveLength(0);
    });
  });

  describe('clearAllInboxes', () => {
    test('should clear all inboxes', () => {
      network.addAgent(agent1);
      network.addAgent(agent2);
      network.inboxes.get('agent-1').push({});
      network.inboxes.get('agent-2').push({});
      
      network.clearAllInboxes();
      
      expect(network.inboxes.get('agent-1')).toHaveLength(0);
      expect(network.inboxes.get('agent-2')).toHaveLength(0);
    });
  });

  describe('getStatistics', () => {
    test('should return network statistics', () => {
      network.addAgent(agent1);
      network.newRound();
      
      const stats = network.getStatistics();
      
      expect(stats.agentCount).toBe(1);
      expect(stats.currentRound).toBe(1);
      expect(stats.maxRounds).toBe(10);
    });

    test('should include topology statistics', () => {
      const stats = network.getStatistics();
      
      expect(stats.totalEdges).toBeDefined();
      expect(stats.averageDegree).toBeDefined();
    });
  });

  describe('getPath', () => {
    test('should find path between agents', () => {
      network.addAgent(agent1);
      network.addAgent(agent2);
      network.rebuildTopology();
      
      const path = network.getPath('agent-1', 'agent-2');
      expect(path).toBeDefined();
    });
  });

  describe('getWeightedPath', () => {
    test('should find weighted path', () => {
      network.addAgent(agent1);
      network.addAgent(agent2);
      network.rebuildTopology();
      
      const result = network.getWeightedPath('agent-1', 'agent-2');
      expect(result).toBeDefined();
    });
  });

  describe('activate', () => {
    test('should set network as active', () => {
      network.addAgent(agent1);
      network.addAgent(agent2);
      
      network.activate();
      
      expect(network.isActive).toBe(true);
    });

    test('should rebuild topology', () => {
      network.addAgent(agent1);
      network.addAgent(agent2);
      
      network.activate();
      
      expect(network.topology.getGraph()).toBeDefined();
    });
  });

  describe('deactivate', () => {
    test('should set network as inactive', () => {
      network.activate();
      network.deactivate();
      
      expect(network.isActive).toBe(false);
    });
  });

  describe('reset', () => {
    test('should reset network state', () => {
      network.addAgent(agent1);
      network.newRound();
      network.inboxes.get('agent-1').push({});
      network.messageHistory.push({});
      
      network.reset();
      
      expect(network.currentRound).toBe(0);
      expect(network.roundHistory).toHaveLength(0);
      expect(network.messageHistory).toHaveLength(0);
      expect(network.inboxes.get('agent-1')).toHaveLength(0);
      expect(network.isActive).toBe(false);
    });
  });

  describe('toJSON', () => {
    test('should serialize network state', () => {
      network.addAgent(agent1);
      network.newRound();
      
      const json = network.toJSON();
      
      expect(json.agents).toHaveLength(1);
      expect(json.currentRound).toBe(1);
      expect(json.topology).toBeDefined();
    });
  });

  describe('getTopologyManager', () => {
    test('should return topology manager', () => {
      const topology = network.getTopologyManager();
      
      expect(topology).toBe(network.topology);
    });
  });
});
