/**
 * Unit tests for AgentNetwork class
 */

const AgentNetwork = require('../src/AgentNetwork');
const Agent = require('../src/Agent');

describe('AgentNetwork', () => {
  describe('constructor', () => {
    it('should create with default options', () => {
      const network = new AgentNetwork();
      
      expect(network.currentRound).toBe(0);
      expect(network.maxRounds).toBe(10);
      expect(network.isActive).toBe(false);
    });

    it('should create with custom options', () => {
      const network = new AgentNetwork({
        maxRounds: 20
      });
      
      expect(network.maxRounds).toBe(20);
    });

    it('should initialize topology manager', () => {
      const network = new AgentNetwork();
      
      expect(network.topology).toBeDefined();
    });
  });

  describe('addAgent and removeAgent', () => {
    it('should add an agent', () => {
      const network = new AgentNetwork();
      const agent = new Agent('agent1');
      
      const result = network.addAgent(agent);
      
      expect(result).toBe(true);
      expect(network.agents.has('agent1')).toBe(true);
    });

    it('should not add duplicate agent', () => {
      const network = new AgentNetwork();
      const agent = new Agent('agent1');
      
      network.addAgent(agent);
      const result = network.addAgent(agent);
      
      expect(result).toBe(false);
    });

    it('should remove an agent', () => {
      const network = new AgentNetwork();
      const agent = new Agent('agent1');
      
      network.addAgent(agent);
      const result = network.removeAgent('agent1');
      
      expect(result).toBe(true);
      expect(network.agents.has('agent1')).toBe(false);
    });

    it('should return false for non-existent agent removal', () => {
      const network = new AgentNetwork();
      
      const result = network.removeAgent('nonexistent');
      
      expect(result).toBe(false);
    });

    it('should clear pending messages on removal', () => {
      const network = new AgentNetwork();
      
      network.addAgent(new Agent('agent1'));
      network.addAgent(new Agent('agent2'));
      
      network.send({
        from: 'agent1',
        to: 'agent2',
        type: 'test',
        content: 'hello'
      });
      
      network.removeAgent('agent2');
      
      expect(network.inboxes.has('agent2')).toBe(false);
    });
  });

  describe('getAgent and getAllAgents', () => {
    it('should get agent by ID', () => {
      const network = new AgentNetwork();
      const agent = new Agent('agent1');
      
      network.addAgent(agent);
      
      expect(network.getAgent('agent1')).toBe(agent);
    });

    it('should return null for non-existent agent', () => {
      const network = new AgentNetwork();
      
      expect(network.getAgent('nonexistent')).toBeNull();
    });

    it('should get all agents', () => {
      const network = new AgentNetwork();
      
      network.addAgent(new Agent('agent1'));
      network.addAgent(new Agent('agent2'));
      
      const agents = network.getAllAgents();
      
      expect(agents.length).toBe(2);
    });
  });

  describe('messaging', () => {
    let network;
    
    beforeEach(() => {
      network = new AgentNetwork({ maxRounds: 5 });
      network.addAgent(new Agent('agent1'));
      network.addAgent(new Agent('agent2'));
      network.rebuildTopology();
      network.newRound();
    });

    it('should send a message', () => {
      const result = network.send({
        from: 'agent1',
        to: 'agent2',
        type: 'test',
        content: 'hello'
      });
      
      expect(result).toBe(true);
    });

    it('should fail for non-existent sender', () => {
      const result = network.send({
        from: 'nonexistent',
        to: 'agent2',
        type: 'test',
        content: 'hello'
      });
      
      expect(result).toBe(false);
    });

    it('should fail for non-existent receiver', () => {
      const result = network.send({
        from: 'agent1',
        to: 'nonexistent',
        type: 'test',
        content: 'hello'
      });
      
      expect(result).toBe(false);
    });

    it('should store message in inbox', () => {
      network.send({
        from: 'agent1',
        to: 'agent2',
        type: 'test',
        content: 'hello'
      });
      
      const inbox = network.inboxes.get('agent2');
      expect(inbox.length).toBe(1);
    });

    it('should record message in history', () => {
      network.send({
        from: 'agent1',
        to: 'agent2',
        type: 'test',
        content: 'hello'
      });
      
      expect(network.messageHistory.length).toBe(1);
    });

    it('should update agent beliefs', () => {
      network.send({
        from: 'agent1',
        to: 'agent2',
        type: 'offer',
        content: { offer: 'test' }
      });
      
      const agent1 = network.getAgent('agent1');
      const belief = agent1.getBelief('agent2');
      
      expect(belief).toBeDefined();
      expect(belief.lastContact).toBeDefined();
    });
  });

  describe('broadcast', () => {
    it('should broadcast to all neighbors', () => {
      const network = new AgentNetwork({ maxRounds: 5 });
      
      network.addAgent(new Agent('agent1', { needs: ['x'], offers: ['y'] }));
      network.addAgent(new Agent('agent2', { needs: ['y'], offers: ['x'] }));
      network.addAgent(new Agent('agent3', { needs: ['y'], offers: ['x'] }));
      network.rebuildTopology();
      network.newRound();
      
      const sentCount = network.broadcast('agent1', {
        type: 'test',
        content: 'broadcast'
      });
      
      expect(sentCount).toBeGreaterThan(0);
    });
  });

  describe('receiveMessage', () => {
    it('should return and mark messages as read', () => {
      const network = new AgentNetwork({ maxRounds: 5 });
      
      network.addAgent(new Agent('agent1'));
      network.addAgent(new Agent('agent2'));
      network.rebuildTopology();
      network.newRound();
      
      network.send({
        from: 'agent1',
        to: 'agent2',
        type: 'test',
        content: 'hello'
      });
      
      const messages = network.receiveMessage('agent2');
      
      expect(messages.length).toBe(1);
      expect(messages[0].read).toBe(true);
    });

    it('should return empty array for no messages', () => {
      const network = new AgentNetwork();
      network.addAgent(new Agent('agent1'));
      
      const messages = network.receiveMessage('agent1');
      
      expect(messages).toEqual([]);
    });
  });

  describe('round management', () => {
    it('should start a new round', () => {
      const network = new AgentNetwork({ maxRounds: 5 });
      
      const result = network.newRound();
      
      expect(result.success).toBe(true);
      expect(result.round).toBe(1);
      expect(network.currentRound).toBe(1);
    });

    it('should not exceed max rounds', () => {
      const network = new AgentNetwork({ maxRounds: 2 });
      
      network.newRound();
      network.newRound();
      const result = network.newRound();
      
      expect(result.success).toBe(false);
    });

    it('should track round history', () => {
      const network = new AgentNetwork({ maxRounds: 5 });
      
      network.newRound();
      network.newRound();
      
      const history = network.getRoundHistory();
      expect(history.length).toBe(2);
    });

    it('should get current round', () => {
      const network = new AgentNetwork();
      
      expect(network.getCurrentRound()).toBe(0);
      
      network.newRound();
      expect(network.getCurrentRound()).toBe(1);
    });
  });

  describe('isComplete', () => {
    it('should return false when rounds remaining', () => {
      const network = new AgentNetwork({ maxRounds: 5 });
      network.addAgent(new Agent('agent1', { needs: ['x'] }));
      
      expect(network.isComplete()).toBe(false);
    });

    it('should return true when max rounds reached', () => {
      const network = new AgentNetwork({ maxRounds: 1 });
      network.newRound();
      
      expect(network.isComplete()).toBe(true);
    });

    it('should return true when all agents satisfied', () => {
      const network = new AgentNetwork({ maxRounds: 5 });
      network.addAgent(new Agent('agent1'));
      network.addAgent(new Agent('agent2'));
      
      expect(network.isComplete()).toBe(true);
    });
  });

  describe('topology integration', () => {
    it('should rebuild topology', () => {
      const network = new AgentNetwork();
      
      network.addAgent(new Agent('agent1', { needs: ['x'], offers: ['y'] }));
      network.addAgent(new Agent('agent2', { needs: ['y'], offers: ['x'] }));
      
      const stats = network.rebuildTopology();
      
      expect(stats.agentCount).toBe(2);
    });

    it('should get neighbors', () => {
      const network = new AgentNetwork({ maxRounds: 5 });
      
      network.addAgent(new Agent('agent1', { needs: ['x'], offers: ['y'] }));
      network.addAgent(new Agent('agent2', { needs: ['y'], offers: ['x'] }));
      network.rebuildTopology();
      
      const neighbors = network.getNeighbors('agent1');
      
      expect(Array.isArray(neighbors)).toBe(true);
    });

    it('should find path between agents', () => {
      const network = new AgentNetwork();
      
      network.addAgent(new Agent('agent1', { needs: ['x'], offers: ['y'] }));
      network.addAgent(new Agent('agent2', { needs: ['y'], offers: ['x'] }));
      network.rebuildTopology();
      
      const path = network.getPath('agent1', 'agent2');
      
      if (path) {
        expect(path[0]).toBe('agent1');
        expect(path[path.length - 1]).toBe('agent2');
      }
    });
  });

  describe('activate and deactivate', () => {
    it('should activate network', () => {
      const network = new AgentNetwork();
      
      network.activate();
      
      expect(network.isActive).toBe(true);
    });

    it('should deactivate network', () => {
      const network = new AgentNetwork();
      
      network.activate();
      network.deactivate();
      
      expect(network.isActive).toBe(false);
    });
  });

  describe('reset', () => {
    it('should reset network state', () => {
      const network = new AgentNetwork({ maxRounds: 5 });
      
      network.addAgent(new Agent('agent1'));
      network.newRound();
      network.newRound();
      network.send({
        from: 'agent1',
        to: 'agent1',
        type: 'test',
        content: 'test'
      });
      
      network.reset();
      
      expect(network.currentRound).toBe(0);
      expect(network.roundHistory.length).toBe(0);
      expect(network.messageHistory.length).toBe(0);
    });
  });

  describe('getStatistics', () => {
    it('should return network statistics', () => {
      const network = new AgentNetwork({ maxRounds: 5 });
      
      network.addAgent(new Agent('agent1'));
      network.addAgent(new Agent('agent2'));
      network.newRound();
      
      const stats = network.getStatistics();
      
      expect(stats.agentCount).toBe(2);
      expect(stats.currentRound).toBe(1);
      expect(stats.maxRounds).toBe(5);
    });
  });

  describe('toJSON', () => {
    it('should export network state', () => {
      const network = new AgentNetwork();
      
      network.addAgent(new Agent('agent1', { needs: ['x'] }));
      
      const json = network.toJSON();
      
      expect(json.agents).toBeDefined();
      expect(json.agents.length).toBe(1);
      expect(json.currentRound).toBe(0);
    });
  });

  describe('inbox management', () => {
    it('should get pending message count', () => {
      const network = new AgentNetwork({ maxRounds: 5 });
      
      network.addAgent(new Agent('agent1'));
      network.addAgent(new Agent('agent2'));
      network.rebuildTopology();
      network.newRound();
      
      network.send({
        from: 'agent1',
        to: 'agent2',
        type: 'test',
        content: 'hello'
      });
      
      const count = network.getPendingMessageCount('agent2');
      
      expect(count).toBe(1);
    });

    it('should clear inbox for agent', () => {
      const network = new AgentNetwork();
      
      network.addAgent(new Agent('agent1'));
      network.clearInbox('agent1');
      
      const inbox = network.inboxes.get('agent1');
      expect(inbox.length).toBe(0);
    });

    it('should clear all inboxes', () => {
      const network = new AgentNetwork();
      
      network.addAgent(new Agent('agent1'));
      network.addAgent(new Agent('agent2'));
      network.clearAllInboxes();
      
      for (const inbox of network.inboxes.values()) {
        expect(inbox.length).toBe(0);
      }
    });
  });

  describe('callbacks', () => {
    it('should call onMessage callback', () => {
      let called = false;
      const network = new AgentNetwork({
        maxRounds: 5,
        onMessage: () => { called = true; }
      });
      
      network.addAgent(new Agent('agent1'));
      network.addAgent(new Agent('agent2'));
      network.rebuildTopology();
      network.newRound();
      
      network.send({
        from: 'agent1',
        to: 'agent2',
        type: 'test',
        content: 'hello'
      });
      
      expect(called).toBe(true);
    });
  });
});
