/**
 * Unit tests for NegotiationRound class
 */

const NegotiationRound = require('../src/NegotiationRound');
const AgentNetwork = require('../src/AgentNetwork');
const Agent = require('../src/Agent');

describe('NegotiationRound', () => {
  let network;
  
  beforeEach(() => {
    network = new AgentNetwork({ maxRounds: 10 });
  });

  describe('constructor', () => {
    it('should create with network reference', () => {
      network.addAgent(new Agent('agent1'));
      network.newRound();
      
      const round = new NegotiationRound(network);
      
      expect(round.network).toBe(network);
      expect(round.roundNumber).toBe(1);
    });

    it('should use default strategy resolver', () => {
      network.addAgent(new Agent('agent1'));
      network.newRound();
      
      const round = new NegotiationRound(network);
      
      expect(typeof round.strategyResolver).toBe('function');
    });
  });

  describe('execute', () => {
    it('should execute round and return results', () => {
      network.addAgent(new Agent('agent1', {
        needs: ['coffee'],
        offers: ['tea']
      }));
      network.addAgent(new Agent('agent2', {
        needs: ['tea'],
        offers: ['coffee']
      }));
      network.rebuildTopology();
      network.newRound();
      
      const round = new NegotiationRound(network);
      const result = round.execute();
      
      expect(result.success).toBe(true);
      expect(result.round).toBeDefined();
      expect(typeof result.messages).toBe('number');
    });

    it('should not execute twice', () => {
      network.addAgent(new Agent('agent1'));
      network.newRound();
      
      const round = new NegotiationRound(network);
      round.execute();
      const result = round.execute();
      
      expect(result.success).toBe(false);
      expect(result.message).toContain('already completed');
    });

    it('should record start and end time', () => {
      network.addAgent(new Agent('agent1'));
      network.newRound();
      
      const round = new NegotiationRound(network);
      round.execute();
      
      expect(round.startTime).toBeDefined();
      expect(round.endTime).toBeDefined();
      expect(round.endTime).toBeGreaterThanOrEqual(round.startTime);
    });
  });

  describe('message handling', () => {
    beforeEach(() => {
      network.addAgent(new Agent('agent1', {
        needs: ['coffee'],
        offers: ['tea']
      }));
      network.addAgent(new Agent('agent2', {
        needs: ['tea'],
        offers: ['coffee']
      }));
      network.rebuildTopology();
      network.newRound();
    });

    it('should handle offer messages', () => {
      const round = new NegotiationRound(network);
      
      // Send an offer
      network.send({
        from: 'agent2',
        to: 'agent1',
        type: 'offer',
        content: { offer: 'coffee' }
      });
      
      round.execute();
      
      const stats = round.getStatistics();
      expect(stats.offersMade).toBeGreaterThan(0);
    });

    it('should handle accept messages', () => {
      const round = new NegotiationRound(network);
      
      network.send({
        from: 'agent2',
        to: 'agent1',
        type: 'offer',
        content: { offer: 'coffee' }
      });
      
      round.execute();
      
      // Check if any acceptances occurred
      const stats = round.getStatistics();
      expect(typeof stats.acceptances).toBe('number');
    });

    it('should handle counter messages', () => {
      const round = new NegotiationRound(network);
      
      network.send({
        from: 'agent2',
        to: 'agent1',
        type: 'offer',
        content: { offer: 'something unrelated' }
      });
      
      round.execute();
      
      const stats = round.getStatistics();
      expect(typeof stats.counters).toBe('number');
    });

    it('should handle reject messages', () => {
      network.addAgent(new Agent('agent3', {
        needs: ['nothing'],
        offers: []
      }));
      network.rebuildTopology();
      
      const round = new NegotiationRound(network);
      round.execute();
      
      const stats = round.getStatistics();
      expect(typeof stats.rejections).toBe('number');
    });

    it('should handle query messages', () => {
      const round = new NegotiationRound(network);
      
      network.send({
        from: 'agent2',
        to: 'agent1',
        type: 'query',
        content: { question: 'what do you need?' }
      });
      
      round.execute();
      
      // Query should trigger a response
      const messages = round.getMessages();
      expect(messages.length).toBeGreaterThan(0);
    });
  });

  describe('getMessages', () => {
    it('should return all messages from round', () => {
      network.addAgent(new Agent('agent1'));
      network.addAgent(new Agent('agent2'));
      network.rebuildTopology();
      network.newRound();
      
      const round = new NegotiationRound(network);
      
      network.send({
        from: 'agent1',
        to: 'agent2',
        type: 'test',
        content: 'hello'
      });
      
      round.execute();
      
      const messages = round.getMessages();
      
      expect(Array.isArray(messages)).toBe(true);
    });

    it('should filter messages by type', () => {
      network.addAgent(new Agent('agent1', {
        needs: ['x'],
        offers: ['y']
      }));
      network.addAgent(new Agent('agent2', {
        needs: ['y'],
        offers: ['x']
      }));
      network.rebuildTopology();
      network.newRound();
      
      const round = new NegotiationRound(network);
      round.execute();
      
      const offers = round.getMessagesByType('offer');
      
      expect(Array.isArray(offers)).toBe(true);
    });

    it('should get messages for specific agent', () => {
      network.addAgent(new Agent('agent1'));
      network.addAgent(new Agent('agent2'));
      network.rebuildTopology();
      network.newRound();
      
      const round = new NegotiationRound(network);
      
      network.send({
        from: 'agent1',
        to: 'agent2',
        type: 'test',
        content: 'hello'
      });
      
      round.execute();
      
      const agentMessages = round.getMessagesForAgent('agent1');
      
      expect(Array.isArray(agentMessages)).toBe(true);
    });
  });

  describe('getOutcome', () => {
    it('should return incomplete before execution', () => {
      network.addAgent(new Agent('agent1'));
      network.newRound();
      
      const round = new NegotiationRound(network);
      
      const outcome = round.getOutcome();
      
      expect(outcome.complete).toBe(false);
    });

    it('should return complete after execution', () => {
      network.addAgent(new Agent('agent1'));
      network.newRound();
      
      const round = new NegotiationRound(network);
      round.execute();
      
      const outcome = round.getOutcome();
      
      expect(outcome.complete).toBe(true);
      expect(outcome.round).toBeDefined();
      expect(outcome.statistics).toBeDefined();
    });

    it('should include agent outcomes', () => {
      network.addAgent(new Agent('agent1'));
      network.addAgent(new Agent('agent2'));
      network.newRound();
      
      const round = new NegotiationRound(network);
      round.execute();
      
      const outcome = round.getOutcome();
      
      expect(outcome.agentOutcomes).toBeDefined();
      expect(outcome.agentOutcomes['agent1']).toBeDefined();
      expect(outcome.agentOutcomes['agent2']).toBeDefined();
    });
  });

  describe('getStatistics', () => {
    it('should return round statistics', () => {
      network.addAgent(new Agent('agent1'));
      network.newRound();
      
      const round = new NegotiationRound(network);
      round.execute();
      
      const stats = round.getStatistics();
      
      expect(stats.round).toBeDefined();
      expect(stats.completed).toBe(true);
      expect(typeof stats.messageCount).toBe('number');
      expect(typeof stats.offersMade).toBe('number');
      expect(typeof stats.acceptances).toBe('number');
      expect(typeof stats.rejections).toBe('number');
      expect(typeof stats.counters).toBe('number');
    });
  });

  describe('isComplete', () => {
    it('should return false before execution', () => {
      network.addAgent(new Agent('agent1'));
      network.newRound();
      
      const round = new NegotiationRound(network);
      
      expect(round.isComplete()).toBe(false);
    });

    it('should return true after execution', () => {
      network.addAgent(new Agent('agent1'));
      network.newRound();
      
      const round = new NegotiationRound(network);
      round.execute();
      
      expect(round.isComplete()).toBe(true);
    });
  });

  describe('getRoundNumber', () => {
    it('should return round number', () => {
      network.addAgent(new Agent('agent1'));
      network.newRound();
      
      const round = new NegotiationRound(network);
      
      expect(round.getRoundNumber()).toBe(1);
    });
  });

  describe('respectTopology option', () => {
    it('should respect topology when enabled', () => {
      const localNetwork = new AgentNetwork({ maxRounds: 10 });
      
      // Create agents that won't be connected
      localNetwork.addAgent(new Agent('agent1', {
        needs: ['x'],
        offers: ['y']
      }));
      localNetwork.addAgent(new Agent('agent2', {
        needs: ['unrelated'],
        offers: ['unrelated2']
      }));
      
      // Set high threshold so they won't connect
      localNetwork.topology.threshold = 0.99;
      localNetwork.rebuildTopology();
      localNetwork.newRound();
      
      const round = new NegotiationRound(localNetwork, {
        respectTopology: true
      });
      
      round.execute();
      
      // Round should complete without error
      expect(round.isComplete()).toBe(true);
    });
  });

  describe('custom strategy resolver', () => {
    it('should use custom strategy resolver', () => {
      let resolverCalled = false;
      
      network.addAgent(new Agent('agent1', {
        needs: ['coffee'],
        offers: ['tea']
      }));
      network.addAgent(new Agent('agent2', {
        needs: ['tea'],
        offers: ['coffee']
      }));
      network.rebuildTopology();
      network.newRound();
      
      const round = new NegotiationRound(network, {
        strategyResolver: (agent, messages) => {
          resolverCalled = true;
          return {
            shouldOffer: true,
            shouldAccept: true,
            shouldQuery: false
          };
        }
      });
      
      round.execute();
      
      expect(resolverCalled).toBe(true);
    });
  });

  describe('agent belief updates', () => {
    it('should update agent beliefs on message', () => {
      network.addAgent(new Agent('agent1'));
      network.addAgent(new Agent('agent2'));
      network.rebuildTopology();
      network.newRound();
      
      const agent1 = network.getAgent('agent1');
      
      network.send({
        from: 'agent2',
        to: 'agent1',
        type: 'offer',
        content: { offer: 'test' }
      });
      
      const round = new NegotiationRound(network);
      round.execute();
      
      const belief = agent1.getBelief('agent2');
      expect(belief).toBeDefined();
    });
  });

  describe('edge cases', () => {
    it('should handle agent with no neighbors', () => {
      network.addAgent(new Agent('agent1'));
      network.rebuildTopology();
      network.newRound();
      
      const round = new NegotiationRound(network);
      const result = round.execute();
      
      expect(result.success).toBe(true);
    });

    it('should handle empty network', () => {
      network.newRound();
      
      const round = new NegotiationRound(network);
      const result = round.execute();
      
      expect(result.success).toBe(true);
    });
  });
});
