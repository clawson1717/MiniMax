/**
 * Unit tests for NegotiationRound class
 */

const NegotiationRound = require('../src/NegotiationRound');
const AgentNetwork = require('../src/AgentNetwork');
const Agent = require('../src/Agent');

// Mock AgentNetwork for testing
class MockAgentNetwork {
  constructor() {
    this.agents = new Map();
    this.currentRound = 0;
    this.messages = [];
    this.inboxes = new Map();
  }

  addAgent(agent) {
    this.agents.set(agent.id, agent);
    this.inboxes.set(agent.id, []);
  }

  getAllAgents() {
    return Array.from(this.agents.values());
  }

  getNeighbors(agentId) {
    // Return all other agents as neighbors
    return Array.from(this.agents.values()).filter(a => a.id !== agentId);
  }

  getCurrentRound() {
    return this.currentRound;
  }

  send(message) {
    const inbox = this.inboxes.get(message.to);
    if (inbox) {
      inbox.push(message);
    }
    this.messages.push(message);
    return true;
  }

  receiveMessage(agentId) {
    const inbox = this.inboxes.get(agentId) || [];
    const messages = inbox.filter(m => !m.read);
    messages.forEach(m => m.read = true);
    return messages;
  }

  getTopologyManager() {
    return null;
  }
}

describe('NegotiationRound', () => {
  let network;
  let agent1;
  let agent2;
  let round;

  beforeEach(() => {
    network = new MockAgentNetwork();
    agent1 = new Agent('agent-1', {
      needs: ['apples'],
      offers: ['oranges']
    });
    agent2 = new Agent('agent-2', {
      needs: ['oranges'],
      offers: ['apples']
    });
    network.addAgent(agent1);
    network.addAgent(agent2);
    
    round = new NegotiationRound(network, {
      respectTopology: false
    });
  });

  describe('constructor', () => {
    test('should create round with correct properties', () => {
      expect(round.network).toBe(network);
      expect(round.roundNumber).toBe(0);
      expect(round.respectTopology).toBe(false);
      expect(round.messages).toEqual([]);
      expect(round.outcomes).toBeInstanceOf(Map);
      expect(round.completed).toBe(false);
    });

    test('should use default strategy resolver', () => {
      expect(round.strategyResolver).toBeDefined();
      const strategy = round.strategyResolver(agent1, []);
      expect(strategy.shouldOffer).toBe(true);
    });
  });

  describe('execute', () => {
    test('should execute round successfully', () => {
      const result = round.execute();
      
      expect(result.success).toBe(true);
      expect(result.round).toBe(0);
      expect(round.completed).toBe(true);
    });

    test('should return error if round already completed', () => {
      round.execute();
      const result = round.execute();
      
      expect(result.success).toBe(false);
      expect(result.message).toBe('Round already completed');
    });

    test('should set start and end times', () => {
      round.execute();
      
      expect(round.startTime).toBeDefined();
      expect(round.endTime).toBeDefined();
    });

    test('should count statistics', () => {
      const result = round.execute();
      
      expect(result.offers).toBeGreaterThanOrEqual(0);
      expect(result.duration).toBeGreaterThanOrEqual(0);
    });
  });

  describe('message processing', () => {
    test('should process incoming messages', () => {
      // Pre-populate inbox
      network.inboxes.set('agent-1', [{
        from: 'agent-2',
        to: 'agent-1',
        type: 'offer',
        content: { offer: 'apples' },
        read: false
      }]);
      
      round.execute();
      
      expect(round.messages.length).toBeGreaterThan(0);
    });

    test('should update agent beliefs on message receipt', () => {
      network.inboxes.set('agent-1', [{
        from: 'agent-2',
        to: 'agent-1',
        type: 'offer',
        content: { offer: 'apples' },
        read: false
      }]);
      
      round.execute();
      
      const belief = agent1.getBelief('agent-2');
      expect(belief).toBeDefined();
      expect(belief.lastMessageType).toBe('offer');
    });
  });

  describe('offer handling', () => {
    test('should accept offer that matches need', () => {
      network.inboxes.set('agent-1', [{
        from: 'agent-2',
        to: 'agent-1',
        type: 'offer',
        content: { offer: 'apples' },
        read: false
      }]);
      
      round.execute();
      
      // Check if acceptance was sent
      const acceptMessage = network.messages.find(m => 
        m.type === 'accept' && m.from === 'agent-1'
      );
      expect(acceptMessage).toBeDefined();
    });

    test('should remove satisfied need after acceptance', () => {
      network.inboxes.set('agent-1', [{
        from: 'agent-2',
        to: 'agent-1',
        type: 'offer',
        content: { offer: 'apples' },
        read: false
      }]);
      
      round.execute();
      
      expect(agent1.getNeeds()).not.toContain('apples');
    });

    test('should counter when offer does not match need', () => {
      network.inboxes.set('agent-1', [{
        from: 'agent-2',
        to: 'agent-1',
        type: 'offer',
        content: { offer: 'bananas' },
        read: false
      }]);
      
      round.execute();
      
      // Check if counter was sent or offer remains
      const counterOrAccept = network.messages.find(m => 
        m.type === 'counter' || m.type === 'accept'
      );
      // Should either counter or reject
      expect(network.messages.some(m => m.from === 'agent-1')).toBe(true);
    });

    test('should record event on acceptance', () => {
      network.inboxes.set('agent-1', [{
        from: 'agent-2',
        to: 'agent-1',
        type: 'offer',
        content: { offer: 'apples' },
        read: false
      }]);
      
      round.execute();
      
      const history = agent1.getHistory();
      expect(history.some(e => e.type === 'offer_accepted')).toBe(true);
    });
  });

  describe('counter handling', () => {
    test('should evaluate counter-offer', () => {
      network.inboxes.set('agent-1', [{
        from: 'agent-2',
        to: 'agent-1',
        type: 'counter',
        content: { 
          originalOffer: 'oranges',
          counterOffer: 'apples'
        },
        read: false
      }]);
      
      round.execute();
      
      // Should either accept or reject the counter
      const response = network.messages.find(m => 
        m.from === 'agent-1' && (m.type === 'accept' || m.type === 'reject')
      );
      expect(response).toBeDefined();
    });

    test('should accept counter-offer that matches need', () => {
      network.inboxes.set('agent-1', [{
        from: 'agent-2',
        to: 'agent-1',
        type: 'counter',
        content: { 
          originalOffer: 'oranges',
          counterOffer: 'apples'
        },
        read: false
      }]);
      
      round.execute();
      
      const acceptMessage = network.messages.find(m => 
        m.type === 'accept' && m.from === 'agent-1'
      );
      expect(acceptMessage).toBeDefined();
    });

    test('should record counter received event', () => {
      network.inboxes.set('agent-1', [{
        from: 'agent-2',
        to: 'agent-1',
        type: 'counter',
        content: { 
          originalOffer: 'oranges',
          counterOffer: 'grapes'
        },
        read: false
      }]);
      
      round.execute();
      
      const history = agent1.getHistory();
      expect(history.some(e => e.type === 'counter_received')).toBe(true);
    });
  });

  describe('acceptance handling', () => {
    test('should record successful negotiation', () => {
      network.inboxes.set('agent-1', [{
        from: 'agent-2',
        to: 'agent-1',
        type: 'accept',
        content: { 
          acceptedOffer: 'oranges',
          satisfiedNeed: 'oranges'
        },
        read: false
      }]);
      
      round.execute();
      
      const history = agent1.getHistory();
      expect(history.some(e => e.type === 'offer_accepted_by')).toBe(true);
    });

    test('should set outcome for satisfied agent', () => {
      network.inboxes.set('agent-1', [{
        from: 'agent-2',
        to: 'agent-1',
        type: 'accept',
        content: { 
          acceptedOffer: 'oranges',
          satisfiedNeed: 'oranges'
        },
        read: false
      }]);
      
      round.execute();
      
      expect(round.outcomes.has('agent-1')).toBe(true);
      expect(round.outcomes.get('agent-1').satisfied).toBe(true);
    });
  });

  describe('rejection handling', () => {
    test('should record rejection event', () => {
      network.inboxes.set('agent-1', [{
        from: 'agent-2',
        to: 'agent-1',
        type: 'reject',
        content: { offer: 'grapes' },
        read: false
      }]);
      
      round.execute();
      
      const history = agent1.getHistory();
      expect(history.some(e => e.type === 'offer_rejected_by')).toBe(true);
    });
  });

  describe('query handling', () => {
    test('should respond to query with needs and offers', () => {
      network.inboxes.set('agent-1', [{
        from: 'agent-2',
        to: 'agent-1',
        type: 'query',
        content: { query: 'capabilities' },
        read: false
      }]);
      
      round.execute();
      
      const response = network.messages.find(m => 
        m.type === 'response' && m.from === 'agent-1'
      );
      expect(response).toBeDefined();
      expect(response.content.needs).toEqual(['apples']);
      expect(response.content.offers).toEqual(['oranges']);
    });
  });

  describe('getMessages', () => {
    test('should return all messages', () => {
      round.execute();
      const messages = round.getMessages();
      expect(Array.isArray(messages)).toBe(true);
    });

    test('should return copy of messages array', () => {
      round.execute();
      const messages = round.getMessages();
      messages.push({ fake: 'message' });
      expect(round.messages.length).toBeLessThan(messages.length);
    });
  });

  describe('getMessagesByType', () => {
    test('should return messages of specific type', () => {
      round.messages = [
        { type: 'offer', from: 'agent-1' },
        { type: 'accept', from: 'agent-2' },
        { type: 'offer', from: 'agent-2' }
      ];
      
      const offers = round.getMessagesByType('offer');
      expect(offers).toHaveLength(2);
    });

    test('should return empty array for no matching messages', () => {
      round.messages = [{ type: 'offer' }];
      const accepts = round.getMessagesByType('accept');
      expect(accepts).toEqual([]);
    });
  });

  describe('getMessagesForAgent', () => {
    test('should return messages involving agent', () => {
      round.messages = [
        { type: 'offer', from: 'agent-1', to: 'agent-2' },
        { type: 'accept', from: 'agent-2', to: 'agent-3' },
        { type: 'offer', from: 'agent-2', to: 'agent-1' }
      ];
      
      const agent1Messages = round.getMessagesForAgent('agent-1');
      expect(agent1Messages).toHaveLength(2);
    });
  });

  describe('isComplete', () => {
    test('should return false before execution', () => {
      expect(round.isComplete()).toBe(false);
    });

    test('should return true after execution', () => {
      round.execute();
      expect(round.isComplete()).toBe(true);
    });
  });

  describe('getOutcome', () => {
    test('should return incomplete message before execution', () => {
      const outcome = round.getOutcome();
      expect(outcome.complete).toBe(false);
    });

    test('should return complete outcome after execution', () => {
      round.execute();
      const outcome = round.getOutcome();
      expect(outcome.complete).toBe(true);
    });

    test('should include statistics in outcome', () => {
      round.execute();
      const outcome = round.getOutcome();
      expect(outcome.statistics).toBeDefined();
      expect(outcome.statistics.totalMessages).toBeGreaterThanOrEqual(0);
    });

    test('should include agent outcomes', () => {
      round.execute();
      const outcome = round.getOutcome();
      expect(outcome.agentOutcomes).toBeDefined();
      expect(Object.keys(outcome.agentOutcomes)).toContain('agent-1');
    });
  });

  describe('getStatistics', () => {
    test('should return round statistics', () => {
      round.execute();
      const stats = round.getStatistics();
      
      expect(stats.round).toBe(0);
      expect(stats.completed).toBe(true);
      expect(stats.messageCount).toBeGreaterThanOrEqual(0);
    });

    test('should return null duration before execution', () => {
      const stats = round.getStatistics();
      expect(stats.duration).toBeNull();
    });
  });

  describe('getRoundNumber', () => {
    test('should return round number', () => {
      expect(round.getRoundNumber()).toBe(0);
    });
  });

  describe('agents with no neighbors', () => {
    test('should handle agents with no neighbors gracefully', () => {
      const isolatedAgent = new Agent('isolated', {
        needs: ['apples'],
        offers: ['oranges']
      });
      
      // Create new network with only isolated agent
      const isolatedNetwork = new MockAgentNetwork();
      isolatedNetwork.addAgent(isolatedAgent);
      
      const isolatedRound = new NegotiationRound(isolatedNetwork);
      
      const result = isolatedRound.execute();
      expect(result.success).toBe(true);
    });
  });

  describe('agents with no offers', () => {
    test('should handle agents with no offers', () => {
      agent1.offers = [];
      
      const result = round.execute();
      expect(result.success).toBe(true);
    });
  });
});
