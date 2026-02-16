/**
 * Unit tests for Agent class
 */

const { Agent } = require('../src/index');

describe('Agent', () => {
  let agent;

  beforeEach(() => {
    agent = new Agent('agent-1', 'Test Agent');
  });

  describe('constructor', () => {
    test('should create agent with correct properties', () => {
      expect(agent.id).toBe('agent-1');
      expect(agent.name).toBe('Test Agent');
      expect(agent.beliefNetwork).toBeDefined();
      expect(agent.messageHistory).toHaveLength(0);
      expect(agent.subscriptions.size).toBe(0);
    });
  });

  describe('addBelief', () => {
    test('should add belief to agent network', () => {
      const belief = agent.addBelief('Test', 0.8, 'Reason');
      
      expect(belief).toBeDefined();
      expect(agent.getBelief('Test')).toBeDefined();
    });
  });

  describe('updateBelief', () => {
    test('should update belief in network', () => {
      agent.addBelief('Test', 0.5, 'Reason');
      const updated = agent.updateBelief('Test', 0.8, 'New reason');
      
      expect(agent.getBelief('Test').confidence).toBe(0.8);
      expect(updated).toBeDefined();
    });
  });

  describe('getBeliefs', () => {
    test('should return all beliefs', () => {
      agent.addBelief('A', 0.9, 'Reason');
      agent.addBelief('B', 0.8, 'Reason');
      
      const beliefs = agent.getBeliefs();
      
      expect(beliefs).toHaveLength(2);
    });
  });

  describe('getBelief', () => {
    test('should return specific belief', () => {
      agent.addBelief('Test', 0.5, 'Reason');
      
      const belief = agent.getBelief('Test');
      
      expect(belief).toBeDefined();
      expect(belief.proposition).toBe('Test');
    });
  });

  describe('receiveMessage', () => {
    test('should handle belief_update message', () => {
      agent.addBelief('Test', 0.5, 'Initial');
      
      const message = {
        type: 'belief_update',
        senderId: 'agent-2',
        payload: {
          proposition: 'Test',
          confidence: 0.9,
          justification: 'New info'
        },
        timestamp: Date.now()
      };
      
      const response = agent.receiveMessage(message);
      
      expect(response).toBeDefined();
      expect(response.type).toBe('response');
      expect(agent.getBelief('Test').confidence).toBe(0.7); // Average of 0.5 and 0.9
    });

    test('should add new belief from belief_update', () => {
      const message = {
        type: 'belief_update',
        senderId: 'agent-2',
        payload: {
          proposition: 'New Belief',
          confidence: 0.8,
          justification: 'From agent-2'
        },
        timestamp: Date.now()
      };
      
      agent.receiveMessage(message);
      
      expect(agent.getBelief('New Belief')).toBeDefined();
    });

    test('should handle query message', () => {
      agent.addBelief('Test', 0.5, 'Reason');
      
      const message = {
        type: 'query',
        senderId: 'agent-2',
        payload: {
          queryType: 'get_belief',
          proposition: 'Test'
        },
        timestamp: Date.now()
      };
      
      const response = agent.receiveMessage(message);
      
      expect(response).toBeDefined();
      expect(response.type).toBe('response');
      expect(response.payload.data.proposition).toBe('Test');
    });

    test('should handle get_all_beliefs query', () => {
      agent.addBelief('A', 0.9, 'Reason');
      agent.addBelief('B', 0.8, 'Reason');
      
      const message = {
        type: 'query',
        senderId: 'agent-2',
        payload: {
          queryType: 'get_all_beliefs'
        },
        timestamp: Date.now()
      };
      
      const response = agent.receiveMessage(message);
      
      expect(response.payload.data).toHaveLength(2);
    });

    test('should log received messages', () => {
      const message = {
        type: 'belief_update',
        senderId: 'agent-2',
        payload: { proposition: 'Test', confidence: 0.5, justification: 'Reason' },
        timestamp: Date.now()
      };
      
      agent.receiveMessage(message);
      
      expect(agent.getMessageHistory()).toHaveLength(1);
    });
  });

  describe('subscriptions', () => {
    test('should subscribe to topic', () => {
      agent.subscribe('weather');
      
      expect(agent.isSubscribed('weather')).toBe(true);
    });

    test('should unsubscribe from topic', () => {
      agent.subscribe('weather');
      agent.unsubscribe('weather');
      
      expect(agent.isSubscribed('weather')).toBe(false);
    });
  });

  describe('getStats', () => {
    test('should return agent statistics', () => {
      agent.addBelief('Test', 0.5, 'Reason');
      
      const stats = agent.getStats();
      
      expect(stats.id).toBe('agent-1');
      expect(stats.name).toBe('Test Agent');
      expect(stats.beliefCount).toBe(1);
    });
  });

  describe('clearMessageHistory', () => {
    test('should clear message history', () => {
      agent.receiveMessage({
        type: 'belief_update',
        senderId: 'agent-2',
        payload: { proposition: 'Test', confidence: 0.5, justification: 'Reason' },
        timestamp: Date.now()
      });
      
      agent.clearMessageHistory();
      
      expect(agent.getMessageHistory()).toHaveLength(0);
    });
  });

  describe('toJSON', () => {
    test('should serialize agent to JSON', () => {
      agent.addBelief('Test', 0.5, 'Reason');
      
      const json = agent.toJSON();
      
      expect(json.id).toBe('agent-1');
      expect(json.name).toBe('Test Agent');
      expect(json.beliefNetwork).toBeDefined();
    });
  });
});
