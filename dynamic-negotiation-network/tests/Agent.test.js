/**
 * Unit tests for Agent class
 */

const Agent = require('../src/Agent');

describe('Agent', () => {
  let agent;

  beforeEach(() => {
    agent = new Agent('agent-1', {
      needs: ['apples', 'bananas'],
      offers: ['oranges', 'grapes'],
      beliefs: { 'agent-2': { trust: 0.8 } },
      strategy: 'counter',
      utility: 1.5
    });
  });

  describe('constructor', () => {
    test('should create agent with correct properties', () => {
      expect(agent.id).toBe('agent-1');
      expect(agent.needs).toEqual(['apples', 'bananas']);
      expect(agent.offers).toEqual(['oranges', 'grapes']);
      expect(agent.beliefs).toEqual({ 'agent-2': { trust: 0.8 } });
      expect(agent.strategy).toBe('counter');
      expect(agent.utility).toBe(1.5);
      expect(agent.history).toEqual([]);
    });

    test('should create agent with default properties', () => {
      const defaultAgent = new Agent('agent-2');
      expect(defaultAgent.id).toBe('agent-2');
      expect(defaultAgent.needs).toEqual([]);
      expect(defaultAgent.offers).toEqual([]);
      expect(defaultAgent.beliefs).toEqual({});
      expect(defaultAgent.strategy).toBe('default');
      expect(defaultAgent.utility).toBe(1.0);
    });
  });

  describe('getNeeds', () => {
    test('should return a copy of needs', () => {
      const needs = agent.getNeeds();
      needs.push('new-need');
      expect(agent.needs).toEqual(['apples', 'bananas']);
    });

    test('should return empty array if no needs', () => {
      agent.needs = [];
      expect(agent.getNeeds()).toEqual([]);
    });
  });

  describe('getOffers', () => {
    test('should return a copy of offers', () => {
      const offers = agent.getOffers();
      offers.push('new-offer');
      expect(agent.offers).toEqual(['oranges', 'grapes']);
    });
  });

  describe('setNeeds', () => {
    test('should replace needs array', () => {
      agent.setNeeds(['pears', 'plums']);
      expect(agent.needs).toEqual(['pears', 'plums']);
    });

    test('should create a copy of the input array', () => {
      const newNeeds = ['pears', 'plums'];
      agent.setNeeds(newNeeds);
      newNeeds.push('apples');
      expect(agent.needs).toEqual(['pears', 'plums']);
    });
  });

  describe('setOffers', () => {
    test('should replace offers array', () => {
      agent.setOffers(['kiwis', 'mangoes']);
      expect(agent.offers).toEqual(['kiwis', 'mangoes']);
    });
  });

  describe('addNeed', () => {
    test('should add a need to the list', () => {
      agent.addNeed('peaches');
      expect(agent.needs).toContain('peaches');
    });

    test('should not add duplicate needs', () => {
      agent.addNeed('apples');
      expect(agent.needs).toEqual(['apples', 'bananas']);
    });

    test('should handle empty need', () => {
      agent.addNeed('');
      expect(agent.needs).toContain('');
    });
  });

  describe('removeNeed', () => {
    test('should remove a need from the list', () => {
      agent.removeNeed('apples');
      expect(agent.needs).not.toContain('apples');
      expect(agent.needs).toEqual(['bananas']);
    });

    test('should handle removing non-existent need', () => {
      agent.removeNeed('non-existent');
      expect(agent.needs).toEqual(['apples', 'bananas']);
    });
  });

  describe('addOffer', () => {
    test('should add an offer to the list', () => {
      agent.addOffer('kiwis');
      expect(agent.offers).toContain('kiwis');
    });

    test('should not add duplicate offers', () => {
      agent.addOffer('oranges');
      expect(agent.offers).toEqual(['oranges', 'grapes']);
    });
  });

  describe('removeOffer', () => {
    test('should remove an offer from the list', () => {
      agent.removeOffer('oranges');
      expect(agent.offers).not.toContain('oranges');
      expect(agent.offers).toEqual(['grapes']);
    });

    test('should handle removing non-existent offer', () => {
      agent.removeOffer('non-existent');
      expect(agent.offers).toEqual(['oranges', 'grapes']);
    });
  });

  describe('updateBelief', () => {
    test('should add new belief about agent', () => {
      agent.updateBelief('agent-3', { trust: 0.5, lastSeen: 12345 });
      expect(agent.beliefs['agent-3']).toMatchObject({ trust: 0.5, lastSeen: 12345 });
      expect(agent.beliefs['agent-3'].timestamp).toBeDefined();
    });

    test('should update existing belief about agent', () => {
      agent.updateBelief('agent-2', { trust: 0.9 });
      expect(agent.beliefs['agent-2'].trust).toBe(0.9);
      expect(agent.beliefs['agent-2'].timestamp).toBeDefined();
    });

    test('should merge new properties with existing belief', () => {
      agent.updateBelief('agent-2', { lastSeen: 99999 });
      expect(agent.beliefs['agent-2'].trust).toBe(0.8); // Original preserved
      expect(agent.beliefs['agent-2'].lastSeen).toBe(99999);
    });
  });

  describe('getBelief', () => {
    test('should return belief for existing agent', () => {
      const belief = agent.getBelief('agent-2');
      expect(belief).toEqual({ trust: 0.8 });
    });

    test('should return null for non-existent agent', () => {
      const belief = agent.getBelief('non-existent');
      expect(belief).toBeNull();
    });
  });

  describe('recordEvent', () => {
    test('should add event to history', () => {
      agent.recordEvent({ type: 'offer_received', from: 'agent-2' });
      expect(agent.history).toHaveLength(1);
      expect(agent.history[0].type).toBe('offer_received');
    });

    test('should add timestamp to event', () => {
      agent.recordEvent({ type: 'test' });
      expect(agent.history[0].timestamp).toBeDefined();
    });
  });

  describe('getHistory', () => {
    test('should return a copy of history', () => {
      agent.recordEvent({ type: 'test' });
      const history = agent.getHistory();
      history.push({ type: 'fake' });
      expect(agent.history).toHaveLength(1);
    });

    test('should return empty array if no history', () => {
      expect(agent.getHistory()).toEqual([]);
    });
  });

  describe('clearHistory', () => {
    test('should clear all history', () => {
      agent.recordEvent({ type: 'test' });
      agent.clearHistory();
      expect(agent.history).toEqual([]);
    });
  });

  describe('needsSatisfied', () => {
    test('should return true when no needs', () => {
      agent.needs = [];
      expect(agent.needsSatisfied()).toBe(true);
    });

    test('should return false when has needs', () => {
      expect(agent.needsSatisfied()).toBe(false);
    });
  });

  describe('getStrategy', () => {
    test('should return agent strategy', () => {
      expect(agent.getStrategy()).toBe('counter');
    });
  });

  describe('setStrategy', () => {
    test('should set agent strategy', () => {
      agent.setStrategy('accept');
      expect(agent.strategy).toBe('accept');
    });
  });

  describe('respondToOffer', () => {
    test('should accept offer that matches a need', () => {
      const response = agent.respondToOffer('fresh apples');
      expect(response.action).toBe('accept');
      expect(response.matchedNeed).toBe('apples');
    });

    test('should accept offer with case-insensitive match', () => {
      const response = agent.respondToOffer('FRESH BANANAS');
      expect(response.action).toBe('accept');
    });

    test('should counter when no match found', () => {
      const response = agent.respondToOffer('something random');
      expect(response.action).toBe('counter');
      expect(agent.getOffers()).toContain(response.counterOffer);
    });

    test('should handle empty offer', () => {
      const response = agent.respondToOffer('');
      expect(response.action).toBe('counter');
    });
  });

  describe('toJSON', () => {
    test('should serialize agent to JSON', () => {
      agent.recordEvent({ type: 'test' });
      const json = agent.toJSON();
      expect(json.id).toBe('agent-1');
      expect(json.needs).toEqual(['apples', 'bananas']);
      expect(json.offers).toEqual(['oranges', 'grapes']);
      expect(json.beliefs).toEqual({ 'agent-2': { trust: 0.8 } });
      expect(json.history).toHaveLength(1);
      expect(json.strategy).toBe('counter');
      expect(json.utility).toBe(1.5);
    });
  });

  describe('fromJSON', () => {
    test('should create agent from JSON', () => {
      const json = {
        id: 'agent-1',
        needs: ['apples', 'bananas'],
        offers: ['oranges', 'grapes'],
        beliefs: { 'agent-2': { trust: 0.8 } },
        history: [{ type: 'test', timestamp: 12345 }],
        strategy: 'counter',
        utility: 1.5
      };
      
      const restoredAgent = Agent.fromJSON(json);
      expect(restoredAgent.id).toBe('agent-1');
      expect(restoredAgent.needs).toEqual(['apples', 'bananas']);
      expect(restoredAgent.offers).toEqual(['oranges', 'grapes']);
      expect(restoredAgent.history).toHaveLength(1);
    });

    test('should handle JSON without history', () => {
      const json = {
        id: 'agent-1',
        needs: ['apples'],
        offers: ['oranges'],
        beliefs: {},
        strategy: 'default',
        utility: 1.0
      };
      
      const restoredAgent = Agent.fromJSON(json);
      expect(restoredAgent.history).toEqual([]);
    });
  });
});
