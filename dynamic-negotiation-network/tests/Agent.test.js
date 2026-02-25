/**
 * Unit tests for Agent class
 */

const Agent = require('../src/Agent');

describe('Agent', () => {
  describe('constructor', () => {
    it('should create an agent with default values', () => {
      const agent = new Agent('agent1');
      expect(agent.id).toBe('agent1');
      expect(agent.needs).toEqual([]);
      expect(agent.offers).toEqual([]);
      expect(agent.beliefs).toEqual({});
      expect(agent.history).toEqual([]);
      expect(agent.strategy).toBe('default');
      expect(agent.utility).toBe(1.0);
    });

    it('should create an agent with custom options', () => {
      const agent = new Agent('agent2', {
        needs: ['coffee', 'lunch'],
        offers: ['tea', 'snacks'],
        beliefs: { otherAgent: { trust: 0.5 } },
        strategy: 'aggressive',
        utility: 0.8
      });
      expect(agent.id).toBe('agent2');
      expect(agent.needs).toEqual(['coffee', 'lunch']);
      expect(agent.offers).toEqual(['tea', 'snacks']);
      expect(agent.beliefs).toEqual({ otherAgent: { trust: 0.5 } });
      expect(agent.strategy).toBe('aggressive');
      expect(agent.utility).toBe(0.8);
    });

    it('should create independent copies of arrays', () => {
      const needs = ['need1'];
      const offers = ['offer1'];
      const agent = new Agent('agent3', { needs, offers });
      
      needs.push('need2');
      offers.push('offer2');
      
      expect(agent.needs).toEqual(['need1']);
      expect(agent.offers).toEqual(['offer1']);
    });
  });

  describe('getNeeds and setNeeds', () => {
    it('should return a copy of needs array', () => {
      const agent = new Agent('agent1', { needs: ['need1'] });
      const needs = agent.getNeeds();
      needs.push('need2');
      expect(agent.needs).toEqual(['need1']);
    });

    it('should set new needs', () => {
      const agent = new Agent('agent1');
      agent.setNeeds(['need1', 'need2']);
      expect(agent.needs).toEqual(['need1', 'need2']);
    });

    it('should create a copy when setting needs', () => {
      const agent = new Agent('agent1');
      const newNeeds = ['need1'];
      agent.setNeeds(newNeeds);
      newNeeds.push('need2');
      expect(agent.needs).toEqual(['need1']);
    });
  });

  describe('getOffers and setOffers', () => {
    it('should return a copy of offers array', () => {
      const agent = new Agent('agent1', { offers: ['offer1'] });
      const offers = agent.getOffers();
      offers.push('offer2');
      expect(agent.offers).toEqual(['offer1']);
    });

    it('should set new offers', () => {
      const agent = new Agent('agent1');
      agent.setOffers(['offer1', 'offer2']);
      expect(agent.offers).toEqual(['offer1', 'offer2']);
    });
  });

  describe('addNeed and removeNeed', () => {
    it('should add a need if not present', () => {
      const agent = new Agent('agent1');
      agent.addNeed('coffee');
      expect(agent.needs).toContain('coffee');
    });

    it('should not duplicate needs', () => {
      const agent = new Agent('agent1', { needs: ['coffee'] });
      agent.addNeed('coffee');
      expect(agent.needs).toEqual(['coffee']);
    });

    it('should remove a need', () => {
      const agent = new Agent('agent1', { needs: ['coffee', 'tea'] });
      agent.removeNeed('coffee');
      expect(agent.needs).toEqual(['tea']);
    });

    it('should handle removing non-existent need', () => {
      const agent = new Agent('agent1', { needs: ['coffee'] });
      agent.removeNeed('nonexistent');
      expect(agent.needs).toEqual(['coffee']);
    });
  });

  describe('addOffer and removeOffer', () => {
    it('should add an offer if not present', () => {
      const agent = new Agent('agent1');
      agent.addOffer('service');
      expect(agent.offers).toContain('service');
    });

    it('should not duplicate offers', () => {
      const agent = new Agent('agent1', { offers: ['service'] });
      agent.addOffer('service');
      expect(agent.offers).toEqual(['service']);
    });

    it('should remove an offer', () => {
      const agent = new Agent('agent1', { offers: ['service', 'product'] });
      agent.removeOffer('service');
      expect(agent.offers).toEqual(['product']);
    });
  });

  describe('belief management', () => {
    it('should update belief about another agent', () => {
      const agent = new Agent('agent1');
      agent.updateBelief('agent2', { trust: 0.8 });
      
      expect(agent.beliefs.agent2).toBeDefined();
      expect(agent.beliefs.agent2.trust).toBe(0.8);
      expect(agent.beliefs.agent2.timestamp).toBeDefined();
    });

    it('should merge beliefs on update', () => {
      const agent = new Agent('agent1');
      agent.updateBelief('agent2', { trust: 0.5 });
      agent.updateBelief('agent2', { reliability: 0.9 });
      
      expect(agent.beliefs.agent2.trust).toBe(0.5);
      expect(agent.beliefs.agent2.reliability).toBe(0.9);
    });

    it('should get belief about another agent', () => {
      const agent = new Agent('agent1');
      agent.updateBelief('agent2', { trust: 0.7 });
      
      const belief = agent.getBelief('agent2');
      expect(belief.trust).toBe(0.7);
    });

    it('should return null for unknown agent belief', () => {
      const agent = new Agent('agent1');
      expect(agent.getBelief('unknown')).toBeNull();
    });
  });

  describe('history management', () => {
    it('should record events in history', () => {
      const agent = new Agent('agent1');
      agent.recordEvent({ type: 'offer', value: 100 });
      
      expect(agent.history.length).toBe(1);
      expect(agent.history[0].type).toBe('offer');
      expect(agent.history[0].value).toBe(100);
      expect(agent.history[0].timestamp).toBeDefined();
    });

    it('should return a copy of history', () => {
      const agent = new Agent('agent1');
      agent.recordEvent({ type: 'offer' });
      
      const history = agent.getHistory();
      history.push({ type: 'fake' });
      
      expect(agent.history.length).toBe(1);
    });

    it('should clear history', () => {
      const agent = new Agent('agent1');
      agent.recordEvent({ type: 'offer' });
      agent.recordEvent({ type: 'accept' });
      
      agent.clearHistory();
      expect(agent.history).toEqual([]);
    });
  });

  describe('needsSatisfied', () => {
    it('should return true when no needs', () => {
      const agent = new Agent('agent1');
      expect(agent.needsSatisfied()).toBe(true);
    });

    it('should return false when needs exist', () => {
      const agent = new Agent('agent1', { needs: ['coffee'] });
      expect(agent.needsSatisfied()).toBe(false);
    });
  });

  describe('strategy management', () => {
    it('should get and set strategy', () => {
      const agent = new Agent('agent1');
      expect(agent.getStrategy()).toBe('default');
      
      agent.setStrategy('aggressive');
      expect(agent.getStrategy()).toBe('aggressive');
    });
  });

  describe('respondToOffer', () => {
    it('should accept offer that matches a need', () => {
      const agent = new Agent('agent1', { needs: ['coffee'] });
      const response = agent.respondToOffer('I have coffee available');
      
      expect(response.action).toBe('accept');
      expect(response.matchedNeed).toBe('coffee');
    });

    it('should counter when offer does not match', () => {
      const agent = new Agent('agent1', {
        needs: ['coffee'],
        offers: ['tea', 'water']
      });
      const response = agent.respondToOffer('I have juice available');
      
      expect(response.action).toBe('counter');
      expect(['tea', 'water']).toContain(response.counterOffer);
    });

    it('should match needs case-insensitively', () => {
      const agent = new Agent('agent1', { needs: ['Coffee'] });
      const response = agent.respondToOffer('i have coffee');
      
      expect(response.action).toBe('accept');
    });
  });

  describe('toJSON and fromJSON', () => {
    it('should serialize to JSON', () => {
      const agent = new Agent('agent1', {
        needs: ['coffee'],
        offers: ['tea'],
        strategy: 'aggressive',
        utility: 0.9
      });
      agent.recordEvent({ type: 'test' });
      
      const json = agent.toJSON();
      
      expect(json.id).toBe('agent1');
      expect(json.needs).toEqual(['coffee']);
      expect(json.offers).toEqual(['tea']);
      expect(json.strategy).toBe('aggressive');
      expect(json.utility).toBe(0.9);
      expect(json.history.length).toBe(1);
    });

    it('should deserialize from JSON', () => {
      const json = {
        id: 'agent1',
        needs: ['coffee'],
        offers: ['tea'],
        beliefs: { other: { trust: 0.5 } },
        history: [{ type: 'test', timestamp: 123 }],
        strategy: 'aggressive',
        utility: 0.8
      };
      
      const agent = Agent.fromJSON(json);
      
      expect(agent.id).toBe('agent1');
      expect(agent.needs).toEqual(['coffee']);
      expect(agent.offers).toEqual(['tea']);
      expect(agent.strategy).toBe('aggressive');
      expect(agent.utility).toBe(0.8);
      expect(agent.history.length).toBe(1);
    });

    it('should handle missing history in JSON', () => {
      const json = {
        id: 'agent1',
        needs: [],
        offers: []
      };
      
      const agent = Agent.fromJSON(json);
      expect(agent.history).toEqual([]);
    });
  });
});
