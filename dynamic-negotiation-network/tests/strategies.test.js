/**
 * Unit tests for negotiation strategies
 */

const {
  AcceptStrategy,
  RejectStrategy,
  CounterStrategy,
  RandomStrategy,
  StrategyFactory
} = require('../src/strategies');
const Agent = require('../src/Agent');

describe('Strategies', () => {
  describe('AcceptStrategy', () => {
    it('should create with default threshold', () => {
      const strategy = new AcceptStrategy();
      
      expect(strategy.threshold).toBe(0.3);
      expect(strategy.name).toBe('accept');
    });

    it('should create with custom threshold', () => {
      const strategy = new AcceptStrategy({ threshold: 0.5 });
      
      expect(strategy.threshold).toBe(0.5);
    });

    it('should accept matching offer', () => {
      const strategy = new AcceptStrategy({ threshold: 0.3 });
      const agent = new Agent('agent1', { needs: ['coffee'] });
      
      const result = strategy.evaluate({
        offer: 'coffee beans',
        agent
      });
      
      expect(result.action).toBe('accept');
      expect(result.matchedNeed).toBeDefined();
    });

    it('should reject non-matching offer', () => {
      const strategy = new AcceptStrategy({ threshold: 0.5 });
      const agent = new Agent('agent1', { needs: ['coffee'] });
      
      const result = strategy.evaluate({
        offer: 'automobile',
        agent
      });
      
      expect(result.action).toBe('reject');
    });

    it('should accept exact match', () => {
      const strategy = new AcceptStrategy();
      const agent = new Agent('agent1', { needs: ['coffee'] });
      
      const result = strategy.evaluate({
        offer: 'coffee',
        agent
      });
      
      expect(result.action).toBe('accept');
      expect(result.confidence).toBe(1);
    });

    it('should handle sender offers', () => {
      const strategy = new AcceptStrategy({ threshold: 0.3 });
      const agent = new Agent('agent1', { needs: ['coffee'] });
      
      const result = strategy.evaluate({
        offer: 'something else',
        agent,
        senderOffers: ['coffee', 'tea']
      });
      
      expect(result.action).toBe('accept');
    });

    it('should return name and description', () => {
      const strategy = new AcceptStrategy();
      
      expect(strategy.getName()).toBe('accept');
      expect(typeof strategy.getDescription()).toBe('string');
    });
  });

  describe('RejectStrategy', () => {
    it('should create with default probability', () => {
      const strategy = new RejectStrategy();
      
      expect(strategy.probability).toBe(1.0);
      expect(strategy.name).toBe('reject');
    });

    it('should create with custom probability', () => {
      const strategy = new RejectStrategy({ probability: 0.8 });
      
      expect(strategy.probability).toBe(0.8);
    });

    it('should reject by default', () => {
      const strategy = new RejectStrategy({ probability: 1.0 });
      const agent = new Agent('agent1');
      
      const result = strategy.evaluate({
        offer: 'anything',
        agent
      });
      
      expect(result.action).toBe('reject');
    });

    it('should occasionally accept when probability < 1', () => {
      const strategy = new RejectStrategy({ probability: 0.5 });
      const agent = new Agent('agent1');
      
      // Run multiple times to test probability
      let accepts = 0;
      for (let i = 0; i < 100; i++) {
        const result = strategy.evaluate({ offer: 'test', agent });
        if (result.action === 'accept') accepts++;
      }
      
      // With probability 0.5, we expect roughly 50 accepts
      expect(accepts).toBeGreaterThan(0);
      expect(accepts).toBeLessThan(100);
    });

    it('should return name and description', () => {
      const strategy = new RejectStrategy();
      
      expect(strategy.getName()).toBe('reject');
      expect(typeof strategy.getDescription()).toBe('string');
    });
  });

  describe('CounterStrategy', () => {
    it('should create with default options', () => {
      const strategy = new CounterStrategy();
      
      expect(strategy.counterThreshold).toBe(0.2);
      expect(strategy.concessionRate).toBe(0.1);
      expect(strategy.name).toBe('counter');
    });

    it('should create with custom options', () => {
      const strategy = new CounterStrategy({
        counterThreshold: 0.3,
        concessionRate: 0.2
      });
      
      expect(strategy.counterThreshold).toBe(0.3);
      expect(strategy.concessionRate).toBe(0.2);
    });

    it('should accept good enough offer', () => {
      const strategy = new CounterStrategy({ counterThreshold: 0.3 });
      const agent = new Agent('agent1', { needs: ['coffee'] });
      
      const result = strategy.evaluate({
        offer: 'coffee',
        agent
      });
      
      expect(result.action).toBe('accept');
    });

    it('should counter non-matching offer', () => {
      const strategy = new CounterStrategy({ counterThreshold: 0.9 });
      const agent = new Agent('agent1', {
        needs: ['coffee'],
        offers: ['tea', 'water']
      });
      
      const result = strategy.evaluate({
        offer: 'automobile',
        agent
      });
      
      expect(result.action).toBe('counter');
      expect(result.counterOffer).toBeDefined();
    });

    it('should reject when no counter available', () => {
      const strategy = new CounterStrategy({ counterThreshold: 0.9 });
      const agent = new Agent('agent1', {
        needs: ['coffee'],
        offers: []
      });
      
      const result = strategy.evaluate({
        offer: 'automobile',
        agent
      });
      
      expect(result.action).toBe('reject');
    });

    it('should generate counter with numeric values', () => {
      const strategy = new CounterStrategy();
      const agent = new Agent('agent1', {
        needs: ['coffee'],
        offers: ['$100 for coffee']
      });
      
      const result = strategy.evaluate({
        offer: '$50 for coffee',
        agent
      });
      
      expect(result.action).toBe('counter');
      expect(result.counterOffer).toBeDefined();
    });

    it('should reset round counter', () => {
      const strategy = new CounterStrategy();
      strategy.roundCount = 10;
      
      strategy.reset();
      
      expect(strategy.roundCount).toBe(0);
    });

    it('should return name and description', () => {
      const strategy = new CounterStrategy();
      
      expect(strategy.getName()).toBe('counter');
      expect(typeof strategy.getDescription()).toBe('string');
    });
  });

  describe('RandomStrategy', () => {
    it('should create with default probabilities', () => {
      const strategy = new RandomStrategy();
      
      expect(strategy.acceptProbability).toBe(0.3);
      expect(strategy.counterProbability).toBe(0.4);
      expect(strategy.name).toBe('random');
    });

    it('should create with custom probabilities', () => {
      const strategy = new RandomStrategy({
        acceptProbability: 0.5,
        counterProbability: 0.3
      });
      
      expect(strategy.acceptProbability).toBe(0.5);
      expect(strategy.counterProbability).toBe(0.3);
    });

    it('should produce random decisions', () => {
      const strategy = new RandomStrategy();
      const agent = new Agent('agent1', { offers: ['tea'] });
      
      const actions = new Set();
      for (let i = 0; i < 100; i++) {
        const result = strategy.evaluate({ offer: 'test', agent });
        actions.add(result.action);
      }
      
      // Should have multiple different actions
      expect(actions.size).toBeGreaterThan(1);
    });

    it('should update accept probability', () => {
      const strategy = new RandomStrategy();
      
      strategy.setAcceptProbability(0.7);
      
      expect(strategy.acceptProbability).toBe(0.7);
    });

    it('should clamp probability to valid range', () => {
      const strategy = new RandomStrategy();
      
      strategy.setAcceptProbability(1.5);
      expect(strategy.acceptProbability).toBe(1);
      
      strategy.setAcceptProbability(-0.5);
      expect(strategy.acceptProbability).toBe(0);
    });

    it('should update counter probability', () => {
      const strategy = new RandomStrategy();
      
      strategy.setCounterProbability(0.5);
      
      expect(strategy.counterProbability).toBe(0.5);
    });

    it('should return name and description', () => {
      const strategy = new RandomStrategy();
      
      expect(strategy.getName()).toBe('random');
      expect(typeof strategy.getDescription()).toBe('string');
    });
  });

  describe('StrategyFactory', () => {
    it('should create accept strategy', () => {
      const strategy = StrategyFactory.create('accept');
      
      expect(strategy).toBeInstanceOf(AcceptStrategy);
    });

    it('should create reject strategy', () => {
      const strategy = StrategyFactory.create('reject');
      
      expect(strategy).toBeInstanceOf(RejectStrategy);
    });

    it('should create counter strategy', () => {
      const strategy = StrategyFactory.create('counter');
      
      expect(strategy).toBeInstanceOf(CounterStrategy);
    });

    it('should create random strategy', () => {
      const strategy = StrategyFactory.create('random');
      
      expect(strategy).toBeInstanceOf(RandomStrategy);
    });

    it('should create counter as default', () => {
      const strategy = StrategyFactory.create('unknown');
      
      expect(strategy).toBeInstanceOf(CounterStrategy);
    });

    it('should pass options to strategy', () => {
      const strategy = StrategyFactory.create('accept', { threshold: 0.7 });
      
      expect(strategy.threshold).toBe(0.7);
    });

    it('should return available strategies', () => {
      const strategies = StrategyFactory.getAvailableStrategies();
      
      expect(strategies).toContain('accept');
      expect(strategies).toContain('reject');
      expect(strategies).toContain('counter');
      expect(strategies).toContain('random');
    });

    it('should handle case-insensitive names', () => {
      const strategy = StrategyFactory.create('ACCEPT');
      
      expect(strategy).toBeInstanceOf(AcceptStrategy);
    });
  });

  describe('similarity calculations', () => {
    it('should calculate exact match similarity', () => {
      const strategy = new AcceptStrategy();
      const agent = new Agent('agent1', { needs: ['exact match'] });
      
      const result = strategy.evaluate({
        offer: 'exact match',
        agent
      });
      
      expect(result.action).toBe('accept');
      expect(result.confidence).toBe(1);
    });

    it('should calculate contains match similarity', () => {
      const strategy = new AcceptStrategy({ threshold: 0.5 });
      const agent = new Agent('agent1', { needs: ['coffee'] });
      
      const result = strategy.evaluate({
        offer: 'coffee beans',
        agent
      });
      
      expect(result.action).toBe('accept');
      expect(result.confidence).toBe(0.8);
    });

    it('should calculate word overlap similarity', () => {
      const strategy = new AcceptStrategy({ threshold: 0.3 });
      const agent = new Agent('agent1', { needs: ['premium coffee beans'] });
      
      const result = strategy.evaluate({
        offer: 'coffee beans special',
        agent
      });
      
      // Should have some similarity due to word overlap
      expect(result.confidence).toBeGreaterThan(0);
    });
  });
});
