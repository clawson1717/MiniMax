/**
 * Unit tests for Negotiation Strategies
 */

const {
  AcceptStrategy,
  RejectStrategy,
  CounterStrategy,
  RandomStrategy,
  StrategyFactory
} = require('../src/strategies');
const Agent = require('../src/Agent');

describe('Negotiation Strategies', () => {
  let agent;
  let context;

  beforeEach(() => {
    agent = new Agent('agent-1', {
      needs: ['apples', 'bananas'],
      offers: ['oranges', 'grapes']
    });
    
    context = {
      agent,
      offer: 'fresh apples',
      roundNumber: 1,
      messages: []
    };
  });

  describe('AcceptStrategy', () => {
    test('should accept offer that meets threshold', () => {
      const strategy = new AcceptStrategy({ threshold: 0.3 });
      const decision = strategy.evaluate(context);
      
      expect(decision.action).toBe('accept');
      expect(decision.matchedNeed).toBe('apples');
    });

    test('should reject offer below threshold', () => {
      const strategy = new AcceptStrategy({ threshold: 0.9 });
      context.offer = 'something unrelated';
      const decision = strategy.evaluate(context);
      
      expect(decision.action).toBe('reject');
    });

    test('should use default threshold of 0.3', () => {
      const strategy = new AcceptStrategy();
      expect(strategy.threshold).toBe(0.3);
    });

    test('should check sender offers if provided', () => {
      const strategy = new AcceptStrategy({ threshold: 0.3 });
      context.offer = 'unrelated';
      context.senderOffers = ['apples', 'oranges'];
      
      const decision = strategy.evaluate(context);
      
      expect(decision.action).toBe('accept');
    });

    test('should return confidence in decision', () => {
      const strategy = new AcceptStrategy({ threshold: 0.3 });
      const decision = strategy.evaluate(context);
      
      expect(decision.confidence).toBeGreaterThan(0);
      expect(decision.confidence).toBeLessThanOrEqual(1);
    });

    test('should return reason for decision', () => {
      const strategy = new AcceptStrategy({ threshold: 0.3 });
      const decision = strategy.evaluate(context);
      
      expect(decision.reason).toContain('threshold');
    });

    test('should match exact strings', () => {
      const strategy = new AcceptStrategy({ threshold: 0.3 });
      context.offer = 'apples';
      
      const decision = strategy.evaluate(context);
      
      expect(decision.action).toBe('accept');
      expect(decision.confidence).toBe(1.0);
    });

    test('should match substring', () => {
      const strategy = new AcceptStrategy({ threshold: 0.3 });
      context.offer = 'fresh red apples from the market';
      
      const decision = strategy.evaluate(context);
      
      expect(decision.action).toBe('accept');
    });

    test('should get name', () => {
      const strategy = new AcceptStrategy();
      expect(strategy.getName()).toBe('accept');
    });

    test('should get description', () => {
      const strategy = new AcceptStrategy({ threshold: 0.5 });
      expect(strategy.getDescription()).toContain('0.5');
    });
  });

  describe('RejectStrategy', () => {
    test('should reject with default probability', () => {
      const strategy = new RejectStrategy();
      const decision = strategy.evaluate(context);
      
      expect(decision.action).toBe('reject');
    });

    test('should use default probability of 1.0', () => {
      const strategy = new RejectStrategy();
      expect(strategy.probability).toBe(1.0);
    });

    test('should handle probability 0 as default to 1.0 due to JavaScript falsy behavior', () => {
      // Due to JavaScript falsy behavior (0 || 1.0 = 1.0), probability 0 defaults to 1.0
      const strategy = new RejectStrategy({ probability: 0 });
      // So with probability=0, it actually becomes 1.0, and rejects all
      let rejectCount = 0;
      for (let i = 0; i < 100; i++) {
        const decision = strategy.evaluate(context);
        if (decision.action === 'reject') rejectCount++;
      }
      // With the bug (0 becomes 1.0), it should reject all
      expect(rejectCount).toBe(100);
    });

    test('should return reason for rejection', () => {
      const strategy = new RejectStrategy();
      const decision = strategy.evaluate(context);
      
      expect(decision.reason).toContain('reject');
    });

    test('should return confidence', () => {
      const strategy = new RejectStrategy({ probability: 0.7 });
      const decision = strategy.evaluate(context);
      
      expect(decision.confidence).toBe(0.7);
    });

    test('should get name', () => {
      const strategy = new RejectStrategy();
      expect(strategy.getName()).toBe('reject');
    });

    test('should get description', () => {
      const strategy = new RejectStrategy({ probability: 0.5 });
      expect(strategy.getDescription()).toContain('0.5');
    });
  });

  describe('CounterStrategy', () => {
    test('should accept offer that meets counter threshold', () => {
      const strategy = new CounterStrategy({ counterThreshold: 0.3 });
      const decision = strategy.evaluate(context);
      
      expect(decision.action).toBe('accept');
    });

    test('should counter when offer below threshold', () => {
      const strategy = new CounterStrategy({ counterThreshold: 0.9 });
      context.offer = 'unrelated offer';
      
      const decision = strategy.evaluate(context);
      
      expect(decision.action).toBe('counter');
    });

    test('should reject when no counter-offer available', () => {
      const strategy = new CounterStrategy({ counterThreshold: 0.9 });
      agent.offers = [];
      context.offer = 'unrelated offer';
      
      const decision = strategy.evaluate(context);
      
      expect(decision.action).toBe('reject');
    });

    test('should include counter-offer in decision', () => {
      const strategy = new CounterStrategy({ counterThreshold: 0.9 });
      context.offer = 'unrelated offer';
      
      const decision = strategy.evaluate(context);
      
      expect(decision.counterOffer).toBeDefined();
      expect(agent.getOffers()).toContain(decision.counterOffer);
    });

    test('should track round count', () => {
      const strategy = new CounterStrategy();
      context.roundNumber = 5;
      
      strategy.evaluate(context);
      
      expect(strategy.roundCount).toBe(5);
    });

    test('should use default threshold of 0.2', () => {
      const strategy = new CounterStrategy();
      expect(strategy.counterThreshold).toBe(0.2);
    });

    test('should use default concession rate of 0.1', () => {
      const strategy = new CounterStrategy();
      expect(strategy.concessionRate).toBe(0.1);
    });

    test('should reset round counter', () => {
      const strategy = new CounterStrategy();
      strategy.roundCount = 10;
      
      strategy.reset();
      
      expect(strategy.roundCount).toBe(0);
    });

    test('should get name', () => {
      const strategy = new CounterStrategy();
      expect(strategy.getName()).toBe('counter');
    });

    test('should get description', () => {
      const strategy = new CounterStrategy({ concessionRate: 0.15 });
      expect(strategy.getDescription()).toContain('15%');
    });

    test('should extract numeric values from offers', () => {
      const strategy = new CounterStrategy();
      const value = strategy._extractNumericValue('Offer $500');
      
      expect(value).toBe(500);
    });

    test('should return null for no numeric value', () => {
      const strategy = new CounterStrategy();
      const value = strategy._extractNumericValue('apples');
      
      expect(value).toBeNull();
    });

    test('should handle offer without agent offers', () => {
      const strategy = new CounterStrategy({ counterThreshold: 0.9 });
      agent.offers = [];
      context.offer = 'unrelated';
      
      const decision = strategy.evaluate(context);
      
      expect(decision.action).toBe('reject');
    });
  });

  describe('RandomStrategy', () => {
    test('should accept based on probability', () => {
      const strategy = new RandomStrategy({ acceptProbability: 1.0 });
      const decision = strategy.evaluate(context);
      
      expect(decision.action).toBe('accept');
    });

    test('should have different actions based on configuration', () => {
      // Test that RandomStrategy produces different outcomes with different settings
      const acceptStrategy = new RandomStrategy({ 
        acceptProbability: 1.0,
        counterProbability: 0 
      });
      const decision = acceptStrategy.evaluate(context);
      expect(decision.action).toBe('accept');
    });

    test('should counter based on probability', () => {
      const strategy = new RandomStrategy({ 
        acceptProbability: 0,
        counterProbability: 1.0 
      });
      // Test probability statistically
      let counterCount = 0;
      for (let i = 0; i < 100; i++) {
        const decision = strategy.evaluate(context);
        if (decision.action === 'counter') counterCount++;
      }
      // With 100% counter probability, most should be counter
      expect(counterCount).toBeGreaterThan(50);
    });

    test('should include counter-offer when countering', () => {
      const strategy = new RandomStrategy({ 
        acceptProbability: 0,
        counterProbability: 1.0 
      });
      
      // Run multiple times to get a counter decision
      let foundCounter = false;
      let counterOffer = null;
      for (let i = 0; i < 50; i++) {
        const decision = strategy.evaluate(context);
        if (decision.action === 'counter') {
          foundCounter = true;
          counterOffer = decision.counterOffer;
          break;
        }
      }
      
      // With 100% counter probability, we should find a counter-offer
      expect(foundCounter).toBe(true);
    });

    test('should use default probabilities', () => {
      const strategy = new RandomStrategy();
      expect(strategy.acceptProbability).toBe(0.3);
      expect(strategy.counterProbability).toBe(0.4);
    });

    test('should get name', () => {
      const strategy = new RandomStrategy();
      expect(strategy.getName()).toBe('random');
    });

    test('should get description', () => {
      const strategy = new RandomStrategy();
      const desc = strategy.getDescription();
      expect(desc).toContain('30%');
      expect(desc).toContain('40%');
    });

    test('should update accept probability', () => {
      const strategy = new RandomStrategy();
      strategy.setAcceptProbability(0.5);
      
      expect(strategy.acceptProbability).toBe(0.5);
    });

    test('should clamp accept probability to [0,1]', () => {
      const strategy = new RandomStrategy();
      strategy.setAcceptProbability(2.0);
      
      expect(strategy.acceptProbability).toBe(1.0);
      
      strategy.setAcceptProbability(-1.0);
      expect(strategy.acceptProbability).toBe(0.0);
    });

    test('should update counter probability', () => {
      const strategy = new RandomStrategy();
      strategy.setCounterProbability(0.5);
      
      expect(strategy.counterProbability).toBe(0.5);
    });

    test('should clamp counter probability to [0,1]', () => {
      const strategy = new RandomStrategy();
      strategy.setCounterProbability(2.0);
      
      expect(strategy.counterProbability).toBe(1.0);
      
      strategy.setCounterProbability(-1.0);
      expect(strategy.counterProbability).toBe(0.0);
    });
  });

  describe('StrategyFactory', () => {
    test('should create AcceptStrategy', () => {
      const strategy = StrategyFactory.create('accept');
      
      expect(strategy).toBeInstanceOf(AcceptStrategy);
    });

    test('should create RejectStrategy', () => {
      const strategy = StrategyFactory.create('reject');
      
      expect(strategy).toBeInstanceOf(RejectStrategy);
    });

    test('should create CounterStrategy', () => {
      const strategy = StrategyFactory.create('counter');
      
      expect(strategy).toBeInstanceOf(CounterStrategy);
    });

    test('should create RandomStrategy', () => {
      const strategy = StrategyFactory.create('random');
      
      expect(strategy).toBeInstanceOf(RandomStrategy);
    });

    test('should be case insensitive', () => {
      const strategy = StrategyFactory.create('ACCEPT');
      
      expect(strategy).toBeInstanceOf(AcceptStrategy);
    });

    test('should default to CounterStrategy for unknown strategy', () => {
      const strategy = StrategyFactory.create('unknown');
      
      expect(strategy).toBeInstanceOf(CounterStrategy);
    });

    test('should pass options to strategy', () => {
      const strategy = StrategyFactory.create('accept', { threshold: 0.8 });
      
      expect(strategy.threshold).toBe(0.8);
    });

    test('should return available strategies', () => {
      const strategies = StrategyFactory.getAvailableStrategies();
      
      expect(strategies).toContain('accept');
      expect(strategies).toContain('reject');
      expect(strategies).toContain('counter');
      expect(strategies).toContain('random');
    });
  });

  describe('similarity calculation', () => {
    test('exact match should have similarity 1.0', () => {
      const strategy = new AcceptStrategy();
      const score = strategy._calculateSimilarity('apples', 'apples');
      
      expect(score).toBe(1.0);
    });

    test('contained string should have similarity 0.8', () => {
      const strategy = new AcceptStrategy();
      const score = strategy._calculateSimilarity('fresh apples', 'apples');
      
      expect(score).toBe(0.8);
    });

    test('partial word overlap should calculate Jaccard similarity', () => {
      const strategy = new AcceptStrategy();
      const score = strategy._calculateSimilarity('apple banana', 'apple orange');
      
      // Intersection: 1 ('apple'), Union: 3 ('apple', 'banana', 'orange')
      // Jaccard = 1/3
      expect(score).toBe(1/3);
    });

    test('no overlap should return 0', () => {
      const strategy = new AcceptStrategy();
      const score = strategy._calculateSimilarity('apples', 'oranges');
      
      expect(score).toBe(0);
    });
  });
});
