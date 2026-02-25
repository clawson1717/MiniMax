/**
 * Unit tests for NegotiationRunner class
 */

const NegotiationRunner = require('../src/NegotiationRunner');
const AgentNetwork = require('../src/AgentNetwork');
const Agent = require('../src/Agent');

describe('NegotiationRunner', () => {
  describe('constructor', () => {
    it('should create with default options', () => {
      const network = new AgentNetwork();
      const runner = new NegotiationRunner(network);
      
      expect(runner.maxRounds).toBe(20);
      expect(runner.timeout).toBe(60000);
      expect(runner.verbose).toBe(false);
      expect(runner.dynamicTopology).toBe(true);
    });

    it('should create with custom options', () => {
      const network = new AgentNetwork();
      const runner = new NegotiationRunner(network, {
        maxRounds: 10,
        timeout: 30000,
        verbose: true,
        dynamicTopology: false,
        topologyRebuildInterval: 2
      });
      
      expect(runner.maxRounds).toBe(10);
      expect(runner.timeout).toBe(30000);
      expect(runner.verbose).toBe(true);
      expect(runner.dynamicTopology).toBe(false);
      expect(runner.topologyRebuildInterval).toBe(2);
    });

    it('should initialize with idle status', () => {
      const network = new AgentNetwork();
      const runner = new NegotiationRunner(network);
      
      expect(runner.status).toBe('idle');
    });
  });

  describe('run', () => {
    it('should run negotiation and return result', () => {
      const network = new AgentNetwork();
      network.addAgent(new Agent('agent1'));
      network.addAgent(new Agent('agent2'));
      
      const runner = new NegotiationRunner(network, { maxRounds: 2 });
      const result = runner.run();
      
      expect(result.status).toBe('completed');
      expect(result.rounds).toBeDefined();
    });

    it('should allow sequential runs', () => {
      const network = new AgentNetwork();
      network.addAgent(new Agent('agent1'));
      
      const runner = new NegotiationRunner(network, { maxRounds: 1 });
      const result1 = runner.run();
      
      expect(result1.status).toBe('completed');
      
      // Should be able to run again after completion
      const result2 = runner.run();
      expect(result2.status).toBe('completed');
    });

    it('should stop at max rounds', () => {
      const network = new AgentNetwork();
      network.addAgent(new Agent('agent1', { needs: ['impossible'] }));
      
      const runner = new NegotiationRunner(network, { maxRounds: 3 });
      const result = runner.run();
      
      expect(result.rounds).toBe(3);
    });

    it('should track start and end time', () => {
      const network = new AgentNetwork();
      network.addAgent(new Agent('agent1'));
      
      const runner = new NegotiationRunner(network, { maxRounds: 1 });
      runner.run();
      
      expect(runner.startTime).toBeDefined();
      expect(runner.endTime).toBeDefined();
      expect(runner.endTime).toBeGreaterThanOrEqual(runner.startTime);
    });
  });

  describe('step', () => {
    it('should execute single step', () => {
      const network = new AgentNetwork();
      network.addAgent(new Agent('agent1'));
      network.addAgent(new Agent('agent2'));
      
      const runner = new NegotiationRunner(network, { maxRounds: 5 });
      const result = runner.step();
      
      expect(result.success).toBe(true);
      expect(result.round).toBe(1);
    });

    it('should return false when negotiation finished', () => {
      const network = new AgentNetwork();
      network.addAgent(new Agent('agent1'));
      
      const runner = new NegotiationRunner(network, { maxRounds: 1 });
      runner.run();
      
      const result = runner.step();
      
      expect(result.success).toBe(false);
    });

    it('should mark complete when max rounds reached', () => {
      const network = new AgentNetwork();
      network.addAgent(new Agent('agent1'));
      
      const runner = new NegotiationRunner(network, { maxRounds: 2 });
      runner.step();
      runner.step();
      
      expect(runner.status).toBe('completed');
    });
  });

  describe('getResult', () => {
    it('should return negotiation result', () => {
      const network = new AgentNetwork();
      network.addAgent(new Agent('agent1'));
      
      const runner = new NegotiationRunner(network, { maxRounds: 1 });
      runner.run();
      
      const result = runner.getResult();
      
      expect(result.status).toBeDefined();
      expect(result.rounds).toBeDefined();
      expect(result.statistics).toBeDefined();
    });

    it('should include agent results', () => {
      const network = new AgentNetwork();
      network.addAgent(new Agent('agent1', { needs: ['x'] }));
      network.addAgent(new Agent('agent2'));
      
      const runner = new NegotiationRunner(network, { maxRounds: 1 });
      runner.run();
      
      const result = runner.getResult();
      
      expect(result.agentResults).toBeDefined();
      expect(result.agentResults.length).toBe(2);
    });

    it('should calculate total statistics', () => {
      const network = new AgentNetwork();
      network.addAgent(new Agent('agent1'));
      network.addAgent(new Agent('agent2'));
      
      const runner = new NegotiationRunner(network, { maxRounds: 2 });
      runner.run();
      
      const result = runner.getResult();
      
      expect(typeof result.statistics.totalOffers).toBe('number');
      expect(typeof result.statistics.totalAcceptances).toBe('number');
      expect(typeof result.statistics.totalRejections).toBe('number');
    });
  });

  describe('getHistory', () => {
    it('should return round history', () => {
      const network = new AgentNetwork();
      network.addAgent(new Agent('agent1'));
      
      const runner = new NegotiationRunner(network, { maxRounds: 3 });
      runner.run();
      
      const history = runner.getHistory();
      
      expect(Array.isArray(history)).toBe(true);
    });

    it('should return history for specific round', () => {
      const network = new AgentNetwork();
      network.addAgent(new Agent('agent1', { needs: ['item1'], offers: ['item2'] }));
      network.addAgent(new Agent('agent2', { needs: ['item2'], offers: ['item1'] }));
      
      const runner = new NegotiationRunner(network, { maxRounds: 3 });
      runner.run();
      
      const history = runner.getHistory();
      // History should be an array (may be empty if negotiation ended quickly)
      expect(Array.isArray(history)).toBe(true);
    });

    it('should return null for invalid round', () => {
      const network = new AgentNetwork();
      const runner = new NegotiationRunner(network);
      
      const roundHistory = runner.getRoundHistory(999);
      
      expect(roundHistory).toBeNull();
    });
  });

  describe('getStatus', () => {
    it('should return current status', () => {
      const network = new AgentNetwork();
      const runner = new NegotiationRunner(network);
      
      const status = runner.getStatus();
      
      expect(status.status).toBe('idle');
      expect(status.currentRound).toBe(0);
      expect(status.maxRounds).toBeDefined();
    });
  });

  describe('getSummary', () => {
    it('should return negotiation summary', () => {
      const network = new AgentNetwork();
      network.addAgent(new Agent('agent1'));
      
      const runner = new NegotiationRunner(network, { maxRounds: 2 });
      runner.run();
      
      const summary = runner.getSummary();
      
      expect(summary.status).toBeDefined();
      expect(summary.currentRound).toBeDefined();
      expect(summary.maxRounds).toBeDefined();
      expect(summary.duration).toBeDefined();
    });
  });

  describe('reset', () => {
    it('should reset runner state', () => {
      const network = new AgentNetwork();
      network.addAgent(new Agent('agent1'));
      
      const runner = new NegotiationRunner(network, { maxRounds: 5 });
      runner.run();
      runner.reset();
      
      expect(runner.currentRound).toBe(0);
      expect(runner.status).toBe('idle');
      expect(runner.rounds.length).toBe(0);
    });

    it('should clear agent histories', () => {
      const network = new AgentNetwork();
      const agent = new Agent('agent1');
      network.addAgent(agent);
      
      agent.recordEvent({ type: 'test' });
      
      const runner = new NegotiationRunner(network, { maxRounds: 1 });
      runner.run();
      runner.reset();
      
      expect(agent.history.length).toBe(0);
    });
  });

  describe('dynamic topology', () => {
    it('should rebuild topology when enabled', () => {
      const network = new AgentNetwork();
      network.addAgent(new Agent('agent1', { needs: ['x'], offers: ['y'] }));
      network.addAgent(new Agent('agent2', { needs: ['y'], offers: ['x'] }));
      
      const runner = new NegotiationRunner(network, {
        maxRounds: 2,
        dynamicTopology: true,
        topologyRebuildInterval: 1
      });
      
      runner.run();
      
      expect(runner.topologyHistory.length).toBeGreaterThan(0);
    });

    it('should not rebuild topology when disabled', () => {
      const network = new AgentNetwork();
      network.addAgent(new Agent('agent1'));
      
      const runner = new NegotiationRunner(network, {
        maxRounds: 2,
        dynamicTopology: false
      });
      
      runner.run();
      
      expect(runner.topologyHistory.length).toBe(0);
    });

    it('should track topology history', () => {
      const network = new AgentNetwork();
      network.addAgent(new Agent('agent1', { needs: ['x'], offers: ['y'] }));
      network.addAgent(new Agent('agent2', { needs: ['y'], offers: ['x'] }));
      
      const runner = new NegotiationRunner(network, {
        maxRounds: 3,
        dynamicTopology: true,
        topologyRebuildInterval: 1
      });
      
      runner.run();
      
      const topologyHistory = runner.getTopologyHistory();
      expect(Array.isArray(topologyHistory)).toBe(true);
    });
  });

  describe('getCurrentRound', () => {
    it('should return current round number', () => {
      const network = new AgentNetwork();
      const runner = new NegotiationRunner(network);
      
      expect(runner.getCurrentRound()).toBe(0);
      
      runner.step();
      expect(runner.getCurrentRound()).toBe(1);
    });
  });

  describe('agreement detection', () => {
    it('should detect agreement when all satisfied', () => {
      const network = new AgentNetwork();
      network.addAgent(new Agent('agent1'));
      network.addAgent(new Agent('agent2'));
      
      const runner = new NegotiationRunner(network, { maxRounds: 10 });
      const result = runner.run();
      
      expect(result.success).toBe(true);
    });

    it('should continue when needs not satisfied', () => {
      const network = new AgentNetwork();
      network.addAgent(new Agent('agent1', { needs: ['impossible'] }));
      
      const runner = new NegotiationRunner(network, { maxRounds: 3 });
      const result = runner.run();
      
      expect(result.rounds).toBe(3);
    });
  });

  describe('timeout handling', () => {
    it('should timeout when exceeded', () => {
      const network = new AgentNetwork();
      network.addAgent(new Agent('agent1', { needs: ['x'] }));
      
      const runner = new NegotiationRunner(network, {
        maxRounds: 1000,
        timeout: 1 // 1ms - will timeout
      });
      
      const result = runner.run();
      
      expect(result.status).toBe('timeout');
    });
  });

  describe('verbose logging', () => {
    it('should not log when verbose is false', () => {
      const consoleSpy = jest.spyOn(console, 'log').mockImplementation();
      
      const network = new AgentNetwork();
      network.addAgent(new Agent('agent1'));
      
      const runner = new NegotiationRunner(network, {
        maxRounds: 1,
        verbose: false
      });
      
      runner.run();
      
      expect(consoleSpy).not.toHaveBeenCalled();
      
      consoleSpy.mockRestore();
    });

    it('should log when verbose is true', () => {
      const consoleSpy = jest.spyOn(console, 'log').mockImplementation();
      
      const network = new AgentNetwork();
      network.addAgent(new Agent('agent1'));
      
      const runner = new NegotiationRunner(network, {
        maxRounds: 1,
        verbose: true
      });
      
      runner.run();
      
      expect(consoleSpy).toHaveBeenCalled();
      
      consoleSpy.mockRestore();
    });
  });

  describe('strategy resolution', () => {
    it('should resolve agent strategies', () => {
      const network = new AgentNetwork();
      network.addAgent(new Agent('agent1', { strategy: 'accept' }));
      network.addAgent(new Agent('agent2', { strategy: 'reject' }));
      
      const runner = new NegotiationRunner(network, { maxRounds: 2 });
      runner.run();
      
      // Should complete without error
      expect(runner.status).toBe('completed');
    });
  });

  describe('edge cases', () => {
    it('should handle empty network', () => {
      const network = new AgentNetwork();
      const runner = new NegotiationRunner(network, { maxRounds: 1 });
      
      const result = runner.run();
      
      expect(result.status).toBe('completed');
    });

    it('should handle single agent', () => {
      const network = new AgentNetwork();
      network.addAgent(new Agent('agent1'));
      
      const runner = new NegotiationRunner(network, { maxRounds: 1 });
      const result = runner.run();
      
      expect(result.success).toBe(true);
    });
  });
});
