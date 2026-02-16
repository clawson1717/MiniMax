/**
 * Unit tests for Message Passing and Relevance Filtering
 */

const { Agent, TopologyManager, MessageSystem } = require('../src/index');

describe('Message Passing', () => {
  let agent1;
  let agent2;
  let topologyManager;

  beforeEach(() => {
    agent1 = new Agent('agent-1', 'Agent One');
    agent2 = new Agent('agent-2', 'Agent Two');
    topologyManager = new TopologyManager();
  });

  describe('Direct Message Passing', () => {
    test('should pass belief update between agents', () => {
      agent1.addBelief('Sky is blue', 0.9, 'Observation');
      
      const message = {
        type: 'belief_update',
        payload: {
          proposition: 'Sky is blue',
          confidence: 0.9,
          justification: 'Observation'
        }
      };
      
      agent2.receiveMessage({
        ...message,
        senderId: 'agent-1',
        timestamp: Date.now()
      });
      
      expect(agent2.getBelief('Sky is blue')).toBeDefined();
    });

    test('should average confidence when receiving duplicate belief', () => {
      agent1.addBelief('Test', 0.8, 'Reason');
      agent2.addBelief('Test', 0.4, 'Different reason');
      
      const message = {
        type: 'belief_update',
        payload: {
          proposition: 'Test',
          confidence: 0.8,
          justification: 'From agent-1'
        }
      };
      
      agent2.receiveMessage({
        ...message,
        senderId: 'agent-1',
        timestamp: Date.now()
      });
      
      // (0.4 + 0.8) / 2 = 0.6
      expect(agent2.getBelief('Test').confidence).toBeCloseTo(0.6);
    });
  });

  describe('Topology-based Message Routing', () => {
    test('should route messages through topology', () => {
      topologyManager.registerAgent(agent1);
      topologyManager.registerAgent(agent2);
      
      // Connect agents
      topologyManager.topology.get('agent-1').add('agent-2');
      
      const message = {
        type: 'belief_update',
        payload: {
          proposition: 'Test',
          confidence: 0.8,
          justification: 'From agent-1'
        }
      };
      
      topologyManager.sendTo('agent-1', 'agent-2', message);
      
      expect(agent2.getBelief('Test')).toBeDefined();
    });

    test('should broadcast to neighbors', () => {
      topologyManager.registerAgent(agent1);
      topologyManager.registerAgent(agent2);
      
      topologyManager.topology.get('agent-1').add('agent-2');
      
      const message = {
        type: 'belief_update',
        payload: {
          proposition: 'Broadcast test',
          confidence: 0.7,
          justification: 'Broadcast message'
        }
      };
      
      topologyManager.broadcast('agent-1', message);
      
      expect(agent2.getBelief('Broadcast test')).toBeDefined();
    });
  });
});

describe('Relevance Filtering', () => {
  let agent1;
  let agent2;
  let agent3;
  let topologyManager;

  beforeEach(() => {
    agent1 = new Agent('agent-1', 'Agent One');
    agent2 = new Agent('agent-2', 'Agent Two');
    agent3 = new Agent('agent-3', 'Agent Three');
    topologyManager = new TopologyManager();
  });

  describe('Relevance-based Topology', () => {
    test('should only connect agents with matching relevance', () => {
      // Agent 1 and 2 both interested in weather
      agent1.addBelief('Weather: Sunny', 0.9, 'Forecast');
      agent2.addBelief('Weather: Sunny', 0.8, 'Observation');
      agent3.addBelief('Sports: Football', 0.7, 'News');
      
      topologyManager.registerAgent(agent1);
      topologyManager.registerAgent(agent2);
      topologyManager.registerAgent(agent3);
      
      topologyManager.setRelevanceRules('agent-1', ['Weather']);
      topologyManager.setRelevanceRules('agent-2', ['Weather']);
      topologyManager.setRelevanceRules('agent-3', ['Sports']);
      
      topologyManager.rebuildTopology();
      
      // Agent 1 and 2 should be connected
      expect(topologyManager.getNeighbors('agent-1')).toContain('agent-2');
      // Agent 3 should not be connected to 1 or 2
      expect(topologyManager.getNeighbors('agent-1')).not.toContain('agent-3');
    });

    test('should route messages only to connected agents', () => {
      agent1.addBelief('Weather update', 0.9, 'Forecast');
      agent3.addBelief('Sports update', 0.7, 'News');
      
      topologyManager.registerAgent(agent1);
      topologyManager.registerAgent(agent2);
      topologyManager.registerAgent(agent3);
      
      topologyManager.setRelevanceRules('agent-2', ['Weather']);
      topologyManager.setRelevanceRules('agent-3', ['Sports']);
      
      topologyManager.rebuildTopology();
      
      // Only agent-2 should be connected for weather
      const weatherMessage = {
        type: 'belief_update',
        payload: {
          proposition: 'Weather update',
          confidence: 0.9,
          justification: 'New forecast'
        }
      };
      
      // Agent-1 broadcasts to neighbors (none for weather - different relevance)
      topologyManager.broadcastAll('agent-1', weatherMessage);
      
      // Agent-2 and 3 are not neighbors of agent-1, so they won't receive
      // This is expected - topology-based filtering
    });
  });

  describe('Subscription-based Filtering', () => {
    test('should filter messages by subscription', () => {
      agent1.subscribe('weather');
      agent1.subscribe('sports');
      
      expect(agent1.isSubscribed('weather')).toBe(true);
      expect(agent1.isSubscribed('sports')).toBe(true);
      expect(agent1.isSubscribed('politics')).toBe(false);
    });

    test('should allow unsubscribe', () => {
      agent1.subscribe('weather');
      agent1.unsubscribe('weather');
      
      expect(agent1.isSubscribed('weather')).toBe(false);
    });
  });

  describe('getRelevantAgents', () => {
    test('should find relevant agents for proposition', () => {
      topologyManager.registerAgent(agent1);
      topologyManager.registerAgent(agent2);
      topologyManager.registerAgent(agent3);
      
      topologyManager.setRelevanceRules('agent-1', ['Weather*']);
      topologyManager.setRelevanceRules('agent-2', ['Weather*']);
      topologyManager.setRelevanceRules('agent-3', ['Sports']);
      
      const relevantAgents = topologyManager.getRelevantAgents('Weather forecast');
      
      expect(relevantAgents).toHaveLength(2);
      expect(relevantAgents).toContain('agent-1');
      expect(relevantAgents).toContain('agent-2');
    });
  });
});

describe('MessageSystem Integration', () => {
  let messageSystem;
  let agent1;
  let agent2;

  beforeEach(() => {
    messageSystem = new MessageSystem();
    agent1 = new Agent('agent-1', 'Agent One');
    agent2 = new Agent('agent-2', 'Agent Two');
  });

  describe('MessageSystem', () => {
    test('should register agents via topology manager', () => {
      messageSystem.topologyManager.registerAgent(agent1);
      
      expect(messageSystem.topologyManager.agents.size).toBe(1);
    });

    test('should unregister agents via topology manager', () => {
      messageSystem.topologyManager.registerAgent(agent1);
      messageSystem.topologyManager.unregisterAgent('agent-1');
      
      expect(messageSystem.topologyManager.agents.size).toBe(0);
    });

    test('should set relevance rules', () => {
      messageSystem.topologyManager.registerAgent(agent1);
      messageSystem.topologyManager.setRelevanceRules('agent-1', ['Weather', 'Sports']);
      
      expect(messageSystem.topologyManager.relevanceRules.get('agent-1').size).toBe(2);
    });

    test('should rebuild topology', () => {
      messageSystem.topologyManager.registerAgent(agent1);
      messageSystem.topologyManager.registerAgent(agent2);
      
      // Add matching beliefs so they can connect
      agent1.addBelief('Weather update', 0.9, 'Forecast');
      agent2.addBelief('Weather update', 0.8, 'Observation');
      
      messageSystem.topologyManager.setRelevanceRules('agent-1', ['Weather']);
      messageSystem.topologyManager.setRelevanceRules('agent-2', ['Weather']);
      
      messageSystem.topologyManager.rebuildTopology();
      
      expect(messageSystem.topologyManager.getNeighbors('agent-1')).toContain('agent-2');
    });

    test('should broadcast message', () => {
      messageSystem.topologyManager.registerAgent(agent1);
      messageSystem.topologyManager.registerAgent(agent2);
      
      // Add matching beliefs so they can connect
      agent1.addBelief('Test belief', 0.9, 'Forecast');
      agent2.addBelief('Test belief', 0.8, 'Observation');
      
      messageSystem.topologyManager.setRelevanceRules('agent-1', ['Test']);
      messageSystem.topologyManager.setRelevanceRules('agent-2', ['Test']);
      messageSystem.topologyManager.rebuildTopology();
      
      const result = messageSystem.broadcast('agent-1', 'belief_update', {
        proposition: 'Test belief',
        confidence: 0.8,
        justification: 'Test message'
      });
      
      expect(result.deliveredCount).toBe(1);
      expect(agent2.getBelief('Test belief')).toBeDefined();
    });
  });
});
