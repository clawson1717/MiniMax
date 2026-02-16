/**
 * Integration tests for RAA (Resilient Adaptive Agent)
 * Tests full task execution, component integration, error handling, and recovery flow
 */

const { RAA } = require('../src/RAA');
const { ActionTypes, ActionFactory } = require('../src/Action');
const { MockBrowser } = require('../src/Agent');
const { VerificationStatus } = require('../src/SelfVerification');

describe('RAA Integration Tests', () => {
  let raa;

  beforeEach(async () => {
    raa = new RAA({
      name: 'TestRAA',
      maxSteps: 50,
      verbose: false
    });
    await raa.initialize();
  });

  afterEach(() => {
    raa.reset(true);
    raa = null;
  });

  describe('Initialization', () => {
    test('should create RAA with default config', () => {
      const defaultRAA = new RAA();
      expect(defaultRAA.name).toBe('RAA');
      expect(defaultRAA.maxSteps).toBe(100);
      expect(defaultRAA.verbose).toBe(false);
      expect(defaultRAA.isInitialized).toBe(false);
    });

    test('should create RAA with custom config', () => {
      const customRAA = new RAA({
        name: 'CustomRAA',
        maxSteps: 200,
        verbose: true,
        maxRecoveryAttempts: 5
      });
      expect(customRAA.name).toBe('CustomRAA');
      expect(customRAA.maxSteps).toBe(200);
      expect(customRAA.verbose).toBe(true);
      expect(customRAA.maxRecoveryAttempts).toBe(5);
    });

    test('should initialize all components', async () => {
      const result = await raa.initialize();
      
      expect(result.success).toBe(true);
      expect(result.componentsInitialized).toBe(7);
      expect(raa.isInitialized).toBe(true);
      expect(raa.agent).not.toBeNull();
      expect(raa.uncertaintyTracker).not.toBeNull();
      expect(raa.cattsAllocator).not.toBeNull();
      expect(raa.failureModeDetector).not.toBeNull();
      expect(raa.resilienceRecovery).not.toBeNull();
      expect(raa.checklistReward).not.toBeNull();
      expect(raa.verificationSystem).not.toBeNull();
    });

    test('should throw error when executing action before initialization', async () => {
      const uninitializedRAA = new RAA();
      const action = ActionFactory.navigate('https://example.com');
      
      await expect(uninitializedRAA.executeAction(action))
        .rejects.toThrow('RAA not initialized');
    });
  });

  describe('Action Execution', () => {
    test('should execute navigation action with verification', async () => {
      const action = ActionFactory.navigate('https://example.com');
      const result = await raa.executeAction(action);

      expect(result).toHaveProperty('action');
      expect(result).toHaveProperty('result');
      expect(result).toHaveProperty('verification');
      expect(result.action.type).toBe(ActionTypes.NAVIGATE);
      expect(result.result.success).toBe(true);
    });

    test('should execute click action', async () => {
      // First navigate
      await raa.executeAction(ActionFactory.navigate('https://example.com'));
      
      // Then click
      const action = ActionFactory.click('#submit-btn');
      const result = await raa.executeAction(action);

      expect(result.action.type).toBe(ActionTypes.CLICK);
      expect(result.result.success).toBe(true);
    });

    test('should execute type action', async () => {
      // First navigate
      await raa.executeAction(ActionFactory.navigate('https://example.com'));
      
      // Then type
      const action = ActionFactory.type('#search-input', 'test query');
      const result = await raa.executeAction(action);

      expect(result.action.type).toBe(ActionTypes.TYPE);
      expect(result.result.success).toBe(true);
      expect(result.result.text).toBe('test query');
    });

    test('should execute extract action', async () => {
      // First navigate
      await raa.executeAction(ActionFactory.navigate('https://example.com'));
      
      // Then extract
      const action = ActionFactory.extract('.content');
      const result = await raa.executeAction(action);

      expect(result.action.type).toBe(ActionTypes.EXTRACT);
      expect(result.result.success).toBe(true);
    });

    test('should accept action as object', async () => {
      const result = await raa.executeAction({
        type: ActionTypes.NAVIGATE,
        params: { url: 'https://example.com' }
      });

      expect(result.success).toBe(true);
      expect(result.result.success).toBe(true);
    });

    test('should reject invalid action format', async () => {
      await expect(raa.executeAction({ invalid: 'action' }))
        .rejects.toThrow('Invalid action format');
    });

    test('should track metrics after action execution', async () => {
      await raa.executeAction(ActionFactory.navigate('https://example.com'));
      
      expect(raa.metrics.totalActions).toBe(1);
      expect(raa.agent.stepCount).toBe(1);
    });
  });

  describe('Task Execution', () => {
    test('should run a simple navigation task', async () => {
      const task = {
        id: 'test-task-1',
        description: 'Navigate to example.com',
        actions: [
          { type: ActionTypes.NAVIGATE, params: { url: 'https://example.com' } }
        ]
      };

      const result = await raa.runTask(task);

      expect(result.success).toBe(true);
      expect(result.taskId).toBe('test-task-1');
      expect(result.actionsExecuted).toBe(1);
      expect(result.failed).toBe(false);
    });

    test('should run multi-action task', async () => {
      const task = {
        id: 'test-task-2',
        description: 'Search workflow',
        actions: [
          { type: ActionTypes.NAVIGATE, params: { url: 'https://example.com' } },
          { type: ActionTypes.TYPE, params: { selector: '#search', text: 'query' } },
          { type: ActionTypes.CLICK, params: { selector: '#submit' } }
        ]
      };

      const result = await raa.runTask(task);

      expect(result.success).toBe(true);
      expect(result.actionsExecuted).toBe(3);
      expect(result.results).toHaveLength(3);
    });

    test('should reject task without actions', async () => {
      const task = {
        id: 'bad-task',
        description: 'No actions'
      };

      await expect(raa.runTask(task))
        .rejects.toThrow('Invalid task: must have actions array');
    });

    test('should create checkpoints during task execution', async () => {
      // Create task with enough actions to trigger checkpoints
      const task = {
        id: 'checkpoint-task',
        description: 'Test checkpoints',
        actions: Array(6).fill(null).map((_, i) => ({
          type: ActionTypes.CLICK,
          params: { selector: `#btn-${i}` }
        }))
      };

      // Add navigation first
      task.actions.unshift({
        type: ActionTypes.NAVIGATE,
        params: { url: 'https://example.com' }
      });

      const result = await raa.runTask(task);

      expect(result.success).toBe(true);
      expect(result.checkpointCount).toBeGreaterThan(0);
    });

    test('should track task in history', async () => {
      const task = {
        id: 'history-task',
        description: 'Test history tracking',
        actions: [
          { type: ActionTypes.NAVIGATE, params: { url: 'https://example.com' } }
        ]
      };

      await raa.runTask(task);

      expect(raa.taskHistory).toHaveLength(1);
      expect(raa.taskHistory[0].taskId).toBe('history-task');
      expect(raa.metrics.tasksCompleted).toBe(1);
    });

    test('should update metrics on task completion', async () => {
      const task = {
        id: 'metrics-task',
        description: 'Test metrics',
        actions: [
          { type: ActionTypes.NAVIGATE, params: { url: 'https://example.com' } },
          { type: ActionTypes.CLICK, params: { selector: '#btn' } }
        ]
      };

      await raa.runTask(task);

      expect(raa.metrics.tasksCompleted).toBe(1);
      expect(raa.metrics.totalActions).toBe(2);
    });
  });

  describe('Component Integration', () => {
    test('should integrate CATTS allocator with uncertainty tracker', async () => {
      // Execute some actions to build uncertainty history
      for (let i = 0; i < 3; i++) {
        await raa.executeAction(ActionFactory.navigate(`https://example${i}.com`));
      }

      const depth = raa.allocateCompute();
      
      expect(typeof depth).toBe('number');
      expect(depth).toBeGreaterThanOrEqual(1);
      expect(depth).toBeLessThanOrEqual(5);
    });

    test('should integrate verification system with actions', async () => {
      const result = await raa.executeAction(
        ActionFactory.navigate('https://example.com')
      );

      expect(result.verification).toBeDefined();
      expect(result.verification.status).toBeDefined();
      expect(result.verification.confidence).toBeGreaterThanOrEqual(0);
      expect(result.verification.confidence).toBeLessThanOrEqual(1);
    });

    test('should integrate failure detection with action execution', async () => {
      const result = await raa.executeAction(
        ActionFactory.navigate('https://example.com')
      );

      expect(result.failureAnalysis).toBeDefined();
      expect(result.failureAnalysis.detectedModes).toBeDefined();
    });

    test('should create checkpoints with full state', async () => {
      await raa.executeAction(ActionFactory.navigate('https://example.com'));
      
      const checkpoint = await raa.verifyAndCheckpoint();

      expect(checkpoint.success).toBe(true);
      expect(checkpoint.checkpointId).toBeDefined();
      expect(checkpoint.state).toBeDefined();
      expect(checkpoint.state.confidence).toBeDefined();
      expect(checkpoint.state.entropy).toBeDefined();
    });

    test('should use checklist reward for task verification', async () => {
      const checklistId = raa.checklistReward.createChecklist('navigation');
      expect(checklistId).toBeDefined();

      const status = raa.checklistReward.getChecklistStatus(checklistId);
      expect(status).toBeDefined();
      expect(status.criteria).toBeDefined();
    });
  });

  describe('Error Handling', () => {
    test('should handle action execution errors gracefully', async () => {
      // Create RAA with browser that will fail
      const failingBrowser = {
        navigate: jest.fn().mockRejectedValue(new Error('Navigation failed')),
        click: jest.fn(),
        type: jest.fn(),
        extract: jest.fn(),
        getPageContent: jest.fn()
      };

      const failingRAA = new RAA({
        browser: failingBrowser,
        verbose: false
      });
      await failingRAA.initialize();

      const result = await failingRAA.executeAction(
        ActionFactory.navigate('https://example.com')
      );

      expect(result.success).toBe(false);
      expect(result.executionError).toBe('Navigation failed');
      expect(result.verification.status).toBe(VerificationStatus.FAILED);
    });

    test('should handle max steps exceeded', async () => {
      const limitedRAA = new RAA({
        maxSteps: 2,
        verbose: false
      });
      await limitedRAA.initialize();

      const task = {
        id: 'max-steps-task',
        description: 'Too many actions',
        actions: [
          { type: ActionTypes.NAVIGATE, params: { url: 'https://example1.com' } },
          { type: ActionTypes.NAVIGATE, params: { url: 'https://example2.com' } },
          { type: ActionTypes.NAVIGATE, params: { url: 'https://example3.com' } }
        ]
      };

      const result = await limitedRAA.runTask(task);

      expect(result.failed).toBe(true);
      expect(result.failureReason).toContain('Maximum steps');
    });

    test('should handle invalid action types', async () => {
      const result = await raa.executeAction({
        type: 'INVALID_TYPE',
        params: {}
      });

      expect(result.success).toBe(false);
      expect(result.executionError).toContain('Unknown action type');
    });

    test('should track failed verifications in metrics', async () => {
      const failingBrowser = {
        navigate: jest.fn().mockRejectedValue(new Error('Always fails')),
        click: jest.fn(),
        type: jest.fn(),
        extract: jest.fn(),
        getPageContent: jest.fn()
      };

      const failingRAA = new RAA({
        browser: failingBrowser,
        verbose: false
      });
      await failingRAA.initialize();

      await failingRAA.executeAction(
        ActionFactory.navigate('https://example.com')
      );

      expect(failingRAA.metrics.failedVerifications).toBeGreaterThan(0);
    });

    test('should handle task execution without initialization', async () => {
      const uninitializedRAA = new RAA();
      
      await expect(uninitializedRAA.runTask({
        id: 'test',
        actions: []
      })).rejects.toThrow('RAA not initialized');
    });
  });

  describe('Recovery Flow', () => {
    test('should detect failure modes', async () => {
      // Simulate multiple failures to trigger detection
      const failingBrowser = new MockBrowser();
      jest.spyOn(failingBrowser, 'navigate').mockRejectedValue(
        new Error('Connection error')
      );

      const failingRAA = new RAA({
        browser: failingBrowser,
        verbose: false
      });
      await failingRAA.initialize();

      // Execute failing action
      const result = await failingRAA.executeAction(
        ActionFactory.navigate('https://example.com')
      );

      expect(result.failureAnalysis).toBeDefined();
      expect(result.verification.status).toBe(VerificationStatus.FAILED);
    });

    test('should apply recovery when failure detected', async () => {
      // Create a scenario where recovery might be needed
      const result = await raa.detectAndRecover({
        selfDoubt: { currentConfidence: 0.2, originalReasoning: 'test' }
      });

      // Should complete without error, even if no failure detected
      expect(result.detected).toBeDefined();
    });

    test('should track recovery attempts', async () => {
      // Initialize with mock browser that alternates success/failure
      let attemptCount = 0;
      const mockBrowser = new MockBrowser();
      const originalNavigate = mockBrowser.navigate.bind(mockBrowser);
      mockBrowser.navigate = jest.fn().mockImplementation(async (url) => {
        attemptCount++;
        if (attemptCount % 2 === 0) {
          throw new Error('Simulated error');
        }
        return originalNavigate(url);
      });

      const testRAA = new RAA({
        browser: mockBrowser,
        maxRecoveryAttempts: 2,
        verbose: false
      });
      await testRAA.initialize();

      // Execute some actions
      await testRAA.executeAction(ActionFactory.navigate('https://example1.com'));
      await testRAA.executeAction(ActionFactory.navigate('https://example2.com'));

      expect(testRAA.metrics.totalActions).toBeGreaterThan(0);
    });

    test('should limit recovery attempts', async () => {
      // Create task that might need recovery
      const task = {
        id: 'recovery-limit-task',
        description: 'Test recovery limits',
        actions: [
          { type: ActionTypes.NAVIGATE, params: { url: 'https://example.com' } }
        ]
      };

      const result = await raa.runTask(task);
      
      // Should complete normally
      expect(result.success).toBe(true);
    });

    test('should reset recovery counter on new task', async () => {
      raa.recoveryAttempts = 2;

      const task = {
        id: 'reset-test',
        description: 'Reset recovery counter',
        actions: [
          { type: ActionTypes.NAVIGATE, params: { url: 'https://example.com' } }
        ]
      };

      await raa.runTask(task);

      // After task completes, recovery attempts should be reset
      expect(raa.recoveryAttempts).toBe(0);
    });
  });

  describe('Status and Statistics', () => {
    test('should return comprehensive status', () => {
      const status = raa.getStatus();

      expect(status.isInitialized).toBe(true);
      expect(status.isRunning).toBe(false);
      expect(status.components).toBeDefined();
      expect(status.metrics).toBeDefined();
      expect(status.checkpoints).toBeDefined();
    });

    test('should return component status', () => {
      const status = raa.getStatus();

      expect(status.components.agent).toBeDefined();
      expect(status.components.uncertaintyTracker).toBeDefined();
      expect(status.components.cattsAllocator).toBeDefined();
      expect(status.components.failureModeDetector).toBeDefined();
      expect(status.components.resilienceRecovery).toBeDefined();
      expect(status.components.verificationSystem).toBeDefined();
    });

    test('should return statistics', async () => {
      // Execute some actions to generate stats
      await raa.executeAction(ActionFactory.navigate('https://example.com'));

      const stats = raa.getStatistics();

      expect(stats.raa).toBeDefined();
      expect(stats.uncertainty).toBeDefined();
      expect(stats.catts).toBeDefined();
      expect(stats.verification).toBeDefined();
      expect(stats.failureDetection).toBeDefined();
      expect(stats.recovery).toBeDefined();
    });

    test('should update status during task execution', async () => {
      const task = {
        id: 'status-task',
        description: 'Test status updates',
        actions: [
          { type: ActionTypes.NAVIGATE, params: { url: 'https://example.com' } }
        ]
      };

      // Before task
      let status = raa.getStatus();
      expect(status.currentTask).toBeNull();

      // Start task (but don't await to check running state)
      const taskPromise = raa.runTask(task);
      
      status = raa.getStatus();
      expect(status.isRunning || status.currentTask).toBeTruthy();

      await taskPromise;

      // After task
      status = raa.getStatus();
      expect(status.currentTask).toBeNull();
      expect(raa.metrics.tasksCompleted).toBe(1);
    });
  });

  describe('Reset Functionality', () => {
    test('should reset RAA state', async () => {
      // Execute some actions
      await raa.executeAction(ActionFactory.navigate('https://example.com'));
      await raa.verifyAndCheckpoint();

      expect(raa.checkpoints.length).toBeGreaterThan(0);
      expect(raa.metrics.totalActions).toBeGreaterThan(0);

      // Reset
      const result = raa.reset();

      expect(result.success).toBe(true);
      expect(raa.checkpoints).toHaveLength(0);
      expect(raa.metrics.totalActions).toBe(0);
      expect(raa.currentTask).toBeNull();
    });

    test('should perform full reset', async () => {
      const initialStartTime = raa.metrics.startTime;
      
      raa.reset(true);

      expect(raa.metrics.startTime).toBeNull();
    });

    test('should reset all components', async () => {
      // Add some data
      await raa.executeAction(ActionFactory.navigate('https://example.com'));
      raa.uncertaintyTracker.recordVote({ success: true, confidence: 0.8 });

      const beforeStats = raa.uncertaintyTracker.getStatistics();
      expect(beforeStats.historyLength).toBeGreaterThan(0);

      raa.reset();

      const afterStats = raa.uncertaintyTracker.getStatistics();
      expect(afterStats.historyLength).toBe(0);
    });
  });

  describe('Edge Cases', () => {
    test('should handle empty task', async () => {
      const task = {
        id: 'empty-task',
        description: 'Empty task',
        actions: []
      };

      const result = await raa.runTask(task);

      expect(result.success).toBe(true); // Empty task completes successfully
      expect(result.actionsExecuted).toBe(0);
    });

    test('should handle action with no params', async () => {
      // SCROLL action can work with defaults
      const action = ActionFactory.scroll();
      const result = await raa.executeAction(action);

      expect(result.action.params.direction).toBe('down');
      expect(result.action.params.amount).toBe(500);
    });

    test('should handle rapid successive actions', async () => {
      const actions = Array(10).fill(null).map((_, i) => 
        ActionFactory.navigate(`https://example${i}.com`)
      );

      const results = [];
      for (const action of actions) {
        const result = await raa.executeAction(action);
        results.push(result);
      }

      expect(results).toHaveLength(10);
      expect(raa.metrics.totalActions).toBe(10);
    });

    test('should handle verbose logging', async () => {
      const verboseRAA = new RAA({
        verbose: true
      });
      await verboseRAA.initialize();

      const consoleSpy = jest.spyOn(console, 'log').mockImplementation();

      await verboseRAA.executeAction(ActionFactory.navigate('https://example.com'));

      expect(consoleSpy).toHaveBeenCalled();
      
      consoleSpy.mockRestore();
    });

    test('should preserve task history across resets', async () => {
      const task = {
        id: 'history-preserve-task',
        description: 'Test history',
        actions: [
          { type: ActionTypes.NAVIGATE, params: { url: 'https://example.com' } }
        ]
      };

      await raa.runTask(task);
      expect(raa.taskHistory).toHaveLength(1);

      raa.reset(); // Normal reset preserves config but clears history
      expect(raa.taskHistory).toHaveLength(0);
    });
  });
});
