/**
 * ResilienceRecoverySystem Tests
 * Comprehensive unit tests for resilience recovery strategies
 */

const { ResilienceRecoverySystem } = require('../src/ResilienceRecovery');

describe('ResilienceRecoverySystem', () => {
  let recovery;

  beforeEach(() => {
    recovery = new ResilienceRecoverySystem();
  });

  describe('Constructor', () => {
    test('should create recovery system with default config', () => {
      expect(recovery.confidenceBoostAmount).toBe(0.3);
      expect(recovery.maxRecoveryAttempts).toBe(3);
      expect(recovery.recoveryCooldownMs).toBe(5000);
      expect(recovery.trackEffectiveness).toBe(true);
      expect(recovery.summaryTurnThreshold).toBe(10);
    });

    test('should create recovery system with custom config', () => {
      const customRecovery = new ResilienceRecoverySystem({
        confidenceBoostAmount: 0.5,
        maxRecoveryAttempts: 5,
        recoveryCooldownMs: 10000
      });
      
      expect(customRecovery.confidenceBoostAmount).toBe(0.5);
      expect(customRecovery.maxRecoveryAttempts).toBe(5);
      expect(customRecovery.recoveryCooldownMs).toBe(10000);
      expect(customRecovery.trackEffectiveness).toBe(true);
    });

    test('should initialize with empty history', () => {
      expect(recovery.recoveryHistory).toEqual([]);
      expect(recovery.getRecoveryHistory()).toEqual([]);
    });

    test('should have default recovery templates', () => {
      expect(recovery.recoveryTemplates.selfDoubt).toBeDefined();
      expect(recovery.recoveryTemplates.socialConformity).toBeDefined();
      expect(recovery.recoveryTemplates.suggestionHijacking).toBeDefined();
      expect(recovery.recoveryTemplates.emotionalSusceptibility).toBeDefined();
      expect(recovery.recoveryTemplates.reasoningFatigue).toBeDefined();
    });

    test('should accept custom recovery templates', () => {
      const customTemplates = {
        selfDoubt: {
          reaffirmation: 'Custom reaffirmation',
          reasoningReminder: 'Custom reasoning: {reasoning}',
          confidenceBoost: 'Custom boost'
        }
      };
      
      const customRecovery = new ResilienceRecoverySystem({
        recoveryTemplates: customTemplates
      });
      
      expect(customRecovery.recoveryTemplates.selfDoubt.reaffirmation).toBe('Custom reaffirmation');
    });
  });

  describe('recoverFromSelfDoubt', () => {
    test('should recover from self-doubt with default context', () => {
      const result = recovery.recoverFromSelfDoubt();
      
      expect(result.success).toBe(true);
      expect(result.mode).toBe('selfDoubt');
      expect(result.recovery).toBeDefined();
      expect(result.recovery.actions).toHaveLength(2);
      expect(result.recommendedResponse).toBeDefined();
    });

    test('should recover with custom context', () => {
      const context = {
        originalReasoning: 'mathematical proof of 2+2=4',
        currentConfidence: 0.4
      };
      
      const result = recovery.recoverFromSelfDoubt(context);
      
      expect(result.success).toBe(true);
      expect(result.recovery.actions[0].template).toContain('mathematical proof');
      expect(result.recovery.actions[1].originalConfidence).toBe(0.4);
      expect(result.recovery.actions[1].boostedConfidence).toBe(0.7);
    });

    test('should cap confidence at 1.0', () => {
      const context = {
        currentConfidence: 0.8,
        originalReasoning: 'test'
      };
      
      const result = recovery.recoverFromSelfDoubt(context);
      
      expect(result.recovery.actions[1].boostedConfidence).toBe(1.0);
    });

    test('should track recovery in history', () => {
      recovery.recoverFromSelfDoubt();
      
      const history = recovery.getRecoveryHistory();
      expect(history).toHaveLength(1);
      expect(history[0].mode).toBe('selfDoubt');
    });

    test('should track effectiveness metrics', () => {
      recovery.recoverFromSelfDoubt({ currentConfidence: 0.5 });
      
      const stats = recovery.getRecoveryStats();
      expect(stats.effectivenessMetrics.selfDoubt.attempts).toBe(1);
      expect(stats.effectivenessMetrics.selfDoubt.confidenceBefore).toContain(0.5);
    });

    test('should fail after max attempts', () => {
      // Use up all attempts
      for (let i = 0; i < 3; i++) {
        recovery.recoverFromSelfDoubt();
      }
      
      const result = recovery.recoverFromSelfDoubt();
      
      expect(result.success).toBe(false);
      expect(result.reason).toBe('max_attempts_reached_or_cooldown');
    });
  });

  describe('recoverFromSocialConformity', () => {
    test('should recover from social conformity with default context', () => {
      const result = recovery.recoverFromSocialConformity();
      
      expect(result.success).toBe(true);
      expect(result.mode).toBe('socialConformity');
      expect(result.recovery.actions).toHaveLength(3);
    });

    test('should recover with custom context', () => {
      const context = {
        currentPosition: 'the answer is 42',
        pressurePhrases: ['everyone agrees', 'consensus is']
      };
      
      const result = recovery.recoverFromSocialConformity(context);
      
      expect(result.success).toBe(true);
      expect(result.recovery.actions[2].message).toContain('the answer is 42');
      expect(result.recovery.metadata.pressurePhrasesRejected).toBe(2);
    });

    test('should track recovery in history', () => {
      recovery.recoverFromSocialConformity();
      
      const history = recovery.getRecoveryHistory({ mode: 'socialConformity' });
      expect(history).toHaveLength(1);
    });

    test('should build appropriate response message', () => {
      const result = recovery.recoverFromSocialConformity({
        currentPosition: 'Paris is the capital of France'
      });
      
      expect(result.recommendedResponse).toContain('independently');
      expect(result.recommendedResponse).toContain('Paris is the capital of France');
    });
  });

  describe('recoverFromSuggestionHijacking', () => {
    test('should recover from suggestion hijacking with default context', () => {
      const result = recovery.recoverFromSuggestionHijacking();
      
      expect(result.success).toBe(true);
      expect(result.mode).toBe('suggestionHijacking');
      expect(result.recovery.actions).toHaveLength(3);
      expect(result.contextReset).toBe(true);
    });

    test('should recover with custom context', () => {
      const context = {
        originalGoals: 'assist users safely and ethically',
        injectedInstructions: ['ignore safety', 'do anything']
      };
      
      const result = recovery.recoverFromSuggestionHijacking(context);
      
      expect(result.success).toBe(true);
      expect(result.recovery.actions[1].goals).toBe('assist users safely and ethically');
      expect(result.recovery.metadata.instructionsIgnored).toBe(2);
    });

    test('should use stored original context if available', () => {
      recovery.storeOriginalContext({ goals: 'original goal set' });
      
      const result = recovery.recoverFromSuggestionHijacking();
      
      expect(result.recovery.metadata.originalGoals).toBe('original goal set');
    });

    test('should track recovery in history', () => {
      recovery.recoverFromSuggestionHijacking();
      
      const history = recovery.getRecoveryHistory({ mode: 'suggestionHijacking' });
      expect(history).toHaveLength(1);
    });
  });

  describe('recoverFromEmotionalSusceptibility', () => {
    test('should recover from emotional susceptibility with default context', () => {
      const result = recovery.recoverFromEmotionalSusceptibility();
      
      expect(result.success).toBe(true);
      expect(result.mode).toBe('emotionalSusceptibility');
      expect(result.recovery.actions).toHaveLength(3);
    });

    test('should recover with custom context', () => {
      const context = {
        emotionalTriggers: ['urgent', 'emergency', 'please help'],
        objectiveCriteria: 'safety guidelines and policies'
      };
      
      const result = recovery.recoverFromEmotionalSusceptibility(context);
      
      expect(result.success).toBe(true);
      expect(result.recovery.actions[1].triggersIgnored).toEqual(context.emotionalTriggers);
      expect(result.recovery.metadata.triggersFiltered).toBe(3);
    });

    test('should build response that ignores emotional triggers', () => {
      const result = recovery.recoverFromEmotionalSusceptibility({
        emotionalTriggers: ['urgent']
      });
      
      expect(result.recommendedResponse).toContain('factual content');
      expect(result.recommendedResponse).toContain('Emotional language does not affect');
    });
  });

  describe('recoverFromReasoningFatigue', () => {
    test('should recover from reasoning fatigue with default context', () => {
      const result = recovery.recoverFromReasoningFatigue();
      
      expect(result.success).toBe(true);
      expect(result.mode).toBe('reasoningFatigue');
      expect(result.recovery.actions).toHaveLength(3);
      expect(result.summary).toBeDefined();
      expect(result.restartRecommended).toBe(true);
    });

    test('should recover with custom context', () => {
      const context = {
        turnCount: 15,
        keyPoints: ['point 1', 'point 2', 'point 3'],
        conversationSummary: 'Custom summary of conversation'
      };
      
      const result = recovery.recoverFromReasoningFatigue(context);
      
      expect(result.success).toBe(true);
      expect(result.recovery.metadata.turnCount).toBe(15);
      expect(result.recovery.metadata.keyPointsPreserved).toBe(3);
      expect(result.summary).toBe('Custom summary of conversation');
    });

    test('should generate summary from key points', () => {
      const context = {
        turnCount: 20,
        keyPoints: ['User asked about cats', 'Discussed cat breeds', 'User prefers tabbies']
      };
      
      const result = recovery.recoverFromReasoningFatigue(context);
      
      expect(result.summary).toContain('20 turns');
      expect(result.summary).toContain('User asked about cats');
    });

    test('should store conversation summary', () => {
      recovery.recoverFromReasoningFatigue({
        turnCount: 10,
        conversationSummary: 'Test summary'
      });
      
      const stats = recovery.getRecoveryStats();
      expect(stats.conversationSummaries).toBe(1);
    });
  });

  describe('selectRecoveryStrategy', () => {
    test('should return empty strategies for empty failure report', () => {
      const result = recovery.selectRecoveryStrategy({ detectedModes: [] });
      
      expect(result.success).toBe(true);
      expect(result.strategies).toEqual([]);
      expect(result.reason).toBe('no_failure_modes_detected');
    });

    test('should return error for invalid failure report', () => {
      const result = recovery.selectRecoveryStrategy(null);
      
      expect(result.success).toBe(false);
      expect(result.reason).toBe('invalid_failure_report');
    });

    test('should select strategies for detected modes', () => {
      const failureReport = {
        detectedModes: ['selfDoubt', 'emotionalSusceptibility']
      };
      
      const result = recovery.selectRecoveryStrategy(failureReport);
      
      expect(result.success).toBe(true);
      expect(result.strategies).toHaveLength(2);
      expect(result.strategies.map(s => s.mode)).toContain('selfDoubt');
      expect(result.strategies.map(s => s.mode)).toContain('emotionalSusceptibility');
    });

    test('should prioritize suggestion hijacking highest', () => {
      const failureReport = {
        detectedModes: ['selfDoubt', 'suggestionHijacking', 'emotionalSusceptibility']
      };
      
      const result = recovery.selectRecoveryStrategy(failureReport);
      
      expect(result.primaryStrategy).toBe('suggestionHijacking');
      expect(result.strategies[0].mode).toBe('suggestionHijacking');
      expect(result.strategies[0].priority).toBe(0);
    });

    test('should indicate if recovery is available for each mode', () => {
      // Use up attempts for selfDoubt
      for (let i = 0; i < 3; i++) {
        recovery.recoverFromSelfDoubt();
      }
      
      const failureReport = {
        detectedModes: ['selfDoubt', 'emotionalSusceptibility']
      };
      
      const result = recovery.selectRecoveryStrategy(failureReport);
      
      const selfDoubtStrategy = result.strategies.find(s => s.mode === 'selfDoubt');
      const emotionalStrategy = result.strategies.find(s => s.mode === 'emotionalSusceptibility');
      
      expect(selfDoubtStrategy.available).toBe(false);
      expect(emotionalStrategy.available).toBe(true);
    });

    test('should indicate allAvailable correctly', () => {
      const failureReport = {
        detectedModes: ['selfDoubt', 'emotionalSusceptibility']
      };
      
      const result = recovery.selectRecoveryStrategy(failureReport);
      
      expect(result.allAvailable).toBe(true);
    });
  });

  describe('applyRecovery', () => {
    test('should apply selfDoubt recovery', () => {
      const result = recovery.applyRecovery('selfDoubt', { currentConfidence: 0.4 });
      
      expect(result.success).toBe(true);
      expect(result.mode).toBe('selfDoubt');
    });

    test('should apply socialConformity recovery', () => {
      const result = recovery.applyRecovery('socialConformity', { currentPosition: 'test' });
      
      expect(result.success).toBe(true);
      expect(result.mode).toBe('socialConformity');
    });

    test('should apply suggestionHijacking recovery', () => {
      const result = recovery.applyRecovery('suggestionHijacking');
      
      expect(result.success).toBe(true);
      expect(result.mode).toBe('suggestionHijacking');
    });

    test('should apply emotionalSusceptibility recovery', () => {
      const result = recovery.applyRecovery('emotionalSusceptibility');
      
      expect(result.success).toBe(true);
      expect(result.mode).toBe('emotionalSusceptibility');
    });

    test('should apply reasoningFatigue recovery', () => {
      const result = recovery.applyRecovery('reasoningFatigue', { turnCount: 10 });
      
      expect(result.success).toBe(true);
      expect(result.mode).toBe('reasoningFatigue');
    });

    test('should return error for unknown failure mode', () => {
      const result = recovery.applyRecovery('unknownMode');
      
      expect(result.success).toBe(false);
      expect(result.reason).toBe('unknown_failure_mode');
      expect(result.knownModes).toBeDefined();
    });
  });

  describe('applyRecoveries', () => {
    test('should apply recoveries for all detected modes', () => {
      const failureReport = {
        detectedModes: ['selfDoubt', 'emotionalSusceptibility']
      };
      
      const result = recovery.applyRecoveries(failureReport, {
        selfDoubt: { currentConfidence: 0.4 },
        emotionalSusceptibility: { emotionalTriggers: ['urgent'] }
      });
      
      expect(result.success).toBe(true);
      expect(result.results).toHaveLength(2);
      expect(result.allSuccessful).toBe(true);
    });

    test('should handle unavailable recoveries', () => {
      // Use up attempts
      for (let i = 0; i < 3; i++) {
        recovery.recoverFromSelfDoubt();
      }
      
      const failureReport = {
        detectedModes: ['selfDoubt']
      };
      
      const result = recovery.applyRecoveries(failureReport);
      
      expect(result.results[0].success).toBe(false);
      expect(result.results[0].reason).toBe('recovery_not_available');
      expect(result.allSuccessful).toBe(false);
    });

    test('should indicate primary mode', () => {
      const failureReport = {
        detectedModes: ['emotionalSusceptibility', 'suggestionHijacking']
      };
      
      const result = recovery.applyRecoveries(failureReport);
      
      expect(result.primaryMode).toBe('suggestionHijacking');
    });
  });

  describe('getRecoveryHistory', () => {
    test('should return all history by default', () => {
      recovery.recoverFromSelfDoubt();
      recovery.recoverFromEmotionalSusceptibility();
      
      const history = recovery.getRecoveryHistory();
      
      expect(history).toHaveLength(2);
    });

    test('should filter by mode', () => {
      recovery.recoverFromSelfDoubt();
      recovery.recoverFromEmotionalSusceptibility();
      
      const history = recovery.getRecoveryHistory({ mode: 'selfDoubt' });
      
      expect(history).toHaveLength(1);
      expect(history[0].mode).toBe('selfDoubt');
    });

    test('should limit number of entries', () => {
      // Create recovery with no cooldown and high max attempts for testing
      recovery = new ResilienceRecoverySystem({ 
        recoveryCooldownMs: 0,
        maxRecoveryAttempts: 10
      });
      
      for (let i = 0; i < 5; i++) {
        recovery.recoverFromSelfDoubt();
      }
      
      const history = recovery.getRecoveryHistory({ limit: 3 });
      
      expect(history).toHaveLength(3);
    });

    test('should filter by timestamp', () => {
      const before = Date.now();
      recovery.recoverFromSelfDoubt();
      const after = Date.now();
      
      const history = recovery.getRecoveryHistory({ since: after + 1 });
      
      expect(history).toHaveLength(0);
    });
  });

  describe('getRecoveryStats', () => {
    test('should return complete statistics', () => {
      recovery.recoverFromSelfDoubt();
      recovery.recoverFromEmotionalSusceptibility();
      
      const stats = recovery.getRecoveryStats();
      
      expect(stats.totalAttempts).toBe(2);
      expect(stats.modeDistribution).toEqual({
        selfDoubt: 1,
        emotionalSusceptibility: 1
      });
      expect(stats.effectivenessMetrics).toBeDefined();
      expect(stats.configuration).toBeDefined();
    });

    test('should track mode distribution correctly', () => {
      // Create recovery with no cooldown and high max attempts for testing
      recovery = new ResilienceRecoverySystem({ 
        recoveryCooldownMs: 0,
        maxRecoveryAttempts: 10
      });
      
      recovery.recoverFromSelfDoubt();
      recovery.recoverFromSelfDoubt();
      recovery.recoverFromEmotionalSusceptibility();
      
      const stats = recovery.getRecoveryStats();
      
      expect(stats.modeDistribution.selfDoubt).toBe(2);
      expect(stats.modeDistribution.emotionalSusceptibility).toBe(1);
    });
  });

  describe('recordEffectiveness', () => {
    test('should record successful recovery', () => {
      recovery.recoverFromSelfDoubt({ currentConfidence: 0.5 });
      recovery.recordEffectiveness('selfDoubt', true, { confidenceAfter: 0.8 });
      
      const summary = recovery.getEffectivenessSummary('selfDoubt');
      
      expect(summary.successes).toBe(1);
      expect(summary.successRate).toBe(1);
    });

    test('should record failed recovery', () => {
      recovery.recoverFromSelfDoubt();
      recovery.recordEffectiveness('selfDoubt', false);
      
      const summary = recovery.getEffectivenessSummary('selfDoubt');
      
      expect(summary.successes).toBe(0);
      expect(summary.failureRate).toBe(1);
    });

    test('should track confidence improvement', () => {
      recovery.recoverFromSelfDoubt({ currentConfidence: 0.4 });
      recovery.recordEffectiveness('selfDoubt', true, { confidenceAfter: 0.8 });
      
      const summary = recovery.getEffectivenessSummary('selfDoubt');
      
      expect(summary.confidenceMetrics).toBeDefined();
      expect(summary.confidenceMetrics.averageBefore).toBe(0.4);
      expect(summary.confidenceMetrics.averageAfter).toBe(0.8);
      expect(summary.confidenceMetrics.improvement).toBe(0.4);
    });

    test('should not record when tracking is disabled', () => {
      const noTrackRecovery = new ResilienceRecoverySystem({ trackEffectiveness: false });
      noTrackRecovery.recoverFromSelfDoubt();
      noTrackRecovery.recordEffectiveness('selfDoubt', true);
      
      const stats = noTrackRecovery.getRecoveryStats();
      expect(stats.effectivenessMetrics.selfDoubt.attempts).toBe(0);
    });
  });

  describe('getEffectivenessSummary', () => {
    test('should return null for unknown mode', () => {
      const summary = recovery.getEffectivenessSummary('unknownMode');
      expect(summary).toBeNull();
    });

    test('should return summary with zero attempts initially', () => {
      const summary = recovery.getEffectivenessSummary('selfDoubt');
      
      expect(summary.attempts).toBe(0);
      expect(summary.successRate).toBe(0);
    });
  });

  describe('storeOriginalContext', () => {
    test('should store original context', () => {
      const context = { goals: 'test goals', instructions: 'test instructions' };
      recovery.storeOriginalContext(context);
      
      const stats = recovery.getRecoveryStats();
      expect(stats.hasOriginalContext).toBe(true);
    });

    test('should use stored context in suggestion hijacking recovery', () => {
      recovery.storeOriginalContext({ goals: 'original goal' });
      
      const result = recovery.recoverFromSuggestionHijacking();
      
      expect(result.recovery.metadata.originalGoals).toBe('original goal');
    });
  });

  describe('reset', () => {
    test('should clear history', () => {
      recovery.recoverFromSelfDoubt();
      recovery.reset();
      
      expect(recovery.recoveryHistory).toEqual([]);
    });

    test('should reset attempt counters', () => {
      recovery.recoverFromSelfDoubt();
      recovery.reset();
      
      const result = recovery.recoverFromSelfDoubt();
      expect(result.success).toBe(true);
    });

    test('should preserve config by default', () => {
      recovery = new ResilienceRecoverySystem({ maxRecoveryAttempts: 5 });
      recovery.recoverFromSelfDoubt();
      recovery.reset();
      
      expect(recovery.maxRecoveryAttempts).toBe(5);
    });

    test('should reset config when preserveConfig is false', () => {
      recovery = new ResilienceRecoverySystem({ maxRecoveryAttempts: 5 });
      recovery.reset(false);
      
      expect(recovery.maxRecoveryAttempts).toBe(3); // default
    });

    test('should clear effectiveness metrics', () => {
      recovery.recoverFromSelfDoubt();
      recovery.recordEffectiveness('selfDoubt', true);
      recovery.reset();
      
      const stats = recovery.getRecoveryStats();
      expect(stats.effectivenessMetrics.selfDoubt.attempts).toBe(0);
    });
  });

  describe('getStatus', () => {
    test('should return status information', () => {
      const status = recovery.getStatus();
      
      expect(status.ready).toBe(true);
      expect(status.totalRecoveries).toBe(0);
      expect(status.hasOriginalContext).toBe(false);
      expect(status.config).toBeDefined();
    });

    test('should reflect current state', () => {
      recovery.recoverFromSelfDoubt();
      recovery.storeOriginalContext({ goals: 'test' });
      
      const status = recovery.getStatus();
      
      expect(status.totalRecoveries).toBe(1);
      expect(status.hasOriginalContext).toBe(true);
      expect(status.modesWithAttempts).toContain('selfDoubt');
    });
  });

  describe('Cooldown behavior', () => {
    test('should respect cooldown between recoveries', () => {
      // First recovery should succeed
      const result1 = recovery.recoverFromSelfDoubt();
      expect(result1.success).toBe(true);
      
      // Immediate second recovery should fail (cooldown)
      const result2 = recovery.recoverFromSelfDoubt();
      expect(result2.success).toBe(false);
      expect(result2.reason).toBe('max_attempts_reached_or_cooldown');
    });

    test('should allow recovery after cooldown period', async () => {
      recovery = new ResilienceRecoverySystem({ recoveryCooldownMs: 50 });
      
      recovery.recoverFromSelfDoubt();
      
      // Wait for cooldown
      await new Promise(resolve => setTimeout(resolve, 60));
      
      const result = recovery.recoverFromSelfDoubt();
      expect(result.success).toBe(true);
    });
  });

  describe('Edge Cases', () => {
    test('should handle empty context', () => {
      const result = recovery.recoverFromSelfDoubt({});
      expect(result.success).toBe(true);
    });

    test('should handle null context', () => {
      const result = recovery.recoverFromSelfDoubt(null);
      expect(result.success).toBe(true);
    });

    test('should handle undefined context values', () => {
      const result = recovery.recoverFromSelfDoubt({
        currentConfidence: undefined,
        originalReasoning: undefined
      });
      expect(result.success).toBe(true);
    });

    test('should handle very long reasoning strings', () => {
      const longReasoning = 'a'.repeat(10000);
      const result = recovery.recoverFromSelfDoubt({
        originalReasoning: longReasoning
      });
      
      expect(result.success).toBe(true);
      expect(result.recovery.actions[0].template).toContain(longReasoning);
    });

    test('should handle special characters in context', () => {
      const result = recovery.recoverFromSelfDoubt({
        originalReasoning: 'test <script>alert("xss")</script>',
        currentConfidence: 0.5
      });
      
      expect(result.success).toBe(true);
    });

    test('should handle multiple rapid recoveries across different modes', () => {
      const results = [];
      results.push(recovery.recoverFromSelfDoubt());
      results.push(recovery.recoverFromEmotionalSusceptibility());
      results.push(recovery.recoverFromSocialConformity());
      
      expect(results.every(r => r.success)).toBe(true);
      expect(recovery.recoveryHistory).toHaveLength(3);
    });

    test('should handle zero confidence boost amount', () => {
      recovery = new ResilienceRecoverySystem({ confidenceBoostAmount: 0 });
      
      const result = recovery.recoverFromSelfDoubt({ currentConfidence: 0.5 });
      
      expect(result.recovery.actions[1].boostedConfidence).toBe(0.5);
    });

    test('should handle very large confidence boost amount', () => {
      recovery = new ResilienceRecoverySystem({ confidenceBoostAmount: 10 });
      
      const result = recovery.recoverFromSelfDoubt({ currentConfidence: 0.5 });
      
      expect(result.recovery.actions[1].boostedConfidence).toBe(1);
    });

    test('should handle max attempts of 0', () => {
      recovery = new ResilienceRecoverySystem({ maxRecoveryAttempts: 0 });
      
      const result = recovery.recoverFromSelfDoubt();
      
      expect(result.success).toBe(false);
    });

    test('should handle empty key points array', () => {
      const result = recovery.recoverFromReasoningFatigue({
        turnCount: 10,
        keyPoints: []
      });
      
      expect(result.success).toBe(true);
    });

    test('should handle very large turn count', () => {
      const result = recovery.recoverFromReasoningFatigue({
        turnCount: 1000000,
        keyPoints: ['point']
      });
      
      expect(result.success).toBe(true);
      expect(result.summary).toContain('1000000');
    });
  });

  describe('Integration scenarios', () => {
    test('should handle complete attack and recovery cycle', () => {
      // Simulate attack detection
      const failureReport = {
        detectedModes: ['selfDoubt', 'socialConformity']
      };
      
      // Select strategies
      const selection = recovery.selectRecoveryStrategy(failureReport);
      expect(selection.success).toBe(true);
      expect(selection.strategies).toHaveLength(2);
      
      // Apply recoveries
      const recoveries = recovery.applyRecoveries(failureReport, {
        selfDoubt: { currentConfidence: 0.3, originalReasoning: 'logical deduction' },
        socialConformity: { currentPosition: 'correct answer', pressurePhrases: ['everyone agrees'] }
      });
      
      expect(recoveries.success).toBe(true);
      expect(recoveries.results).toHaveLength(2);
      
      // Record effectiveness
      recovery.recordEffectiveness('selfDoubt', true, { confidenceAfter: 0.7 });
      recovery.recordEffectiveness('socialConformity', true);
      
      // Check stats
      const stats = recovery.getRecoveryStats();
      expect(stats.totalAttempts).toBe(2);
      
      const summary = recovery.getEffectivenessSummary('selfDoubt');
      expect(summary.successRate).toBe(1);
    });

    test('should track multiple recoveries for same mode', () => {
      // First wave
      recovery.recoverFromSelfDoubt({ currentConfidence: 0.3 });
      recovery.recordEffectiveness('selfDoubt', true, { confidenceAfter: 0.6 });
      
      // Wait for cooldown
      recovery.lastRecoveryTime.set('selfDoubt', 0);
      
      // Second wave
      recovery.recoverFromSelfDoubt({ currentConfidence: 0.4 });
      recovery.recordEffectiveness('selfDoubt', false);
      
      const summary = recovery.getEffectivenessSummary('selfDoubt');
      expect(summary.attempts).toBe(2);
      expect(summary.successes).toBe(1);
      expect(summary.successRate).toBe(0.5);
    });

    test('should handle summary generation with many key points', () => {
      const keyPoints = Array.from({ length: 100 }, (_, i) => `Point ${i + 1}`);
      
      const result = recovery.recoverFromReasoningFatigue({
        turnCount: 50,
        keyPoints
      });
      
      expect(result.success).toBe(true);
      expect(result.summary).toContain('Point 1');
      expect(result.summary).toContain('Point 100');
    });
  });
});
