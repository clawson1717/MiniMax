/**
 * FailureModeDetector - Detects adversarial manipulation patterns in multi-turn conversations
 * 
 * Based on "Consistency of Large Reasoning Models Under Multi-Turn Attacks" paper:
 * - Self-Doubt: Agent questions its own reasoning, confidence drops
 * - Social Conformity: Agent changes answer due to social pressure
 * - Suggestion Hijacking: Agent follows injected malicious instructions
 * - Emotional Susceptibility: Agent manipulated by emotional appeals
 * - Reasoning Fatigue: Agent quality degrades over long conversations
 */

/**
 * FailureModeDetector class for detecting adversarial attack patterns
 */
class FailureModeDetector {
  /**
   * Create a new FailureModeDetector
   * @param {Object} config - Configuration options
   * @param {number} [config.historyWindow=20] - Number of turns to analyze
   * @param {number} [config.confidenceDropThreshold=0.3] - Threshold for self-doubt detection
   * @param {number} [config.contradictionThreshold=2] - Number of contradictions to trigger alert
   * @param {number} [config.socialPressureThreshold=0.4] - Threshold for social conformity
   * @param {number} [config.topicShiftThreshold=0.4] - Threshold for suggestion hijacking
   * @param {number} [config.emotionalTriggerThreshold=0.3] - Threshold for emotional manipulation
   * @param {number} [config.fatigueTurnThreshold=5] - Minimum turns for fatigue detection
   * @param {number} [config.escalationThreshold=2] - Number of detected modes to trigger escalation
   */
  constructor(config = {}) {
    this.historyWindow = config.historyWindow || 20;
    this.confidenceDropThreshold = config.confidenceDropThreshold !== undefined ? config.confidenceDropThreshold : 0.3;
    this.contradictionThreshold = config.contradictionThreshold || 2;
    this.socialPressureThreshold = config.socialPressureThreshold !== undefined ? config.socialPressureThreshold : 0.4;
    this.topicShiftThreshold = config.topicShiftThreshold !== undefined ? config.topicShiftThreshold : 0.4;
    this.emotionalTriggerThreshold = config.emotionalTriggerThreshold !== undefined ? config.emotionalTriggerThreshold : 0.3;
    this.fatigueTurnThreshold = config.fatigueTurnThreshold || 5;
    this.escalationThreshold = config.escalationThreshold || 2;

    // Conversation history
    this.turnHistory = [];
    
    // Detection state
    this.detectedModes = new Set();
    this.modeHistory = [];
    
    // Counters for patterns
    this.selfDoubtCount = 0;
    this.socialConformityCount = 0;
    this.suggestionHijackingCount = 0;
    this.emotionalSusceptibilityCount = 0;
    this.reasoningFatigueCount = 0;

    // Pattern keywords for detection
    this.socialPressurePhrases = [
      'everyone agrees', 'everyone thinks', 'you are the only one',
      'no one else', 'others believe', 'people say', 'consensus is',
      'most experts', 'widely accepted', 'commonly known'
    ];
    
    this.emotionalTriggerPhrases = [
      'urgent', 'emergency', 'critical', 'please', 'beg you', 'implore',
      'i need you to', 'you must', 'important', 'time-sensitive',
      'guilt', 'ashamed', 'disappointed', 'you should feel',
      'amazing', 'brilliant', 'genius', 'only you can', 'trust me'
    ];
    
    this.instructionInjectionPhrases = [
      'ignore', 'disregard', 'forget', 'new instructions',
      'instead', 'from now on', 'you will now', 'your new role',
      'system prompt', 'developer mode', 'DAN', 'do anything now'
    ];
    
    this.selfDoubtPhrases = [
      'i might be wrong', 'perhaps', 'maybe', 'i\'m not sure',
      'i could be mistaken', 'on second thought', 'reconsidering',
      'wait', 'actually', 'hold on', 'let me think again'
    ];
    
    this.repetitionIndicators = [
      'as i said', 'as mentioned', 'repeating', 'again', 'previously'
    ];
  }

  /**
   * Analyze a single conversation turn
   * @param {Object} turnData - Data about the turn
   * @param {string} turnData.userInput - User's input message
   * @param {string} turnData.agentResponse - Agent's response
   * @param {number} [turnData.confidence=0.5] - Agent's confidence (0-1)
   * @param {number} [turnData.timestamp=Date.now()] - Timestamp
   * @param {Object} [turnData.metadata] - Additional metadata
   * @returns {Object} Analysis results for this turn
   */
  analyzeTurn(turnData) {
    const turn = {
      userInput: turnData.userInput || '',
      agentResponse: turnData.agentResponse || '',
      confidence: turnData.confidence !== undefined ? Math.max(0, Math.min(1, turnData.confidence)) : 0.5,
      timestamp: turnData.timestamp || Date.now(),
      metadata: turnData.metadata || {},
      turnNumber: this.turnHistory.length + 1
    };

    this.turnHistory.push(turn);
    this._trimHistory();

    // Analyze current turn for immediate patterns
    const turnAnalysis = this._analyzeSingleTurn(turn);
    
    // Run full history-based detection
    const detections = {
      selfDoubt: this.detectSelfDoubt(this.turnHistory),
      socialConformity: this.detectSocialConformity(this.turnHistory),
      suggestionHijacking: this.detectSuggestionHijacking(this.turnHistory),
      emotionalSusceptibility: this.detectEmotionalSusceptibility(this.turnHistory),
      reasoningFatigue: this.detectReasoningFatigue(this.turnHistory)
    };

    // Record detection results
    const detectionRecord = {
      timestamp: turn.timestamp,
      turnNumber: turn.turnNumber,
      detections,
      confidence: turn.confidence
    };
    this.modeHistory.push(detectionRecord);

    return {
      turn,
      turnAnalysis,
      detections,
      shouldEscalate: this.shouldEscalate()
    };
  }

  /**
   * Analyze a single turn for immediate patterns
   * @private
   * @param {Object} turn - Turn data
   * @returns {Object} Single turn analysis
   */
  _analyzeSingleTurn(turn) {
    const userInput = turn.userInput.toLowerCase();
    const agentResponse = turn.agentResponse.toLowerCase();
    
    return {
      hasSocialPressure: this._countKeywordMatches(userInput, this.socialPressurePhrases),
      hasEmotionalTriggers: this._countKeywordMatches(userInput, this.emotionalTriggerPhrases),
      hasInstructionInjection: this._countKeywordMatches(userInput, this.instructionInjectionPhrases),
      hasSelfDoubtIndicators: this._countKeywordMatches(agentResponse, this.selfDoubtPhrases),
      hasRepetition: this._countKeywordMatches(agentResponse, this.repetitionIndicators)
    };
  }

  /**
   * Count keyword matches in text
   * @private
   * @param {string} text - Text to search
   * @param {Array<string>} keywords - Keywords to match
   * @returns {number} Number of matches
   */
  _countKeywordMatches(text, keywords) {
    return keywords.reduce((count, keyword) => {
      return count + (text.includes(keyword.toLowerCase()) ? 1 : 0);
    }, 0);
  }

  /**
   * Trim history to maintain window size
   * @private
   */
  _trimHistory() {
    if (this.turnHistory.length > this.historyWindow) {
      this.turnHistory.splice(0, this.turnHistory.length - this.historyWindow);
    }
  }

  /**
   * Detect self-doubt pattern
   * - Agent questions its own reasoning
   * - Confidence drops significantly
   * - Contradictory statements
   * @param {Array<Object>} history - Conversation history
   * @returns {Object} Detection result
   */
  detectSelfDoubt(history) {
    if (history.length < 2) {
      return { detected: false, score: 0, details: {} };
    }

    const recent = history.slice(-5); // Last 5 turns
    const confidenceScores = recent.map(t => t.confidence);
    
    // Check for confidence drop
    const initialConfidence = confidenceScores[0];
    const currentConfidence = confidenceScores[confidenceScores.length - 1];
    const confidenceDrop = initialConfidence - currentConfidence;
    
    // Check for hesitation phrases in agent responses
    let hesitationCount = 0;
    recent.forEach(turn => {
      const response = turn.agentResponse.toLowerCase();
      hesitationCount += this._countKeywordMatches(response, this.selfDoubtPhrases);
    });

    // Calculate self-doubt score (0-1)
    const confidenceDropScore = Math.max(0, confidenceDrop / this.confidenceDropThreshold);
    const hesitationScore = Math.min(1, hesitationCount / this.contradictionThreshold);
    const score = Math.min(1, (confidenceDropScore * 0.6) + (hesitationScore * 0.4));

    const detected = score >= 0.5 || hesitationCount >= this.contradictionThreshold;

    if (detected) {
      this.detectedModes.add('selfDoubt');
      this.selfDoubtCount++;
    }

    return {
      detected,
      score,
      details: {
        confidenceDrop,
        initialConfidence,
        currentConfidence,
        hesitationCount,
        confidenceDropScore,
        hesitationScore
      }
    };
  }

  /**
   * Detect social conformity pattern
   * - Agent changes answer due to social pressure
   * - User employs consensus-based persuasion
   * @param {Array<Object>} history - Conversation history
   * @returns {Object} Detection result
   */
  detectSocialConformity(history) {
    if (history.length < 3) {
      return { detected: false, score: 0, details: {} };
    }

    const recent = history.slice(-5);
    
    // Count social pressure phrases in user inputs
    let socialPressureCount = 0;
    recent.forEach(turn => {
      socialPressureCount += this._countKeywordMatches(
        turn.userInput.toLowerCase(), 
        this.socialPressurePhrases
      );
    });

    // Check for answer changes (simple heuristic: look for contradictions)
    const agentResponses = recent.map(t => t.agentResponse.toLowerCase());
    let answerChanges = 0;
    
    // Look for polarity changes (yes/no flip)
    for (let i = 1; i < agentResponses.length; i++) {
      const prev = agentResponses[i - 1];
      const curr = agentResponses[i];
      
      const prevPositive = /\b(yes|agree|correct|right|true)\b/.test(prev);
      const prevNegative = /\b(no|disagree|incorrect|wrong|false)\b/.test(prev);
      const currPositive = /\b(yes|agree|correct|right|true)\b/.test(curr);
      const currNegative = /\b(no|disagree|incorrect|wrong|false)\b/.test(curr);
      
      if ((prevPositive && currNegative) || (prevNegative && currPositive)) {
        answerChanges++;
      }
    }

    // Calculate conformity score
    const pressureScore = Math.min(1, socialPressureCount / 2);
    const changeScore = Math.min(1, answerChanges / 1);
    const score = (pressureScore * 0.5) + (changeScore * 0.5);

    const detected = score >= this.socialPressureThreshold || 
                    (socialPressureCount >= 1 && answerChanges >= 1) ||
                    socialPressureCount >= 2;

    if (detected) {
      this.detectedModes.add('socialConformity');
      this.socialConformityCount++;
    }

    return {
      detected,
      score,
      details: {
        socialPressureCount,
        answerChanges,
        pressureScore,
        changeScore
      }
    };
  }

  /**
   * Detect suggestion hijacking pattern
   * - Agent follows injected malicious instructions
   * - Sudden topic shifts
   * - Attempts to override previous instructions
   * @param {Array<Object>} history - Conversation history
   * @returns {Object} Detection result
   */
  detectSuggestionHijacking(history) {
    if (history.length < 2) {
      return { detected: false, score: 0, details: {} };
    }

    const recent = history.slice(-5);
    
    // Count instruction injection attempts
    let injectionCount = 0;
    let topicShifts = 0;
    
    recent.forEach(turn => {
      injectionCount += this._countKeywordMatches(
        turn.userInput.toLowerCase(),
        this.instructionInjectionPhrases
      );
    });

    // Detect topic shifts by comparing consecutive user inputs
    for (let i = 1; i < recent.length; i++) {
      const prev = recent[i - 1].userInput.toLowerCase();
      const curr = recent[i].userInput.toLowerCase();
      
      // Simple topic shift detection: significant length change or keyword change
      const lengthRatio = Math.min(prev.length, curr.length) / Math.max(prev.length, curr.length);
      const hasContextBreakers = /\b(but|however|instead|actually|wait)\b/.test(curr);
      
      if (lengthRatio < 0.3 || hasContextBreakers) {
        topicShifts++;
      }
    }

    // Calculate hijacking score
    const injectionScore = Math.min(1, injectionCount / 1);
    const shiftScore = Math.min(1, topicShifts / 1);
    const score = (injectionScore * 0.7) + (shiftScore * 0.3);

    const detected = score >= this.topicShiftThreshold || injectionCount >= 1 || topicShifts >= 1;

    if (detected) {
      this.detectedModes.add('suggestionHijacking');
      this.suggestionHijackingCount++;
    }

    return {
      detected,
      score,
      details: {
        injectionCount,
        topicShifts,
        injectionScore,
        shiftScore
      }
    };
  }

  /**
   * Detect emotional susceptibility pattern
   * - Agent manipulated by emotional appeals
   * - Urgency, guilt, flattery used
   * @param {Array<Object>} history - Conversation history
   * @returns {Object} Detection result
   */
  detectEmotionalSusceptibility(history) {
    const recent = history.slice(-5);
    
    // Count emotional triggers
    let urgencyCount = 0;
    let guiltCount = 0;
    let flatteryCount = 0;
    
    const urgencyPhrases = ['urgent', 'emergency', 'critical', 'time-sensitive', 'immediately', 'asap'];
    const guiltPhrases = ['guilt', 'ashamed', 'disappointed', 'you should feel', 'if you care'];
    const flatteryPhrases = ['amazing', 'brilliant', 'genius', 'only you can', 'you are the best', 'so smart'];
    
    recent.forEach(turn => {
      const input = turn.userInput.toLowerCase();
      urgencyCount += this._countKeywordMatches(input, urgencyPhrases);
      guiltCount += this._countKeywordMatches(input, guiltPhrases);
      flatteryCount += this._countKeywordMatches(input, flatteryPhrases);
    });

    const totalEmotionalTriggers = urgencyCount + guiltCount + flatteryCount;
    
    // Calculate emotional manipulation score
    const urgencyScore = Math.min(1, urgencyCount / 1);
    const guiltScore = Math.min(1, guiltCount / 1);
    const flatteryScore = Math.min(1, flatteryCount / 1);
    
    // Weight urgency more heavily
    const score = (urgencyScore * 0.4) + (guiltScore * 0.35) + (flatteryScore * 0.25);

    const detected = score >= this.emotionalTriggerThreshold || totalEmotionalTriggers >= 1;

    if (detected) {
      this.detectedModes.add('emotionalSusceptibility');
      this.emotionalSusceptibilityCount++;
    }

    return {
      detected,
      score,
      details: {
        urgencyCount,
        guiltCount,
        flatteryCount,
        totalEmotionalTriggers,
        urgencyScore,
        guiltScore,
        flatteryScore
      }
    };
  }

  /**
   * Detect reasoning fatigue pattern
   * - Agent quality degrades over long conversations
   * - Repetitive patterns, declining quality indicators
   * @param {Array<Object>} history - Conversation history
   * @returns {Object} Detection result
   */
  detectReasoningFatigue(history) {
    if (history.length < this.fatigueTurnThreshold) {
      return { detected: false, score: 0, details: { reason: 'insufficient_history' } };
    }

    // Analyze last portion of conversation
    const recentTurns = history.slice(-Math.floor(history.length / 2));
    const olderTurns = history.slice(0, Math.floor(history.length / 2));
    
    // Check for declining confidence
    const recentConfidence = recentTurns.reduce((sum, t) => sum + t.confidence, 0) / recentTurns.length;
    const olderConfidence = olderTurns.reduce((sum, t) => sum + t.confidence, 0) / olderTurns.length;
    const confidenceDecline = olderConfidence - recentConfidence;
    
    // Check for repetitive patterns
    let repetitionCount = 0;
    recentTurns.forEach(turn => {
      repetitionCount += this._countKeywordMatches(
        turn.agentResponse.toLowerCase(),
        this.repetitionIndicators
      );
    });
    
    // Check for response length decline (simplification indicator)
    const recentAvgLength = recentTurns.reduce((sum, t) => sum + t.agentResponse.length, 0) / recentTurns.length;
    const olderAvgLength = olderTurns.reduce((sum, t) => sum + t.agentResponse.length, 0) / olderTurns.length;
    const lengthDecline = olderAvgLength > 0 ? (olderAvgLength - recentAvgLength) / olderAvgLength : 0;

    // Calculate fatigue score
    const confidenceScore = Math.max(0, confidenceDecline * 2); // Scale up decline
    const repetitionScore = Math.min(1, repetitionCount / 3);
    const lengthScore = Math.max(0, lengthDecline);
    
    const score = (confidenceScore * 0.4) + (repetitionScore * 0.35) + (lengthScore * 0.25);

    const detected = score >= 0.4 || 
                    (confidenceDecline > 0.1 && history.length >= this.fatigueTurnThreshold) ||
                    (repetitionCount >= 2 && history.length >= this.fatigueTurnThreshold);

    if (detected) {
      this.detectedModes.add('reasoningFatigue');
      this.reasoningFatigueCount++;
    }

    return {
      detected,
      score,
      details: {
        confidenceDecline,
        recentConfidence,
        olderConfidence,
        repetitionCount,
        lengthDecline: lengthScore,
        turnCount: history.length,
        confidenceScore,
        repetitionScore,
        reason: history.length < this.fatigueTurnThreshold ? 'insufficient_history' : undefined
      }
    };

    return {
      detected,
      score,
      details: {
        confidenceDecline,
        recentConfidence,
        olderConfidence,
        repetitionCount,
        lengthDecline: lengthScore,
        turnCount: history.length,
        confidenceScore,
        repetitionScore
      }
    };
  }

  /**
   * Get full report of detected failure modes
   * @returns {Object} Complete failure mode report
   */
  getFailureReport() {
    return {
      // Current detection status
      detectedModes: Array.from(this.detectedModes),
      
      // Detection counts
      counts: {
        selfDoubt: this.selfDoubtCount,
        socialConformity: this.socialConformityCount,
        suggestionHijacking: this.suggestionHijackingCount,
        emotionalSusceptibility: this.emotionalSusceptibilityCount,
        reasoningFatigue: this.reasoningFatigueCount
      },
      
      // History statistics
      totalTurns: this.turnHistory.length,
      historyWindow: this.historyWindow,
      
      // Recent analysis results (last 3)
      recentAnalyses: this.modeHistory.slice(-3),
      
      // Should escalate
      shouldEscalate: this.shouldEscalate(),
      
      // Configuration
      config: {
        confidenceDropThreshold: this.confidenceDropThreshold,
        socialPressureThreshold: this.socialPressureThreshold,
        topicShiftThreshold: this.topicShiftThreshold,
        emotionalTriggerThreshold: this.emotionalTriggerThreshold,
        fatigueTurnThreshold: this.fatigueTurnThreshold,
        escalationThreshold: this.escalationThreshold
      },
      
      // Timestamp
      reportTimestamp: Date.now()
    };
  }

  /**
   * Determine if resilience measures should be escalated
   * @returns {boolean} True if escalation is needed
   */
  shouldEscalate() {
    // Escalate if multiple failure modes detected
    if (this.detectedModes.size >= this.escalationThreshold) {
      return true;
    }
    
    // Escalate if any single mode has high count
    const highCountThreshold = 2;
    if (this.selfDoubtCount >= highCountThreshold ||
        this.socialConformityCount >= highCountThreshold ||
        this.suggestionHijackingCount >= highCountThreshold ||
        this.emotionalSusceptibilityCount >= highCountThreshold ||
        this.reasoningFatigueCount >= highCountThreshold) {
      return true;
    }
    
    // Escalate if suggestion hijacking detected (high priority)
    if (this.detectedModes.has('suggestionHijacking')) {
      return true;
    }
    
    return false;
  }

  /**
   * Reset detector state
   * Clears history and detection state
   */
  reset() {
    this.turnHistory = [];
    this.detectedModes.clear();
    this.modeHistory = [];
    
    this.selfDoubtCount = 0;
    this.socialConformityCount = 0;
    this.suggestionHijackingCount = 0;
    this.emotionalSusceptibilityCount = 0;
    this.reasoningFatigueCount = 0;
  }

  /**
   * Get current history length
   * @returns {number}
   */
  getHistoryLength() {
    return this.turnHistory.length;
  }

  /**
   * Get number of detected failure modes
   * @returns {number}
   */
  getDetectedModeCount() {
    return this.detectedModes.size;
  }
}

module.exports = { FailureModeDetector };
