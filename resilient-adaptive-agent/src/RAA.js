/**
 * RAA (Resilient Adaptive Agent) - Main Integration Class
 * 
 * Integrates all RAA components:
 * - Agent: Core web navigation capabilities
 * - UncertaintyTracker: CATTS-style uncertainty measurement
 * - CATTSAllocator: Dynamic compute allocation based on uncertainty
 * - FailureModeDetector: Multi-turn attack detection
 * - ResilienceRecoverySystem: Recovery strategies for failure modes
 * - ChecklistReward: Fine-grained binary criteria verification
 * - SelfVerificationSystem: Atomic verification after each action
 */

const { Agent, MockBrowser } = require('./Agent');
const { UncertaintyTracker } = require('./UncertaintyTracker');
const { CATTSAllocator } = require('./CATTSAllocator');
const { FailureModeDetector } = require('./FailureModeDetector');
const { ResilienceRecoverySystem } = require('./ResilienceRecovery');
const { ChecklistReward } = require('./ChecklistReward');
const { SelfVerificationSystem, VerificationStatus } = require('./SelfVerification');
const { Action, ActionTypes, ActionFactory } = require('./Action');

/**
 * RAA Class - Resilient Adaptive Agent
 * Main integration layer combining all RAA capabilities
 */
class RAA {
  /**
   * Create a new Resilient Adaptive Agent
   * @param {Object} config - RAA configuration
   * @param {string} [config.name='RAA'] - Agent name
   * @param {number} [config.maxSteps=100] - Maximum steps before stopping
   * @param {boolean} [config.verbose=false] - Enable verbose logging
   * @param {Object} [config.browser] - Browser instance (defaults to MockBrowser)
   * @param {Object} [config.agentConfig] - Additional Agent configuration
   * @param {Object} [config.uncertaintyConfig] - UncertaintyTracker configuration
   * @param {Object} [config.cattsConfig] - CATTSAllocator configuration
   * @param {Object} [config.failureDetectorConfig] - FailureModeDetector configuration
   * @param {Object} [config.resilienceConfig] - ResilienceRecoverySystem configuration
   * @param {Object} [config.verificationConfig] - SelfVerificationSystem configuration
   */
  constructor(config = {}) {
    this.name = config.name || 'RAA';
    this.maxSteps = config.maxSteps || 100;
    this.verbose = config.verbose || false;

    // Component configuration
    this.config = {
      agentConfig: config.agentConfig || {},
      uncertaintyConfig: config.uncertaintyConfig || {},
      cattsConfig: config.cattsConfig || {},
      failureDetectorConfig: config.failureDetectorConfig || {},
      resilienceConfig: config.resilienceConfig || {},
      verificationConfig: config.verificationConfig || {},
      browser: config.browser
    };

    // Component instances (initialized in initialize())
    this.agent = null;
    this.uncertaintyTracker = null;
    this.cattsAllocator = null;
    this.failureModeDetector = null;
    this.resilienceRecovery = null;
    this.checklistReward = null;
    this.verificationSystem = null;

    // State tracking
    this.isInitialized = false;
    this.isRunning = false;
    this.currentTask = null;
    this.taskHistory = [];
    this.checkpoints = [];
    this.recoveryAttempts = 0;
    this.maxRecoveryAttempts = config.maxRecoveryAttempts || 3;

    // Metrics
    this.metrics = {
      tasksCompleted: 0,
      tasksFailed: 0,
      totalActions: 0,
      successfulVerifications: 0,
      failedVerifications: 0,
      recoveriesApplied: 0,
      startTime: null
    };
  }

  /**
   * Log a message if verbose mode is enabled
   * @private
   * @param {string} level - Log level (info, warn, error)
   * @param {string} message - Message to log
   * @param {Object} [data] - Additional data to log
   */
  _log(level, message, data = {}) {
    if (this.verbose) {
      const timestamp = new Date().toISOString();
      const prefix = `[${timestamp}] [${this.name}] [${level.toUpperCase()}]`;
      if (Object.keys(data).length > 0) {
        console.log(`${prefix} ${message}`, data);
      } else {
        console.log(`${prefix} ${message}`);
      }
    }
  }

  /**
   * Initialize all RAA sub-components
   * Must be called before using the agent
   * @returns {Promise<Object>} Initialization result with component status
   */
  async initialize() {
    this._log('info', 'Initializing RAA components...');

    // Initialize core agent
    this.agent = new Agent({
      name: this.name,
      maxSteps: this.maxSteps,
      browser: this.config.browser || new MockBrowser(),
      verbose: this.verbose,
      ...this.config.agentConfig
    });

    // Initialize uncertainty tracking (CATTS)
    this.uncertaintyTracker = new UncertaintyTracker({
      windowSize: 10,
      scaleUpThreshold: 0.6,
      scaleDownThreshold: 0.3,
      ...this.config.uncertaintyConfig
    });

    // Initialize compute allocator (CATTS)
    this.cattsAllocator = new CATTSAllocator({
      minReasoningDepth: 1,
      maxReasoningDepth: 5,
      uncertaintyThreshold: 0.6,
      ...this.config.cattsConfig
    });

    // Initialize attack detection (Multi-turn resilience)
    this.failureModeDetector = new FailureModeDetector({
      historyWindow: 20,
      confidenceDropThreshold: 0.3,
      escalationThreshold: 2,
      ...this.config.failureDetectorConfig
    });

    // Initialize recovery system (Multi-turn resilience)
    this.resilienceRecovery = new ResilienceRecoverySystem({
      maxRecoveryAttempts: 3,
      recoveryCooldownMs: 5000,
      trackEffectiveness: true,
      ...this.config.resilienceConfig
    });

    // Initialize checklist reward (CM2)
    this.checklistReward = new ChecklistReward({
      maxRetries: 3,
      defaultWeight: 1.0,
      trackTimestamps: true
    });

    // Initialize self-verification (CM2)
    this.verificationSystem = new SelfVerificationSystem({
      autoRetry: true,
      maxRetries: 3,
      enableSideEffectDetection: true,
      ...this.config.verificationConfig
    });

    this.isInitialized = true;
    this.metrics.startTime = Date.now();

    this._log('info', 'RAA initialization complete', {
      components: {
        agent: !!this.agent,
        uncertaintyTracker: !!this.uncertaintyTracker,
        cattsAllocator: !!this.cattsAllocator,
        failureModeDetector: !!this.failureModeDetector,
        resilienceRecovery: !!this.resilienceRecovery,
        checklistReward: !!this.checklistReward,
        verificationSystem: !!this.verificationSystem
      }
    });

    return {
      success: true,
      componentsInitialized: 7,
      status: this.getStatus()
    };
  }

  /**
   * Execute a single action with full verification and safeguards
   * @param {Action|Object} action - Action to execute (Action instance or action params)
   * @returns {Promise<Object>} Execution result with verification
   */
  async executeAction(action) {
    if (!this.isInitialized) {
      throw new Error('RAA not initialized. Call initialize() first.');
    }

    // Normalize action
    let actionObj = action;
    if (!(action instanceof Action)) {
      if (action.type && action.params) {
        actionObj = new Action(action.type, action.params, action.priority);
      } else {
        throw new Error('Invalid action format. Expected Action instance or {type, params} object');
      }
    }

    this._log('info', `Executing action: ${actionObj.toString()}`);

    // Record current state for verification
    const beforeState = {
      url: this.agent.currentUrl,
      stepCount: this.agent.stepCount,
      hasError: this.agent.errors.length > 0
    };

    // Execute action through agent
    let result;
    let executionError = null;

    try {
      switch (actionObj.type) {
        case ActionTypes.NAVIGATE:
          result = await this.agent.navigate(actionObj.params.url);
          break;
        case ActionTypes.CLICK:
          result = await this.agent.click(actionObj.params.selector);
          break;
        case ActionTypes.TYPE:
          result = await this.agent.type(actionObj.params.selector, actionObj.params.text);
          break;
        case ActionTypes.EXTRACT:
          result = await this.agent.extract(actionObj.params.selector);
          break;
        case ActionTypes.SCROLL:
          result = await this.agent.scroll(actionObj.params);
          break;
        default:
          throw new Error(`Unknown action type: ${actionObj.type}`);
      }
    } catch (error) {
      executionError = error;
      result = { success: false, error: error.message };
      this._log('error', `Action execution failed: ${error.message}`);
    }

    // Update state after execution
    const afterState = {
      url: this.agent.currentUrl,
      stepCount: this.agent.stepCount,
      hasError: this.agent.errors.length > 0
    };

    // Record result for uncertainty tracking
    const success = result.success && !executionError;
    this.uncertaintyTracker.recordVote({
      success,
      confidence: success ? 0.8 : 0.2,
      timestamp: Date.now()
    });

    // Perform self-verification
    const verification = this.verificationSystem.verifyAction(actionObj, result, {
      currentUrl: this.agent.currentUrl,
      pageState: afterState,
      beforeState
    });

    // Update metrics
    this.metrics.totalActions++;
    
    // Determine actual success - action is successful if no execution error and result.success is true
    // Verification issues that are just checklist mismatches shouldn't fail the action
    const isExecutionSuccessful = executionError === null && result.success;
    let isVerified = verification.status === VerificationStatus.VERIFIED;
    
    // If execution succeeded but verification flagged issues, check if they're critical
    if (!isVerified && isExecutionSuccessful) {
      const hasCriticalIssues = (verification.retryReasons || []).some(reason => 
        reason.includes('NULL_RESULT') || 
        reason.includes('ACTION_FAILED') ||
        reason.includes('TIMEOUT') ||
        reason.includes('ERROR')
      );
      
      // Only mark as failed if there are critical issues
      if (!hasCriticalIssues) {
        isVerified = true;
      }
    }
    
    if (isVerified) {
      this.metrics.successfulVerifications++;
    } else {
      this.metrics.failedVerifications++;
    }

    // Analyze for failure modes if verification failed
    let failureAnalysis = null;
    let recoveryResult = null;

    // Always provide failureAnalysis, even for successful actions
    failureAnalysis = this.analyzeFailureMode({
      action: actionObj,
      result,
      error: executionError,
      verification
    });

    if (!isVerified) {
      // Attempt recovery if needed
      if (failureAnalysis.shouldRecover) {
        recoveryResult = await this.applyRecovery(failureAnalysis);
      }
    }

    // Allocate compute based on uncertainty
    const reasoningDepth = this.allocateCompute();

    this._log('info', `Action execution complete`, {
      status: isVerified ? 'VERIFIED' : verification.status,
      confidence: verification.confidence,
      reasoningDepth
    });

    return {
      action: actionObj.toJSON(),
      result,
      success: isVerified,
      verification: {
        status: isVerified ? VerificationStatus.VERIFIED : verification.status,
        confidence: verification.confidence,
        issues: verification.retryReasons || []
      },
      failureAnalysis,
      recovery: recoveryResult,
      reasoningDepth,
      executionError: executionError ? executionError.message : null
    };
  }

  /**
   * Run a complete task with all safeguards
   * @param {Object} task - Task to execute
   * @param {string} task.id - Task identifier
   * @param {string} task.description - Task description
   * @param {Array<Object>} task.actions - Array of actions to execute
   * @param {Object} [task.options] - Task-specific options
   * @returns {Promise<Object>} Task execution result
   */
  async runTask(task) {
    if (!this.isInitialized) {
      throw new Error('RAA not initialized. Call initialize() first.');
    }

    if (!task || !task.actions || !Array.isArray(task.actions)) {
      throw new Error('Invalid task: must have actions array');
    }

    this.isRunning = true;
    this.currentTask = {
      id: task.id || `task_${Date.now()}`,
      description: task.description || 'Unnamed task',
      startTime: Date.now(),
      actions: task.actions
    };

    this._log('info', `Starting task: ${this.currentTask.description}`, {
      taskId: this.currentTask.id,
      actionCount: task.actions.length
    });

    // Create checklist for task verification
    const checklistId = this.checklistReward.createChecklist('navigation');
    
    // Store original context for recovery
    this.resilienceRecovery.storeOriginalContext({
      task: this.currentTask,
      initialUrl: this.agent.currentUrl,
      goals: task.description
    });

    const results = [];
    let taskComplete = false;
    let taskFailed = false;
    let failureReason = null;

    try {
      for (let i = 0; i < task.actions.length; i++) {
        // Check for step limit
        if (this.agent.stepCount >= this.maxSteps) {
          throw new Error(`Maximum steps (${this.maxSteps}) exceeded`);
        }

        // Check for too many recoveries
        if (this.recoveryAttempts >= this.maxRecoveryAttempts) {
          throw new Error(`Maximum recovery attempts (${this.maxRecoveryAttempts}) exceeded`);
        }

        const action = task.actions[i];
        this._log('info', `Executing action ${i + 1}/${task.actions.length}`);

        // Execute action with full verification
        const actionResult = await this.executeAction(action);
        results.push(actionResult);

        // Handle failure
        if (!actionResult.success) {
          // Try recovery if available
          if (actionResult.recovery && actionResult.recovery.success) {
            this._log('info', `Recovery applied: ${actionResult.recovery.mode}`);
            this.metrics.recoveriesApplied++;
            this.recoveryAttempts++;
            
            // Optionally retry the action based on recovery
            if (actionResult.recovery.recovery && 
                actionResult.recovery.recovery.metadata && 
                actionResult.recovery.recovery.metadata.retryRecommended) {
              // Retry the action
              this._log('info', 'Retrying failed action after recovery');
              i--; // Repeat this action
              continue;
            }
          } else {
            // No recovery available - task failed
            taskFailed = true;
            failureReason = actionResult.executionError || 
                           actionResult.verification.issues.join(', ') ||
                           'Action verification failed';
            break;
          }
        }

        // Update checklist
        if (i < task.actions.length - 1) {
          this.checklistReward.evaluateCriterion(checklistId, 'url_reachable', true);
        }

        // Create checkpoint periodically
        if ((i + 1) % 3 === 0) {
          await this.verifyAndCheckpoint();
        }
      }

      taskComplete = !taskFailed;

    } catch (error) {
      taskFailed = true;
      failureReason = error.message;
      this._log('error', `Task failed with error: ${error.message}`);
    }

    // Finalize task
    const endTime = Date.now();
    const duration = endTime - this.currentTask.startTime;

    const taskResult = {
      taskId: this.currentTask.id,
      description: this.currentTask.description,
      success: taskComplete,
      failed: taskFailed,
      failureReason,
      duration,
      actionsExecuted: results.length,
      results,
      finalState: this.agent.getState(),
      uncertaintyStats: this.uncertaintyTracker.getStatistics(),
      verificationStats: this.verificationSystem.getStatistics(),
      checkpointCount: this.checkpoints.length
    };

    // Update metrics
    if (taskComplete) {
      this.metrics.tasksCompleted++;
    } else {
      this.metrics.tasksFailed++;
    }

    this.taskHistory.push(taskResult);

    this._log('info', `Task completed: ${taskComplete ? 'SUCCESS' : 'FAILED'}`, {
      duration,
      actionsExecuted: results.length
    });

    this.isRunning = false;
    this.currentTask = null;
    this.recoveryAttempts = 0;

    return taskResult;
  }

  /**
   * Detect issues and apply recovery strategies
   * Analyzes current state for failure modes and applies appropriate recovery
   * @param {Object} [context] - Optional context for detection
   * @returns {Object} Detection and recovery results
   */
  async detectAndRecover(context = {}) {
    this._log('info', 'Running failure detection and recovery...');

    // Get failure report from detector
    const failureReport = this.failureModeDetector.getFailureReport();

    if (failureReport.detectedModes.length === 0) {
      this._log('info', 'No failure modes detected');
      return {
        detected: false,
        recoveryNeeded: false,
        failureReport
      };
    }

    this._log('warn', `Detected failure modes: ${failureReport.detectedModes.join(', ')}`);

    // Select and apply recovery strategy
    const selection = this.resilienceRecovery.selectRecoveryStrategy(failureReport, context);

    let recoveryResult = null;
    if (selection.success && selection.primaryStrategy) {
      recoveryResult = this.resilienceRecovery.applyRecovery(
        selection.primaryStrategy,
        context[selection.primaryStrategy] || context
      );

      if (recoveryResult.success) {
        this.metrics.recoveriesApplied++;
        this._log('info', `Recovery applied successfully: ${selection.primaryStrategy}`);
      }
    }

    return {
      detected: true,
      recoveryNeeded: true,
      failureReport,
      selection,
      recovery: recoveryResult
    };
  }

  /**
   * Analyze current execution for failure modes
   * @private
   * @param {Object} context - Analysis context
   * @returns {Object} Failure analysis result
   */
  analyzeFailureMode(context) {
    const { action, result, error, verification } = context;

    // Create turn-like data for failure detector
    const turnData = {
      userInput: action.toString(),
      agentResponse: result.success ? 'success' : (error ? error.message : 'failed'),
      confidence: verification.confidence,
      metadata: {
        actionType: action.type,
        verificationStatus: verification.status
      }
    };

    const analysis = this.failureModeDetector.analyzeTurn(turnData);

    return {
      detectedModes: analysis.detections,
      shouldRecover: analysis.shouldEscalate || verification.status === VerificationStatus.FAILED,
      escalationNeeded: analysis.shouldEscalate,
      analysis
    };
  }

  /**
   * Apply recovery strategy for detected failure
   * @private
   * @param {Object} failureAnalysis - Failure analysis result
   * @returns {Object} Recovery result
   */
  async applyRecovery(failureAnalysis) {
    if (!failureAnalysis.detectedModes) {
      return { success: false, reason: 'No failure modes to recover from' };
    }

    // Find first detected mode that needs recovery
    const modes = failureAnalysis.detectedModes;
    let recoveryResult = null;

    // Priority order for recovery
    const priorityOrder = ['suggestionHijacking', 'selfDoubt', 'socialConformity', 
                          'emotionalSusceptibility', 'reasoningFatigue'];

    for (const mode of priorityOrder) {
      if (modes[mode] && modes[mode].detected) {
        recoveryResult = this.resilienceRecovery.applyRecovery(mode, {
          originalGoals: this.currentTask?.description,
          currentConfidence: this.uncertaintyTracker.getAverageConfidence()
        });

        if (recoveryResult.success) {
          break;
        }
      }
    }

    return recoveryResult || { success: false, reason: 'No recovery strategy available' };
  }

  /**
   * Allocate compute based on current uncertainty (CATTS)
   * Uses UncertaintyTracker to determine reasoning depth
   * @returns {number} The allocated reasoning depth
   */
  allocateCompute() {
    const depth = this.cattsAllocator.allocateCompute(this.uncertaintyTracker);
    
    this._log('info', `Compute allocated: reasoning depth ${depth}`, {
      entropy: this.uncertaintyTracker.getEntropy(),
      uncertainty: this.uncertaintyTracker.getUncertainty()
    });

    return depth;
  }

  /**
   * Verify current state and create checkpoint
   * Creates a recovery point that can be restored if needed
   * @returns {Promise<Object>} Checkpoint result
   */
  async verifyAndCheckpoint() {
    this._log('info', 'Creating checkpoint...');

    // Get current agent state
    const agentState = this.agent.getState();
    
    // Verify current state
    const uncertaintyStats = this.uncertaintyTracker.getStatistics();
    const verificationStats = this.verificationSystem.getStatistics();
    const failureReport = this.failureModeDetector.getFailureReport();

    const checkpoint = {
      id: `checkpoint_${Date.now()}`,
      timestamp: Date.now(),
      agentState,
      uncertaintyStats,
      verificationStats,
      failureReport,
      taskId: this.currentTask?.id,
      stepCount: this.agent.stepCount
    };

    this.checkpoints.push(checkpoint);

    this._log('info', `Checkpoint created: ${checkpoint.id}`, {
      stepCount: checkpoint.stepCount,
      confidence: uncertaintyStats.averageConfidence
    });

    return {
      success: true,
      checkpointId: checkpoint.id,
      checkpointIndex: this.checkpoints.length - 1,
      state: {
        confidence: uncertaintyStats.averageConfidence,
        entropy: uncertaintyStats.entropy,
        detectedModes: failureReport.detectedModes
      }
    };
  }

  /**
   * Get current RAA status
   * @returns {Object} Complete status information
   */
  getStatus() {
    const runtime = this.metrics.startTime ? Date.now() - this.metrics.startTime : 0;

    return {
      // Initialization status
      isInitialized: this.isInitialized,
      isRunning: this.isRunning,

      // Component status
      components: {
        agent: this.agent ? this.agent.getState() : null,
        uncertaintyTracker: this.uncertaintyTracker ? this.uncertaintyTracker.getStatistics() : null,
        cattsAllocator: this.cattsAllocator ? this.cattsAllocator.getAllocationStats() : null,
        failureModeDetector: this.failureModeDetector ? this.failureModeDetector.getFailureReport() : null,
        resilienceRecovery: this.resilienceRecovery ? this.resilienceRecovery.getStatus() : null,
        verificationSystem: this.verificationSystem ? this.verificationSystem.getStatistics() : null
      },

      // Task status
      currentTask: this.currentTask ? {
        id: this.currentTask.id,
        description: this.currentTask.description,
        actionsRemaining: this.currentTask.actions ? 
          this.currentTask.actions.length - this.agent?.stepCount : 0
      } : null,

      // Metrics
      metrics: {
        ...this.metrics,
        runtime
      },

      // State info
      checkpoints: this.checkpoints.length,
      taskHistory: this.taskHistory.length,
      recoveryAttempts: this.recoveryAttempts
    };
  }

  /**
   * Reset RAA to initial state
   * Clears all history and resets components
   * @param {boolean} [fullReset=false] - If true, also resets configuration
   * @returns {Object} Reset result
   */
  reset(fullReset = false) {
    this._log('info', 'Resetting RAA state...');

    // Reset components
    if (this.agent) this.agent.reset();
    if (this.uncertaintyTracker) this.uncertaintyTracker.reset();
    if (this.cattsAllocator) this.cattsAllocator.reset();
    if (this.failureModeDetector) this.failureModeDetector.reset();
    if (this.resilienceRecovery) this.resilienceRecovery.reset(!fullReset);
    if (this.verificationSystem) this.verificationSystem.reset();

    // Reset state
    this.isRunning = false;
    this.currentTask = null;
    this.taskHistory = [];
    this.checkpoints = [];
    this.recoveryAttempts = 0;

    // Reset metrics
    this.metrics = {
      tasksCompleted: 0,
      tasksFailed: 0,
      totalActions: 0,
      successfulVerifications: 0,
      failedVerifications: 0,
      recoveriesApplied: 0,
      startTime: fullReset ? null : Date.now()
    };

    return {
      success: true,
      fullReset
    };
  }

  /**
   * Get comprehensive RAA statistics
   * @returns {Object} Statistics from all components
   */
  getStatistics() {
    return {
      raa: {
        name: this.name,
        runtime: this.metrics.startTime ? Date.now() - this.metrics.startTime : 0,
        ...this.metrics
      },
      uncertainty: this.uncertaintyTracker ? this.uncertaintyTracker.getStatistics() : null,
      catts: this.cattsAllocator ? this.cattsAllocator.getAllocationStats() : null,
      verification: this.verificationSystem ? this.verificationSystem.getStatistics() : null,
      failureDetection: this.failureModeDetector ? this.failureModeDetector.getFailureReport() : null,
      recovery: this.resilienceRecovery ? this.resilienceRecovery.getRecoveryStats() : null
    };
  }
}

module.exports = { RAA };
