/**
 * SelfVerificationSystem - Atomic verification after each action
 * 
 * Based on CM2 paper concepts:
 * - Verification as atomic operation after each step
 * - Immediate detection of failures
 * - Retry with adjusted strategy if needed
 * 
 * Integrates with:
 * - ChecklistReward for fine-grained binary criteria verification
 * - FailureModeDetector to check for adversarial manipulation patterns
 */

const { ChecklistReward } = require('./ChecklistReward');
const { FailureModeDetector } = require('./FailureModeDetector');
const { ActionTypes } = require('./Action');

/**
 * Verification status constants
 */
const VerificationStatus = {
  PENDING: 'PENDING',
  VERIFIED: 'VERIFIED',
  FAILED: 'FAILED',
  PARTIAL: 'PARTIAL',
  ERROR: 'ERROR'
};

/**
 * SelfVerificationSystem class
 * Performs atomic verification after each action execution
 */
class SelfVerificationSystem {
  /**
   * Create a new SelfVerificationSystem
   * @param {Object} config - Configuration options
   * @param {Object} [config.checklistConfig] - Configuration for ChecklistReward
   * @param {Object} [config.failureDetectorConfig] - Configuration for FailureModeDetector
   * @param {boolean} [config.autoRetry=true] - Whether to automatically suggest retries
   * @param {number} [config.maxRetries=3] - Maximum number of retries
   * @param {number} [config.timeoutMs=30000] - Default timeout for verification
   * @param {boolean} [config.enableSideEffectDetection=true] - Enable side effect checking
   * @param {Function} [config.logger] - Logger function for verification events
   */
  constructor(config = {}) {
    this.config = {
      autoRetry: config.autoRetry !== false,
      maxRetries: config.maxRetries || 3,
      timeoutMs: config.timeoutMs || 30000,
      enableSideEffectDetection: config.enableSideEffectDetection !== false,
      logger: config.logger || null,
      ...config
    };

    // Initialize ChecklistReward for fine-grained verification
    this.checklistReward = new ChecklistReward(config.checklistConfig || {});

    // Initialize FailureModeDetector for detecting issues
    this.failureDetector = new FailureModeDetector(config.failureDetectorConfig || {});

    // Verification history
    this.verificationHistory = [];
    this.verificationResults = new Map();
    this.verificationCounter = 0;

    // Side effect tracking
    this.sideEffects = new Map();
    this.expectedState = new Map();
  }

  /**
   * Main entry point: Verify a single action after execution
   * @param {Action} action - The action that was executed
   * @param {Object} result - The result of the action execution
   * @param {Object} context - Additional context for verification
   * @param {string} context.currentUrl - Current URL after action
   * @param {Object} context.pageState - Current page state
   * @param {Object} context.beforeState - State before action execution
   * @param {Object} context.metadata - Additional metadata
   * @returns {Object} Verification result
   */
  verifyAction(action, result, context = {}) {
    // Handle null or undefined action
    if (!action) {
      return {
        id: `verify_${++this.verificationCounter}_error`,
        status: VerificationStatus.ERROR,
        error: 'Action is null or undefined',
        shouldRetry: false,
        preConditions: { passed: false, issues: [{ type: 'NULL_ACTION', message: 'Action is null or undefined' }] },
        postConditions: { passed: false, issues: [] },
        sideEffects: { passed: true, issues: [], detectedEffects: [] },
        checklist: { used: false },
        failureCheck: { analyzed: false },
        confidence: 0
      };
    }

    const verificationId = `verify_${++this.verificationCounter}_${action.id || Date.now()}`;
    
    this._log('info', `Starting verification for action: ${action.type}`, { verificationId });

    // Step 1: Pre-condition checks
    const preConditionCheck = this._checkPreConditions(action, context);
    
    // Step 2: Post-condition checks
    const postConditionCheck = this._checkPostConditions(action, result, context);
    
    // Step 3: Side-effect checks
    const sideEffectCheck = this.config.enableSideEffectDetection 
      ? this._checkSideEffects(action, result, context)
      : { passed: true, issues: [] };

    // Step 4: Checklist-based fine-grained verification
    const checklistVerification = this._verifyWithChecklist(action, result, context);

    // Step 5: Check for failure modes/issues
    const failureCheck = this._checkForFailureModes(action, result, context);

    // Compile verification result
    const verificationResult = {
      id: verificationId,
      actionId: action.id,
      actionType: action.type,
      timestamp: Date.now(),
      status: this._determineStatus(preConditionCheck, postConditionCheck, sideEffectCheck, checklistVerification, failureCheck),
      preConditions: preConditionCheck,
      postConditions: postConditionCheck,
      sideEffects: sideEffectCheck,
      checklist: checklistVerification,
      failureCheck: failureCheck,
      shouldRetry: false,
      retryReasons: [],
      confidence: this._calculateConfidence(preConditionCheck, postConditionCheck, checklistVerification)
    };

    // Determine if retry is needed
    verificationResult.shouldRetry = this._shouldRetryBasedOnVerification(verificationResult);
    verificationResult.retryReasons = this._getRetryReasons(verificationResult);

    // Store result
    this.verificationResults.set(verificationId, verificationResult);
    this.verificationHistory.push(verificationResult);

    this._log('info', `Verification complete: ${verificationResult.status}`, { 
      verificationId, 
      status: verificationResult.status,
      shouldRetry: verificationResult.shouldRetry 
    });

    return verificationResult;
  }

  /**
   * Verify navigation was successful
   * @param {string} url - Target URL
   * @param {Object} result - Navigation result
   * @param {boolean} result.success - Whether navigation succeeded
   * @param {string} result.currentUrl - Actual URL after navigation
   * @param {string} result.title - Page title after navigation
   * @param {number} result.statusCode - HTTP status code
   * @param {Object} context - Additional context
   * @returns {Object} Navigation verification result
   */
  verifyNavigation(url, result, context = {}) {
    const action = {
      id: `nav_verify_${Date.now()}`,
      type: ActionTypes.NAVIGATE,
      params: { url }
    };

    const verification = this.verifyAction(action, result, {
      ...context,
      targetUrl: url,
      actionSpecific: true
    });

    // Add navigation-specific checks
    const navChecks = {
      urlMatch: this._checkUrlMatch(url, result?.currentUrl),
      pageLoaded: result?.success && !!result.title,
      statusOk: result?.statusCode >= 200 && result?.statusCode < 400,
      redirectDetected: this._detectRedirect(url, result?.currentUrl),
      errorPage: this._detectErrorPage(result)
    };

    verification.navigationChecks = navChecks;
    
    // Expose key checks at top level for convenience
    verification.urlMatch = navChecks.urlMatch;
    verification.pageLoaded = navChecks.pageLoaded;
    verification.error = result?.error;
    
    // Update status based on navigation checks
    if (!navChecks.urlMatch || !navChecks.pageLoaded || navChecks.errorPage) {
      verification.status = VerificationStatus.FAILED;
      verification.shouldRetry = true;
    }

    return verification;
  }

  /**
   * Verify extraction was correct
   * @param {string} selector - CSS selector used for extraction
   * @param {Object} extractedData - The extracted data
   * @param {Object} context - Additional context
   * @param {Object} context.expectedSchema - Expected data schema
   * @param {Array} context.expectedFields - Expected fields
   * @returns {Object} Extraction verification result
   */
  verifyExtraction(selector, extractedData, context = {}) {
    const action = {
      id: `extract_verify_${Date.now()}`,
      type: ActionTypes.EXTRACT,
      params: { selector }
    };

    const result = {
      success: extractedData !== null && extractedData !== undefined,
      data: extractedData,
      selector
    };

    const verification = this.verifyAction(action, result, {
      ...context,
      actionSpecific: true
    });

    // Add extraction-specific checks
    const extractChecks = {
      elementFound: result.success,
      dataNotEmpty: this._checkDataNotEmpty(extractedData),
      schemaValid: context.expectedSchema 
        ? this._validateSchema(extractedData, context.expectedSchema)
        : true,
      fieldsPresent: context.expectedFields
        ? this._checkFieldsPresent(extractedData, context.expectedFields)
        : true,
      dataTypeCorrect: this._checkDataType(extractedData, context.expectedType)
    };

    verification.extractionChecks = extractChecks;
    
    // Expose key checks at top level for convenience
    verification.elementFound = extractChecks.elementFound;
    verification.dataExtracted = extractChecks.dataNotEmpty;
    verification.extractedData = extractedData;

    // Update status based on extraction checks
    if (!extractChecks.elementFound || !extractChecks.dataNotEmpty) {
      verification.status = VerificationStatus.FAILED;
      verification.shouldRetry = true;
    } else if (!extractChecks.schemaValid || !extractChecks.fieldsPresent) {
      verification.status = VerificationStatus.PARTIAL;
    }

    return verification;
  }

  /**
   * Verify form submission worked
   * @param {Object} formData - Data submitted in the form
   * @param {Object} result - Submission result
   * @param {boolean} result.success - Whether submission succeeded
   * @param {string} result.confirmationMessage - Confirmation message if any
   * @param {Object} result.errors - Form errors if submission failed
   * @param {string} result.redirectUrl - URL after submission
   * @param {Object} context - Additional context
   * @returns {Object} Form submission verification result
   */
  verifyFormSubmission(formData, result, context = {}) {
    const action = {
      id: `form_verify_${Date.now()}`,
      type: ActionTypes.CLICK, // Forms are typically submitted via click
      params: { formData }
    };

    const verification = this.verifyAction(action, result, {
      ...context,
      actionSpecific: true
    });

    // Add form-specific checks
    const formChecks = {
      allFieldsSubmitted: this._checkFieldsSubmitted(formData, context.requiredFields || []),
      noValidationErrors: !result.errors || Object.keys(result.errors).length === 0,
      confirmationReceived: !!result.confirmationMessage || result.success,
      redirectAfterSubmit: !!result.redirectUrl,
      errorMessages: result.errors || {}
    };

    verification.formChecks = formChecks;

    // Update status based on form checks
    if (!formChecks.noValidationErrors || (!formChecks.confirmationReceived && !result.success)) {
      verification.status = VerificationStatus.FAILED;
      verification.shouldRetry = true;
    } else if (!formChecks.redirectAfterSubmit && context.expectRedirect) {
      verification.status = VerificationStatus.PARTIAL;
    }

    return verification;
  }

  /**
   * Get result of a specific verification
   * @param {string} verificationId - Verification identifier
   * @returns {Object|null} Verification result or null if not found
   */
  getVerificationResult(verificationId) {
    const result = this.verificationResults.get(verificationId);
    return result ? { ...result } : null;
  }

  /**
   * Check if action should be retried based on verification
   * @param {string} verificationId - Verification identifier
   * @returns {boolean} True if action should be retried
   */
  shouldRetry(verificationId) {
    const result = this.verificationResults.get(verificationId);
    if (!result) {
      return false;
    }

    // Check retry count
    const retryCount = this._getRetryCount(result.actionId);
    if (retryCount >= this.config.maxRetries) {
      return false;
    }

    return result.shouldRetry;
  }

  /**
   * Get history of all verifications
   * @param {Object} options - Query options
   * @param {string} [options.actionType] - Filter by action type
   * @param {string} [options.status] - Filter by status
   * @param {number} [options.limit] - Limit number of results
   * @returns {Array} Verification history
   */
  getVerificationHistory(options = {}) {
    let history = [...this.verificationHistory];

    if (options.actionType) {
      history = history.filter(v => v.actionType === options.actionType);
    }

    if (options.status) {
      history = history.filter(v => v.status === options.status);
    }

    if (options.limit) {
      history = history.slice(-options.limit);
    }

    return history.map(v => ({ ...v }));
  }

  /**
   * Get verification statistics
   * @returns {Object} Statistics about verifications
   */
  getStatistics() {
    const history = this.verificationHistory;
    const total = history.length;

    if (total === 0) {
      return { total: 0 };
    }

    const byStatus = {};
    const byType = {};
    let retryAttempts = 0;
    let successfulRetries = 0;

    history.forEach(v => {
      // Count by status
      byStatus[v.status] = (byStatus[v.status] || 0) + 1;

      // Count by action type
      byType[v.actionType] = (byType[v.actionType] || 0) + 1;

      // Count retries
      if (v.shouldRetry) {
        retryAttempts++;
      }
    });

    // Calculate success rate of retries
    const failedVerifications = history.filter(v => v.status === VerificationStatus.FAILED);
    const laterSuccess = failedVerifications.filter(fv => {
      const later = history.find(v => v.actionId === fv.actionId && v.timestamp > fv.timestamp && v.status === VerificationStatus.VERIFIED);
      return !!later;
    });
    successfulRetries = laterSuccess.length;

    return {
      total,
      byStatus,
      byType,
      successRate: (byStatus[VerificationStatus.VERIFIED] || 0) / total,
      partialRate: (byStatus[VerificationStatus.PARTIAL] || 0) / total,
      failureRate: (byStatus[VerificationStatus.FAILED] || 0) / total,
      retryAttempts,
      successfulRetries,
      retrySuccessRate: retryAttempts > 0 ? successfulRetries / retryAttempts : 0,
      averageConfidence: history.reduce((sum, v) => sum + v.confidence, 0) / total
    };
  }

  /**
   * Reset verification system state
   */
  reset() {
    this.verificationHistory = [];
    this.verificationResults.clear();
    this.sideEffects.clear();
    this.expectedState.clear();
    this.verificationCounter = 0;
    
    if (this.failureDetector) {
      this.failureDetector.reset();
    }
  }

  // ==================== PRIVATE METHODS ====================

  /**
   * Check pre-conditions before action
   * @private
   */
  _checkPreConditions(action, context) {
    const checks = {
      passed: true,
      issues: []
    };

    // Check if action is valid
    if (!action || !action.type) {
      checks.passed = false;
      checks.issues.push({ type: 'INVALID_ACTION', message: 'Action is invalid or missing type' });
      return checks;
    }

    // Check if required params are present
    if (action.params) {
      switch (action.type) {
        case ActionTypes.NAVIGATE:
          if (!action.params.url) {
            checks.passed = false;
            checks.issues.push({ type: 'MISSING_PARAM', message: 'URL parameter is required for navigation' });
          }
          break;
        case ActionTypes.CLICK:
        case ActionTypes.EXTRACT:
          if (!action.params.selector) {
            checks.passed = false;
            checks.issues.push({ type: 'MISSING_PARAM', message: 'Selector parameter is required' });
          }
          break;
        case ActionTypes.TYPE:
          if (!action.params.selector || action.params.text === undefined) {
            checks.passed = false;
            checks.issues.push({ type: 'MISSING_PARAM', message: 'Selector and text parameters are required for TYPE' });
          }
          break;
      }
    }

    // Check page state if available
    if (context.beforeState) {
      if (context.beforeState.isErrorPage) {
        checks.passed = false;
        checks.issues.push({ type: 'ERROR_PAGE', message: 'Starting from an error page' });
      }
    }

    return checks;
  }

  /**
   * Check post-conditions after action
   * @private
   */
  _checkPostConditions(action, result, context) {
    const checks = {
      passed: true,
      issues: []
    };

    // Check if result is null or action failed
    if (!result) {
      checks.passed = false;
      checks.issues.push({ type: 'NULL_RESULT', message: 'Action returned null result' });
      return checks;
    }

    if (result.success === false) {
      checks.passed = false;
      checks.issues.push({ 
        type: 'ACTION_FAILED', 
        message: result?.error || 'Action execution failed' 
      });
      return checks;
    }

    // Action-specific post-condition checks
    switch (action.type) {
      case ActionTypes.NAVIGATE:
        if (result.currentUrl && action.params.url) {
          const urlMatch = this._checkUrlMatch(action.params.url, result.currentUrl);
          if (!urlMatch) {
            checks.issues.push({ 
              type: 'URL_MISMATCH', 
              message: `Expected ${action.params.url}, got ${result.currentUrl}` 
            });
          }
        }
        break;

      case ActionTypes.CLICK:
        // Check if element is still interactable (might indicate the click didn't work)
        if (result.elementStillPresent && result.expectedNavigation && !result.urlChanged) {
          checks.issues.push({ 
            type: 'CLICK_NO_EFFECT', 
            message: 'Element still present after click, expected navigation did not occur' 
          });
        }
        break;

      case ActionTypes.TYPE:
        // Check if text was actually entered
        if (result.actualValue !== undefined && result.actualValue !== action.params.text) {
          checks.issues.push({ 
            type: 'TEXT_MISMATCH', 
            message: `Expected "${action.params.text}", got "${result.actualValue}"` 
          });
        }
        break;

      case ActionTypes.EXTRACT:
        if (!result.data && result.success !== false) {
          checks.issues.push({ 
            type: 'NO_DATA_EXTRACTED', 
            message: 'Extraction succeeded but no data was returned' 
          });
        }
        break;
    }

    // Update passed status based on issues
    checks.passed = checks.issues.length === 0;

    return checks;
  }

  /**
   * Check for side effects
   * @private
   */
  _checkSideEffects(action, result, context) {
    const checks = {
      passed: true,
      issues: [],
      detectedEffects: []
    };

    // Handle null or undefined result
    if (!result) {
      checks.passed = false;
      checks.detectedEffects.push({
        type: 'NULL_RESULT',
        message: 'Action returned null or undefined result',
        severity: 'error'
      });
      return checks;
    }

    // Track unexpected state changes
    if (context.beforeState && context.pageState) {
      const before = context.beforeState;
      const after = context.pageState;

      // Check for unexpected URL changes (for non-navigation actions)
      if (action.type !== ActionTypes.NAVIGATE && before.url !== after.url) {
        checks.detectedEffects.push({
          type: 'UNEXPECTED_NAVIGATION',
          message: `URL changed unexpectedly from ${before.url} to ${after.url}`,
          severity: 'warning'
        });
      }

      // Check for error messages that appeared
      if (!before.hasError && after.hasError) {
        checks.detectedEffects.push({
          type: 'ERROR_APPEARED',
          message: 'Error message appeared after action',
          severity: 'error'
        });
        checks.passed = false;
      }

      // Check for console errors
      if (after.consoleErrors && after.consoleErrors.length > (before.consoleErrors?.length || 0)) {
        const newErrors = after.consoleErrors.slice(before.consoleErrors?.length || 0);
        checks.detectedEffects.push({
          type: 'CONSOLE_ERRORS',
          message: `New console errors: ${newErrors.length}`,
          errors: newErrors,
          severity: 'warning'
        });
      }
    }

    // Check for timeout
    if (result.timedOut) {
      checks.detectedEffects.push({
        type: 'TIMEOUT',
        message: 'Action timed out',
        severity: 'error'
      });
      checks.passed = false;
    }

    return checks;
  }

  /**
   * Verify using ChecklistReward for fine-grained criteria
   * @private
   */
  _verifyWithChecklist(action, result, context) {
    // Map action types to checklist templates
    const templateMap = {
      [ActionTypes.NAVIGATE]: 'navigation',
      [ActionTypes.EXTRACT]: 'extraction',
      [ActionTypes.CLICK]: 'form', // Often form-related
      [ActionTypes.TYPE]: 'form'
    };

    const templateName = templateMap[action.type];
    if (!templateName) {
      return { used: false, reason: 'No template for action type' };
    }

    // Create checklist
    const checklistId = this.checklistReward.createChecklist(templateName);
    
    // Evaluate criteria based on action and result
    const status = this.checklistReward.getChecklistStatus(checklistId);
    
    status.criteria.forEach(criterion => {
      let passed = false;

      switch (criterion.id) {
        case 'url_reachable':
          passed = result && result.success !== false && !result.timedOut;
          break;
        case 'page_loaded':
          passed = result && result.success && !!result.title && !result.isErrorPage;
          break;
        case 'correct_page':
          passed = action.type !== ActionTypes.NAVIGATE || 
                   this._checkUrlMatch(action.params.url, result?.currentUrl);
          break;
        case 'element_found':
          passed = result && result.success && result.elementFound !== false;
          break;
        case 'text_extracted':
          passed = result && result.success && !!result.data;
          break;
        case 'format_correct':
          passed = result && result.success && (!context.expectedSchema || 
                   this._validateSchema(result.data, context.expectedSchema));
          break;
        case 'fields_filled':
          passed = result && result.success && (!context.requiredFields || 
                   this._checkFieldsSubmitted(result.submittedData, context.requiredFields));
          break;
        case 'submit_successful':
          passed = result && result.success && !result.errors;
          break;
        case 'confirmation_received':
          passed = result && result.success && (!!result.confirmationMessage || result.redirectUrl);
          break;
      }

      this.checklistReward.evaluateCriterion(checklistId, criterion.id, passed);
    });

    const finalStatus = this.checklistReward.getChecklistStatus(checklistId);
    
    return {
      used: true,
      checklistId,
      reward: finalStatus.reward,
      passed: finalStatus.passed,
      failed: finalStatus.failed,
      isComplete: finalStatus.isComplete,
      criteria: finalStatus.criteria
    };
  }

  /**
   * Check for failure modes using FailureModeDetector
   * @private
   */
  _checkForFailureModes(action, result, context) {
    // Only analyze if we have turn-like data
    if (!context.turnData) {
      return { analyzed: false, reason: 'No turn data provided' };
    }

    const analysis = this.failureDetector.analyzeTurn(context.turnData);
    
    return {
      analyzed: true,
      shouldEscalate: analysis.shouldEscalate,
      detections: analysis.detections,
      detectedModes: this.failureDetector.getFailureReport().detectedModes
    };
  }

  /**
   * Determine overall verification status
   * @private
   */
  _determineStatus(pre, post, sideEffects, checklist, failureCheck) {
    // If pre-conditions failed, the action shouldn't have been attempted
    if (!pre.passed) {
      return VerificationStatus.ERROR;
    }

    // If side effects have null result or other critical issues
    const hasNullResult = sideEffects.detectedEffects.some(e => e.type === 'NULL_RESULT');
    if (hasNullResult) {
      return VerificationStatus.ERROR;
    }

    // If side effects are concerning
    if (!sideEffects.passed) {
      return VerificationStatus.FAILED;
    }

    // If post-conditions failed, the action didn't work
    if (!post.passed) {
      return VerificationStatus.FAILED;
    }

    // Check checklist results
    if (checklist.used) {
      if (checklist.reward === 1.0) {
        // All criteria passed - VERIFIED
      } else if (checklist.reward > 0) {
        return VerificationStatus.PARTIAL;
      } else {
        return VerificationStatus.FAILED;
      }
    }

    // Check failure mode detection
    if (failureCheck.analyzed && failureCheck.shouldEscalate) {
      return VerificationStatus.ERROR;
    }

    return VerificationStatus.VERIFIED;
  }

  /**
   * Calculate confidence score
   * @private
   */
  _calculateConfidence(pre, post, checklist) {
    let score = 1.0;

    // Reduce confidence for pre-condition issues
    if (!pre.passed) {
      score -= 0.3;
    }

    // Reduce confidence for post-condition issues
    if (!post.passed) {
      score -= 0.4;
    }

    // Incorporate checklist reward
    if (checklist.used) {
      score = (score + checklist.reward) / 2;
    }

    return Math.max(0, Math.min(1, score));
  }

  /**
   * Check if action should be retried
   * @private
   */
  _shouldRetryBasedOnVerification(verification) {
    if (!this.config.autoRetry) {
      return false;
    }

    // Don't retry if already verified
    if (verification.status === VerificationStatus.VERIFIED) {
      return false;
    }

    // Retry if failed or partial
    if (verification.status === VerificationStatus.FAILED || 
        verification.status === VerificationStatus.PARTIAL) {
      return true;
    }

    // Don't retry errors (they need different handling)
    if (verification.status === VerificationStatus.ERROR) {
      return false;
    }

    return false;
  }

  /**
   * Get reasons for retry
   * @private
   */
  _getRetryReasons(verification) {
    const reasons = [];

    if (verification.preConditions && !verification.preConditions.passed) {
      reasons.push(...verification.preConditions.issues.map(i => i.message));
    }

    if (verification.postConditions && !verification.postConditions.passed) {
      reasons.push(...verification.postConditions.issues.map(i => i.message));
    }

    if (verification.sideEffects && !verification.sideEffects.passed) {
      reasons.push(...verification.sideEffects.detectedEffects
        .filter(e => e.severity === 'error')
        .map(e => e.message));
    }

    if (verification.checklist && verification.checklist.used) {
      const failedCriteria = verification.checklist.criteria.filter(c => !c.passed && c.timestamp);
      reasons.push(...failedCriteria.map(c => `Checklist failed: ${c.description}`));
    }

    return reasons;
  }

  /**
   * Get retry count for an action
   * @private
   */
  _getRetryCount(actionId) {
    return this.verificationHistory.filter(v => v.actionId === actionId).length;
  }

  /**
   * Check if URLs match (accounting for redirects)
   * @private
   */
  _checkUrlMatch(expected, actual) {
    if (!expected || !actual) return false;
    
    try {
      const expectedUrl = new URL(expected);
      const actualUrl = new URL(actual);
      
      // Compare host and pathname
      return expectedUrl.host === actualUrl.host && 
             expectedUrl.pathname === actualUrl.pathname;
    } catch (e) {
      // Fallback to simple string comparison
      return expected === actual;
    }
  }

  /**
   * Detect if a redirect occurred
   * @private
   */
  _detectRedirect(expected, actual) {
    if (!expected || !actual) return false;
    return expected !== actual;
  }

  /**
   * Detect if result indicates an error page
   * @private
   */
  _detectErrorPage(result) {
    if (!result) return false;
    
    // Check for common error indicators
    const errorIndicators = [
      result.statusCode >= 400,
      result.title && /error|not found|forbidden|unauthorized/i.test(result.title),
      result.isErrorPage,
      result.url && /error|404|500|403/.test(result.url)
    ];
    
    return errorIndicators.some(i => i);
  }

  /**
   * Check if data is not empty
   * @private
   */
  _checkDataNotEmpty(data) {
    if (data === null || data === undefined) return false;
    if (typeof data === 'string') return data.trim().length > 0;
    if (Array.isArray(data)) return data.length > 0;
    if (typeof data === 'object') return Object.keys(data).length > 0;
    return true;
  }

  /**
   * Validate data against schema
   * @private
   */
  _validateSchema(data, schema) {
    if (!schema || !data) return true;
    
    // Simple schema validation
    for (const [key, type] of Object.entries(schema)) {
      if (!(key in data)) return false;
      
      const expectedType = type.toLowerCase();
      const actualType = typeof data[key];
      
      if (expectedType !== actualType && 
          !(expectedType === 'array' && Array.isArray(data[key]))) {
        return false;
      }
    }
    
    return true;
  }

  /**
   * Check if expected fields are present in data
   * @private
   */
  _checkFieldsPresent(data, fields) {
    if (!data || !fields) return true;
    
    for (const field of fields) {
      if (Array.isArray(data)) {
        if (!data.some(item => field in item)) return false;
      } else {
        if (!(field in data)) return false;
      }
    }
    
    return true;
  }

  /**
   * Check data type
   * @private
   */
  _checkDataType(data, expectedType) {
    if (!expectedType) return true;
    
    const actualType = typeof data;
    if (expectedType === 'array') return Array.isArray(data);
    return actualType === expectedType;
  }

  /**
   * Check if all required fields were submitted
   * @private
   */
  _checkFieldsSubmitted(formData, requiredFields) {
    if (!requiredFields || requiredFields.length === 0) return true;
    if (!formData) return false;
    
    for (const field of requiredFields) {
      if (!(field in formData) || formData[field] === undefined || formData[field] === '') {
        return false;
      }
    }
    
    return true;
  }

  /**
   * Log a message
   * @private
   */
  _log(level, message, data = {}) {
    if (this.config.logger) {
      this.config.logger(level, message, data);
    }
  }
}

module.exports = {
  SelfVerificationSystem,
  VerificationStatus
};
