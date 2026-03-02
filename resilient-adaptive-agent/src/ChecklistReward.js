/**
 * ChecklistReward Module
 * Fine-grained binary criteria verification based on CM2 paper concepts
 * 
 * Key concepts from CM2:
 * - Fine-grained binary criteria for multi-step verification
 * - Each criterion is pass/fail (no partial credit)
 * - Reward is proportion of criteria passed
 * - Failed criteria can be retried
 */

class ChecklistReward {
  constructor(config = {}) {
    this.checklists = new Map();
    this.criterionCounter = 0;
    this.checklistCounter = 0;
    
    // Configuration options
    this.config = {
      maxRetries: config.maxRetries || 3,
      defaultWeight: config.defaultWeight || 1.0,
      trackTimestamps: config.trackTimestamps !== false,
      ...config
    };
    
    // Initialize built-in checklist templates
    this.templates = this._initializeTemplates();
  }

  /**
   * Initialize built-in checklist templates for common web agent tasks
   */
  _initializeTemplates() {
    return {
      navigation: {
        name: 'Navigation Task Checklist',
        description: 'Verify successful web page navigation',
        criteria: [
          { id: 'url_reachable', description: 'URL is reachable', weight: 1.0 },
          { id: 'page_loaded', description: 'Page successfully loaded', weight: 1.0 },
          { id: 'correct_page', description: 'Navigated to correct page', weight: 1.0 }
        ]
      },
      extraction: {
        name: 'Information Extraction Checklist',
        description: 'Verify successful data extraction',
        criteria: [
          { id: 'element_found', description: 'Target element found on page', weight: 1.0 },
          { id: 'text_extracted', description: 'Text successfully extracted', weight: 1.0 },
          { id: 'format_correct', description: 'Extracted data format is correct', weight: 1.0 }
        ]
      },
      form: {
        name: 'Form Submission Checklist',
        description: 'Verify successful form completion and submission',
        criteria: [
          { id: 'fields_filled', description: 'All required fields filled', weight: 1.0 },
          { id: 'submit_successful', description: 'Form submitted successfully', weight: 1.0 },
          { id: 'confirmation_received', description: 'Confirmation message received', weight: 1.0 }
        ]
      }
    };
  }

  /**
   * Create a new checklist for a task type
   * @param {string} taskType - Type of task (e.g., 'navigation', 'extraction', 'form', or custom)
   * @returns {string} checklistId - Unique identifier for the created checklist
   */
  createChecklist(taskType) {
    const checklistId = `checklist_${++this.checklistCounter}`;
    
    let criteria = [];
    
    // If it's a built-in template, copy the criteria
    if (this.templates[taskType]) {
      criteria = this.templates[taskType].criteria.map(c => ({
        ...c,
        passed: false,
        timestamp: null,
        retryCount: 0
      }));
    }
    
    this.checklists.set(checklistId, {
      id: checklistId,
      taskType,
      criteria: new Map(criteria.map(c => [c.id, c])),
      createdAt: this.config.trackTimestamps ? Date.now() : null,
      updatedAt: this.config.trackTimestamps ? Date.now() : null
    });
    
    return checklistId;
  }

  /**
   * Add a binary criterion to a checklist
   * @param {string} checklistId - Checklist identifier
   * @param {Object} criterion - Criterion definition
   * @param {string} criterion.id - Unique criterion identifier
   * @param {string} criterion.description - Description of what to verify
   * @param {number} criterion.weight - Importance weight (default: 1.0)
   * @returns {Object} The added criterion with initial state
   */
  addCriterion(checklistId, criterion) {
    const checklist = this._getChecklist(checklistId);
    
    if (!criterion.id) {
      criterion.id = `criterion_${++this.criterionCounter}`;
    }
    
    if (checklist.criteria.has(criterion.id)) {
      throw new Error(`Criterion with id '${criterion.id}' already exists in checklist`);
    }
    
    const fullCriterion = {
      id: criterion.id,
      description: criterion.description || '',
      weight: criterion.weight !== undefined ? criterion.weight : this.config.defaultWeight,
      passed: false,
      timestamp: null,
      retryCount: 0
    };
    
    checklist.criteria.set(criterion.id, fullCriterion);
    checklist.updatedAt = this.config.trackTimestamps ? Date.now() : null;
    
    return { ...fullCriterion };
  }

  /**
   * Evaluate a single criterion (pass/fail)
   * @param {string} checklistId - Checklist identifier
   * @param {string} criterionId - Criterion identifier
   * @param {boolean} result - Pass (true) or fail (false)
   * @returns {Object} Updated criterion state
   */
  evaluateCriterion(checklistId, criterionId, result) {
    const checklist = this._getChecklist(checklistId);
    const criterion = this._getCriterion(checklist, criterionId);
    
    // If failing, increment retry count
    if (!result) {
      criterion.retryCount++;
    }
    
    criterion.passed = result;
    criterion.timestamp = this.config.trackTimestamps ? Date.now() : null;
    checklist.updatedAt = this.config.trackTimestamps ? Date.now() : null;
    
    return { ...criterion };
  }

  /**
   * Get current status of a checklist
   * @param {string} checklistId - Checklist identifier
   * @returns {Object} Complete checklist status
   */
  getChecklistStatus(checklistId) {
    const checklist = this._getChecklist(checklistId);
    const criteria = Array.from(checklist.criteria.values());
    
    const passedCount = criteria.filter(c => c.passed).length;
    const totalCount = criteria.length;
    const failedCount = criteria.filter(c => !c.passed && c.timestamp !== null).length;
    const pendingCount = totalCount - passedCount - failedCount;
    
    return {
      id: checklist.id,
      taskType: checklist.taskType,
      totalCriteria: totalCount,
      passed: passedCount,
      failed: failedCount,
      pending: pendingCount,
      isComplete: this.isComplete(checklistId),
      reward: this.getReward(checklistId),
      criteria: criteria.map(c => ({ ...c }))
    };
  }

  /**
   * Check if all criteria in a checklist have passed
   * @param {string} checklistId - Checklist identifier
   * @returns {boolean} True if all criteria passed
   */
  isComplete(checklistId) {
    const checklist = this._getChecklist(checklistId);
    
    if (checklist.criteria.size === 0) {
      return true; // Empty checklist is considered complete
    }
    
    for (const criterion of checklist.criteria.values()) {
      if (!criterion.passed) {
        return false;
      }
    }
    
    return true;
  }

  /**
   * Calculate reward score (0-1 based on pass rate)
   * @param {string} checklistId - Checklist identifier
   * @returns {number} Reward score between 0 and 1
   */
  getReward(checklistId) {
    const checklist = this._getChecklist(checklistId);
    
    if (checklist.criteria.size === 0) {
      return 0; // No criteria means no reward
    }
    
    let totalWeight = 0;
    let passedWeight = 0;
    
    for (const criterion of checklist.criteria.values()) {
      totalWeight += criterion.weight;
      if (criterion.passed) {
        passedWeight += criterion.weight;
      }
    }
    
    return totalWeight > 0 ? passedWeight / totalWeight : 0;
  }

  /**
   * Get list of failed criteria for retry
   * @param {string} checklistId - Checklist identifier
   * @returns {Array} List of failed criteria with retry information
   */
  getFailedCriteria(checklistId) {
    const checklist = this._getChecklist(checklistId);
    const failed = [];
    
    for (const criterion of checklist.criteria.values()) {
      if (!criterion.passed && criterion.timestamp !== null) {
        failed.push({
          ...criterion,
          canRetry: criterion.retryCount < this.config.maxRetries
        });
      }
    }
    
    return failed;
  }

  /**
   * Reset checklist for re-evaluation
   * @param {string} checklistId - Checklist identifier
   * @returns {Object} Reset checklist status
   */
  resetChecklist(checklistId) {
    const checklist = this._getChecklist(checklistId);
    
    for (const criterion of checklist.criteria.values()) {
      criterion.passed = false;
      criterion.timestamp = null;
      criterion.retryCount = 0;
    }
    
    checklist.updatedAt = this.config.trackTimestamps ? Date.now() : null;
    
    return this.getChecklistStatus(checklistId);
  }

  /**
   * Get available built-in template names
   * @returns {Array} List of available template names
   */
  getAvailableTemplates() {
    return Object.keys(this.templates);
  }

  /**
   * Get template details
   * @param {string} templateName - Template name
   * @returns {Object} Template details
   */
  getTemplate(templateName) {
    return this.templates[templateName] ? { ...this.templates[templateName] } : null;
  }

  /**
   * Helper: Get checklist by ID or throw error
   * @private
   */
  _getChecklist(checklistId) {
    const checklist = this.checklists.get(checklistId);
    if (!checklist) {
      throw new Error(`Checklist '${checklistId}' not found`);
    }
    return checklist;
  }

  /**
   * Helper: Get criterion by ID or throw error
   * @private
   */
  _getCriterion(checklist, criterionId) {
    const criterion = checklist.criteria.get(criterionId);
    if (!criterion) {
      throw new Error(`Criterion '${criterionId}' not found in checklist`);
    }
    return criterion;
  }
}

module.exports = { ChecklistReward };
