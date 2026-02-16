/**
 * ChecklistReward Tests
 * Comprehensive unit tests for fine-grained binary criteria verification (CM2)
 */

const { ChecklistReward } = require('../src/ChecklistReward');

describe('ChecklistReward', () => {
  let reward;

  beforeEach(() => {
    reward = new ChecklistReward();
  });

  describe('Constructor', () => {
    test('should create with default config', () => {
      expect(reward.checklists).toBeInstanceOf(Map);
      expect(reward.checklists.size).toBe(0);
      expect(reward.config.maxRetries).toBe(3);
      expect(reward.config.defaultWeight).toBe(1.0);
      expect(reward.config.trackTimestamps).toBe(true);
    });

    test('should create with custom config', () => {
      const customReward = new ChecklistReward({
        maxRetries: 5,
        defaultWeight: 2.0,
        trackTimestamps: false
      });
      
      expect(customReward.config.maxRetries).toBe(5);
      expect(customReward.config.defaultWeight).toBe(2.0);
      expect(customReward.config.trackTimestamps).toBe(false);
    });

    test('should handle partial config', () => {
      const customReward = new ChecklistReward({ maxRetries: 2 });
      
      expect(customReward.config.maxRetries).toBe(2);
      expect(customReward.config.defaultWeight).toBe(1.0);
      expect(customReward.config.trackTimestamps).toBe(true);
    });

    test('should initialize built-in templates', () => {
      expect(reward.templates.navigation).toBeDefined();
      expect(reward.templates.extraction).toBeDefined();
      expect(reward.templates.form).toBeDefined();
    });
  });

  describe('createChecklist', () => {
    test('should create empty checklist for custom task type', () => {
      const checklistId = reward.createChecklist('custom');
      
      expect(checklistId).toMatch(/^checklist_\d+$/);
      expect(reward.checklists.has(checklistId)).toBe(true);
      
      const checklist = reward.checklists.get(checklistId);
      expect(checklist.taskType).toBe('custom');
      expect(checklist.criteria.size).toBe(0);
    });

    test('should create checklist with navigation template', () => {
      const checklistId = reward.createChecklist('navigation');
      const checklist = reward.checklists.get(checklistId);
      
      expect(checklist.taskType).toBe('navigation');
      expect(checklist.criteria.size).toBe(3);
      expect(checklist.criteria.has('url_reachable')).toBe(true);
      expect(checklist.criteria.has('page_loaded')).toBe(true);
      expect(checklist.criteria.has('correct_page')).toBe(true);
    });

    test('should create checklist with extraction template', () => {
      const checklistId = reward.createChecklist('extraction');
      const checklist = reward.checklists.get(checklistId);
      
      expect(checklist.taskType).toBe('extraction');
      expect(checklist.criteria.size).toBe(3);
      expect(checklist.criteria.has('element_found')).toBe(true);
      expect(checklist.criteria.has('text_extracted')).toBe(true);
      expect(checklist.criteria.has('format_correct')).toBe(true);
    });

    test('should create checklist with form template', () => {
      const checklistId = reward.createChecklist('form');
      const checklist = reward.checklists.get(checklistId);
      
      expect(checklist.taskType).toBe('form');
      expect(checklist.criteria.size).toBe(3);
      expect(checklist.criteria.has('fields_filled')).toBe(true);
      expect(checklist.criteria.has('submit_successful')).toBe(true);
      expect(checklist.criteria.has('confirmation_received')).toBe(true);
    });

    test('should generate unique IDs for multiple checklists', () => {
      const id1 = reward.createChecklist('navigation');
      const id2 = reward.createChecklist('navigation');
      const id3 = reward.createChecklist('extraction');
      
      expect(id1).not.toBe(id2);
      expect(id2).not.toBe(id3);
      expect(reward.checklists.size).toBe(3);
    });

    test('should initialize criteria with default values', () => {
      const checklistId = reward.createChecklist('navigation');
      const criterion = reward.checklists.get(checklistId).criteria.get('url_reachable');
      
      expect(criterion.passed).toBe(false);
      expect(criterion.timestamp).toBeNull();
      expect(criterion.retryCount).toBe(0);
      expect(criterion.weight).toBe(1.0);
      expect(criterion.description).toBe('URL is reachable');
    });
  });

  describe('addCriterion', () => {
    test('should add criterion to checklist', () => {
      const checklistId = reward.createChecklist('custom');
      const criterion = reward.addCriterion(checklistId, {
        id: 'test_criterion',
        description: 'Test description',
        weight: 2.0
      });
      
      expect(criterion.id).toBe('test_criterion');
      expect(criterion.description).toBe('Test description');
      expect(criterion.weight).toBe(2.0);
      expect(criterion.passed).toBe(false);
      expect(criterion.retryCount).toBe(0);
      
      expect(reward.checklists.get(checklistId).criteria.has('test_criterion')).toBe(true);
    });

    test('should auto-generate criterion ID if not provided', () => {
      const checklistId = reward.createChecklist('custom');
      const criterion = reward.addCriterion(checklistId, {
        description: 'Auto ID test'
      });
      
      expect(criterion.id).toMatch(/^criterion_\d+$/);
    });

    test('should use default weight if not specified', () => {
      const checklistId = reward.createChecklist('custom');
      const criterion = reward.addCriterion(checklistId, {
        id: 'default_weight',
        description: 'Test'
      });
      
      expect(criterion.weight).toBe(1.0);
    });

    test('should throw error for duplicate criterion ID', () => {
      const checklistId = reward.createChecklist('custom');
      reward.addCriterion(checklistId, { id: 'duplicate', description: 'First' });
      
      expect(() => {
        reward.addCriterion(checklistId, { id: 'duplicate', description: 'Second' });
      }).toThrow("Criterion with id 'duplicate' already exists in checklist");
    });

    test('should throw error for non-existent checklist', () => {
      expect(() => {
        reward.addCriterion('nonexistent', { id: 'test', description: 'Test' });
      }).toThrow("Checklist 'nonexistent' not found");
    });
  });

  describe('evaluateCriterion', () => {
    test('should mark criterion as passed', () => {
      const checklistId = reward.createChecklist('navigation');
      const result = reward.evaluateCriterion(checklistId, 'url_reachable', true);
      
      expect(result.passed).toBe(true);
      expect(result.timestamp).not.toBeNull();
      expect(result.retryCount).toBe(0);
    });

    test('should mark criterion as failed and increment retry count', () => {
      const checklistId = reward.createChecklist('navigation');
      const result = reward.evaluateCriterion(checklistId, 'url_reachable', false);
      
      expect(result.passed).toBe(false);
      expect(result.retryCount).toBe(1);
    });

    test('should accumulate retry count on multiple failures', () => {
      const checklistId = reward.createChecklist('navigation');
      
      reward.evaluateCriterion(checklistId, 'url_reachable', false);
      reward.evaluateCriterion(checklistId, 'url_reachable', false);
      const result = reward.evaluateCriterion(checklistId, 'url_reachable', false);
      
      expect(result.retryCount).toBe(3);
    });

    test('should reset retry count when criterion passes', () => {
      const checklistId = reward.createChecklist('navigation');
      
      reward.evaluateCriterion(checklistId, 'url_reachable', false);
      expect(reward.checklists.get(checklistId).criteria.get('url_reachable').retryCount).toBe(1);
      
      // retryCount should stay at 1 even after passing (we don't reset it)
      reward.evaluateCriterion(checklistId, 'url_reachable', true);
      expect(reward.checklists.get(checklistId).criteria.get('url_reachable').retryCount).toBe(1);
    });

    test('should throw error for non-existent checklist', () => {
      expect(() => {
        reward.evaluateCriterion('nonexistent', 'criterion', true);
      }).toThrow("Checklist 'nonexistent' not found");
    });

    test('should throw error for non-existent criterion', () => {
      const checklistId = reward.createChecklist('custom');
      
      expect(() => {
        reward.evaluateCriterion(checklistId, 'nonexistent', true);
      }).toThrow("Criterion 'nonexistent' not found in checklist");
    });
  });

  describe('getChecklistStatus', () => {
    test('should return complete status for empty checklist', () => {
      const checklistId = reward.createChecklist('custom');
      const status = reward.getChecklistStatus(checklistId);
      
      expect(status.id).toBe(checklistId);
      expect(status.taskType).toBe('custom');
      expect(status.totalCriteria).toBe(0);
      expect(status.passed).toBe(0);
      expect(status.failed).toBe(0);
      expect(status.pending).toBe(0);
      expect(status.isComplete).toBe(true);
      expect(status.reward).toBe(0);
      expect(status.criteria).toEqual([]);
    });

    test('should return correct counts for evaluated checklist', () => {
      const checklistId = reward.createChecklist('navigation');
      
      // Pass 1, fail 1, leave 1 pending
      reward.evaluateCriterion(checklistId, 'url_reachable', true);
      reward.evaluateCriterion(checklistId, 'page_loaded', false);
      
      const status = reward.getChecklistStatus(checklistId);
      
      expect(status.totalCriteria).toBe(3);
      expect(status.passed).toBe(1);
      expect(status.failed).toBe(1);
      expect(status.pending).toBe(1);
      expect(status.isComplete).toBe(false);
    });

    test('should throw error for non-existent checklist', () => {
      expect(() => {
        reward.getChecklistStatus('nonexistent');
      }).toThrow("Checklist 'nonexistent' not found");
    });
  });

  describe('isComplete', () => {
    test('should return true for empty checklist', () => {
      const checklistId = reward.createChecklist('custom');
      expect(reward.isComplete(checklistId)).toBe(true);
    });

    test('should return false when not all criteria passed', () => {
      const checklistId = reward.createChecklist('navigation');
      
      reward.evaluateCriterion(checklistId, 'url_reachable', true);
      reward.evaluateCriterion(checklistId, 'page_loaded', true);
      // correct_page still false
      
      expect(reward.isComplete(checklistId)).toBe(false);
    });

    test('should return true when all criteria passed', () => {
      const checklistId = reward.createChecklist('navigation');
      
      reward.evaluateCriterion(checklistId, 'url_reachable', true);
      reward.evaluateCriterion(checklistId, 'page_loaded', true);
      reward.evaluateCriterion(checklistId, 'correct_page', true);
      
      expect(reward.isComplete(checklistId)).toBe(true);
    });

    test('should return false when some criteria failed', () => {
      const checklistId = reward.createChecklist('navigation');
      
      reward.evaluateCriterion(checklistId, 'url_reachable', true);
      reward.evaluateCriterion(checklistId, 'page_loaded', false);
      reward.evaluateCriterion(checklistId, 'correct_page', true);
      
      expect(reward.isComplete(checklistId)).toBe(false);
    });

    test('should throw error for non-existent checklist', () => {
      expect(() => {
        reward.isComplete('nonexistent');
      }).toThrow("Checklist 'nonexistent' not found");
    });
  });

  describe('getReward', () => {
    test('should return 0 for empty checklist', () => {
      const checklistId = reward.createChecklist('custom');
      expect(reward.getReward(checklistId)).toBe(0);
    });

    test('should return 0 when no criteria passed', () => {
      const checklistId = reward.createChecklist('navigation');
      
      reward.evaluateCriterion(checklistId, 'url_reachable', false);
      reward.evaluateCriterion(checklistId, 'page_loaded', false);
      reward.evaluateCriterion(checklistId, 'correct_page', false);
      
      expect(reward.getReward(checklistId)).toBe(0);
    });

    test('should return 1 when all criteria passed', () => {
      const checklistId = reward.createChecklist('navigation');
      
      reward.evaluateCriterion(checklistId, 'url_reachable', true);
      reward.evaluateCriterion(checklistId, 'page_loaded', true);
      reward.evaluateCriterion(checklistId, 'correct_page', true);
      
      expect(reward.getReward(checklistId)).toBe(1);
    });

    test('should calculate weighted reward correctly', () => {
      const checklistId = reward.createChecklist('custom');
      
      // Add criteria with different weights
      reward.addCriterion(checklistId, { id: 'high', weight: 3.0 });
      reward.addCriterion(checklistId, { id: 'medium', weight: 2.0 });
      reward.addCriterion(checklistId, { id: 'low', weight: 1.0 });
      
      // Pass high and low, fail medium
      reward.evaluateCriterion(checklistId, 'high', true);
      reward.evaluateCriterion(checklistId, 'medium', false);
      reward.evaluateCriterion(checklistId, 'low', true);
      
      // Expected: (3 + 1) / (3 + 2 + 1) = 4/6 = 0.666...
      expect(reward.getReward(checklistId)).toBeCloseTo(0.667, 2);
    });

    test('should handle mixed pass/fail correctly', () => {
      const checklistId = reward.createChecklist('navigation');
      
      reward.evaluateCriterion(checklistId, 'url_reachable', true);
      reward.evaluateCriterion(checklistId, 'page_loaded', true);
      reward.evaluateCriterion(checklistId, 'correct_page', false);
      
      // 2 out of 3 passed
      expect(reward.getReward(checklistId)).toBeCloseTo(0.667, 2);
    });

    test('should throw error for non-existent checklist', () => {
      expect(() => {
        reward.getReward('nonexistent');
      }).toThrow("Checklist 'nonexistent' not found");
    });
  });

  describe('getFailedCriteria', () => {
    test('should return empty array when no failures', () => {
      const checklistId = reward.createChecklist('navigation');
      
      reward.evaluateCriterion(checklistId, 'url_reachable', true);
      reward.evaluateCriterion(checklistId, 'page_loaded', true);
      
      const failed = reward.getFailedCriteria(checklistId);
      expect(failed).toEqual([]);
    });

    test('should return failed criteria only', () => {
      const checklistId = reward.createChecklist('navigation');
      
      reward.evaluateCriterion(checklistId, 'url_reachable', true);
      reward.evaluateCriterion(checklistId, 'page_loaded', false);
      reward.evaluateCriterion(checklistId, 'correct_page', false);
      
      const failed = reward.getFailedCriteria(checklistId);
      
      expect(failed).toHaveLength(2);
      expect(failed.map(f => f.id)).toContain('page_loaded');
      expect(failed.map(f => f.id)).toContain('correct_page');
    });

    test('should include retry information', () => {
      const checklistId = reward.createChecklist('navigation');
      
      reward.evaluateCriterion(checklistId, 'page_loaded', false);
      reward.evaluateCriterion(checklistId, 'page_loaded', false);
      
      const failed = reward.getFailedCriteria(checklistId);
      
      expect(failed[0].retryCount).toBe(2);
      expect(failed[0].canRetry).toBe(true); // 2 < 3 (maxRetries)
    });

    test('should indicate when max retries reached', () => {
      const rewardWithLowRetries = new ChecklistReward({ maxRetries: 2 });
      const checklistId = rewardWithLowRetries.createChecklist('navigation');
      
      rewardWithLowRetries.evaluateCriterion(checklistId, 'page_loaded', false);
      rewardWithLowRetries.evaluateCriterion(checklistId, 'page_loaded', false);
      
      const failed = rewardWithLowRetries.getFailedCriteria(checklistId);
      
      expect(failed[0].canRetry).toBe(false); // 2 >= 2 (maxRetries)
    });

    test('should not include unevaluated criteria', () => {
      const checklistId = reward.createChecklist('navigation');
      
      // Only evaluate one criterion
      reward.evaluateCriterion(checklistId, 'url_reachable', false);
      
      const failed = reward.getFailedCriteria(checklistId);
      
      expect(failed).toHaveLength(1);
      expect(failed[0].id).toBe('url_reachable');
    });

    test('should throw error for non-existent checklist', () => {
      expect(() => {
        reward.getFailedCriteria('nonexistent');
      }).toThrow("Checklist 'nonexistent' not found");
    });
  });

  describe('resetChecklist', () => {
    test('should reset all criteria to initial state', () => {
      const checklistId = reward.createChecklist('navigation');
      
      reward.evaluateCriterion(checklistId, 'url_reachable', true);
      reward.evaluateCriterion(checklistId, 'page_loaded', false);
      
      reward.resetChecklist(checklistId);
      
      const checklist = reward.checklists.get(checklistId);
      for (const criterion of checklist.criteria.values()) {
        expect(criterion.passed).toBe(false);
        expect(criterion.timestamp).toBeNull();
        expect(criterion.retryCount).toBe(0);
      }
    });

    test('should return reset status', () => {
      const checklistId = reward.createChecklist('navigation');
      
      reward.evaluateCriterion(checklistId, 'url_reachable', true);
      
      const status = reward.resetChecklist(checklistId);
      
      expect(status.passed).toBe(0);
      expect(status.failed).toBe(0);
      expect(status.pending).toBe(3);
      expect(status.reward).toBe(0);
    });

    test('should throw error for non-existent checklist', () => {
      expect(() => {
        reward.resetChecklist('nonexistent');
      }).toThrow("Checklist 'nonexistent' not found");
    });
  });

  describe('Built-in templates', () => {
    test('getAvailableTemplates should return template names', () => {
      const templates = reward.getAvailableTemplates();
      
      expect(templates).toContain('navigation');
      expect(templates).toContain('extraction');
      expect(templates).toContain('form');
      expect(templates).toHaveLength(3);
    });

    test('getTemplate should return template details', () => {
      const navTemplate = reward.getTemplate('navigation');
      
      expect(navTemplate.name).toBe('Navigation Task Checklist');
      expect(navTemplate.description).toBe('Verify successful web page navigation');
      expect(navTemplate.criteria).toHaveLength(3);
    });

    test('getTemplate should return null for unknown template', () => {
      expect(reward.getTemplate('unknown')).toBeNull();
    });

    describe('Navigation template', () => {
      test('should have correct criteria', () => {
        const checklistId = reward.createChecklist('navigation');
        const status = reward.getChecklistStatus(checklistId);
        
        const urlReachable = status.criteria.find(c => c.id === 'url_reachable');
        expect(urlReachable.description).toBe('URL is reachable');
        
        const pageLoaded = status.criteria.find(c => c.id === 'page_loaded');
        expect(pageLoaded.description).toBe('Page successfully loaded');
        
        const correctPage = status.criteria.find(c => c.id === 'correct_page');
        expect(correctPage.description).toBe('Navigated to correct page');
      });
    });

    describe('Extraction template', () => {
      test('should have correct criteria', () => {
        const checklistId = reward.createChecklist('extraction');
        const status = reward.getChecklistStatus(checklistId);
        
        const elementFound = status.criteria.find(c => c.id === 'element_found');
        expect(elementFound.description).toBe('Target element found on page');
        
        const textExtracted = status.criteria.find(c => c.id === 'text_extracted');
        expect(textExtracted.description).toBe('Text successfully extracted');
        
        const formatCorrect = status.criteria.find(c => c.id === 'format_correct');
        expect(formatCorrect.description).toBe('Extracted data format is correct');
      });
    });

    describe('Form template', () => {
      test('should have correct criteria', () => {
        const checklistId = reward.createChecklist('form');
        const status = reward.getChecklistStatus(checklistId);
        
        const fieldsFilled = status.criteria.find(c => c.id === 'fields_filled');
        expect(fieldsFilled.description).toBe('All required fields filled');
        
        const submitSuccessful = status.criteria.find(c => c.id === 'submit_successful');
        expect(submitSuccessful.description).toBe('Form submitted successfully');
        
        const confirmationReceived = status.criteria.find(c => c.id === 'confirmation_received');
        expect(confirmationReceived.description).toBe('Confirmation message received');
      });
    });
  });

  describe('Edge cases', () => {
    test('should handle checklist with single criterion', () => {
      const checklistId = reward.createChecklist('custom');
      reward.addCriterion(checklistId, { id: 'only', description: 'Only criterion' });
      
      expect(reward.isComplete(checklistId)).toBe(false);
      
      reward.evaluateCriterion(checklistId, 'only', true);
      
      expect(reward.isComplete(checklistId)).toBe(true);
      expect(reward.getReward(checklistId)).toBe(1);
    });

    test('should handle all criteria failed', () => {
      const checklistId = reward.createChecklist('navigation');
      
      reward.evaluateCriterion(checklistId, 'url_reachable', false);
      reward.evaluateCriterion(checklistId, 'page_loaded', false);
      reward.evaluateCriterion(checklistId, 'correct_page', false);
      
      expect(reward.isComplete(checklistId)).toBe(false);
      expect(reward.getReward(checklistId)).toBe(0);
      expect(reward.getFailedCriteria(checklistId)).toHaveLength(3);
    });

    test('should handle all criteria passed', () => {
      const checklistId = reward.createChecklist('navigation');
      
      reward.evaluateCriterion(checklistId, 'url_reachable', true);
      reward.evaluateCriterion(checklistId, 'page_loaded', true);
      reward.evaluateCriterion(checklistId, 'correct_page', true);
      
      expect(reward.isComplete(checklistId)).toBe(true);
      expect(reward.getReward(checklistId)).toBe(1);
      expect(reward.getFailedCriteria(checklistId)).toHaveLength(0);
    });

    test('should handle zero-weight criteria', () => {
      const checklistId = reward.createChecklist('custom');
      
      reward.addCriterion(checklistId, { id: 'important', weight: 1.0 });
      reward.addCriterion(checklistId, { id: 'optional', weight: 0.0 });
      
      reward.evaluateCriterion(checklistId, 'important', true);
      reward.evaluateCriterion(checklistId, 'optional', false);
      
      // Only important criterion counts (weight 1 passed, total weight 1)
      expect(reward.getReward(checklistId)).toBe(1);
    });

    test('should handle timestamp tracking disabled', () => {
      const noTimestampReward = new ChecklistReward({ trackTimestamps: false });
      const checklistId = noTimestampReward.createChecklist('navigation');
      
      const checklist = noTimestampReward.checklists.get(checklistId);
      expect(checklist.createdAt).toBeNull();
      
      const result = noTimestampReward.evaluateCriterion(checklistId, 'url_reachable', true);
      expect(result.timestamp).toBeNull();
    });

    test('should handle multiple checklists independently', () => {
      const navChecklist = reward.createChecklist('navigation');
      const formChecklist = reward.createChecklist('form');
      
      reward.evaluateCriterion(navChecklist, 'url_reachable', true);
      reward.evaluateCriterion(formChecklist, 'fields_filled', true);
      
      // Navigation checklist has 1 passed, 2 pending
      expect(reward.getChecklistStatus(navChecklist).passed).toBe(1);
      
      // Form checklist has 1 passed, 2 pending
      expect(reward.getChecklistStatus(formChecklist).passed).toBe(1);
      
      // They don't interfere with each other
      expect(reward.checklists.get(navChecklist).criteria.get('url_reachable').passed).toBe(true);
      expect(reward.checklists.get(formChecklist).criteria.get('fields_filled').passed).toBe(true);
    });

    test('should handle re-evaluation of criteria', () => {
      const checklistId = reward.createChecklist('navigation');
      
      // First evaluation: fail
      reward.evaluateCriterion(checklistId, 'url_reachable', false);
      expect(reward.checklists.get(checklistId).criteria.get('url_reachable').passed).toBe(false);
      expect(reward.checklists.get(checklistId).criteria.get('url_reachable').retryCount).toBe(1);
      
      // Second evaluation: pass
      reward.evaluateCriterion(checklistId, 'url_reachable', true);
      expect(reward.checklists.get(checklistId).criteria.get('url_reachable').passed).toBe(true);
      // Retry count stays at 1
      expect(reward.checklists.get(checklistId).criteria.get('url_reachable').retryCount).toBe(1);
    });
  });

  describe('Integration scenarios', () => {
    test('should handle complete navigation task workflow', () => {
      // Create navigation checklist
      const checklistId = reward.createChecklist('navigation');
      
      // Simulate: URL not reachable initially
      reward.evaluateCriterion(checklistId, 'url_reachable', false);
      expect(reward.getReward(checklistId)).toBe(0);
      expect(reward.isComplete(checklistId)).toBe(false);
      
      // Retry: URL now reachable
      reward.evaluateCriterion(checklistId, 'url_reachable', true);
      expect(reward.getReward(checklistId)).toBeCloseTo(0.333, 2);
      
      // Page loads
      reward.evaluateCriterion(checklistId, 'page_loaded', true);
      expect(reward.getReward(checklistId)).toBeCloseTo(0.667, 2);
      
      // Wrong page initially
      reward.evaluateCriterion(checklistId, 'correct_page', false);
      expect(reward.getReward(checklistId)).toBeCloseTo(0.667, 2);
      expect(reward.isComplete(checklistId)).toBe(false);
      
      // After retry: correct page
      reward.evaluateCriterion(checklistId, 'correct_page', true);
      expect(reward.getReward(checklistId)).toBe(1);
      expect(reward.isComplete(checklistId)).toBe(true);
    });

    test('should handle form submission with retries', () => {
      const checklistId = reward.createChecklist('form');
      
      // All fields filled
      reward.evaluateCriterion(checklistId, 'fields_filled', true);
      
      // Submit fails initially
      reward.evaluateCriterion(checklistId, 'submit_successful', false);
      reward.evaluateCriterion(checklistId, 'submit_successful', false);
      
      // After retries, submit succeeds
      reward.evaluateCriterion(checklistId, 'submit_successful', true);
      
      // Confirmation received
      reward.evaluateCriterion(checklistId, 'confirmation_received', true);
      
      expect(reward.isComplete(checklistId)).toBe(true);
      expect(reward.getChecklistStatus(checklistId).criteria
        .find(c => c.id === 'submit_successful').retryCount).toBe(2);
    });

    test('should provide actionable feedback for failed criteria', () => {
      const checklistId = reward.createChecklist('extraction');
      
      // Element not found, extraction fails
      reward.evaluateCriterion(checklistId, 'element_found', false);
      reward.evaluateCriterion(checklistId, 'text_extracted', false);
      reward.evaluateCriterion(checklistId, 'format_correct', false);
      
      const failed = reward.getFailedCriteria(checklistId);
      
      expect(failed).toHaveLength(3);
      failed.forEach(criterion => {
        expect(criterion.description).toBeDefined();
        expect(criterion.canRetry).toBe(true);
      });
      
      // Root cause analysis: element not being found likely causes other failures
      const rootCause = failed.find(c => c.id === 'element_found');
      expect(rootCause.retryCount).toBe(1);
    });
  });
});
