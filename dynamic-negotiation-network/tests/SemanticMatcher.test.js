/**
 * Unit tests for SemanticMatcher class
 */

const SemanticMatcher = require('../src/SemanticMatcher');
const Agent = require('../src/Agent');

describe('SemanticMatcher', () => {
  let matcher;

  beforeEach(() => {
    matcher = new SemanticMatcher({
      method: 'tfidf',
      threshold: 0.3,
      hashDimensions: 64
    });
  });

  describe('constructor', () => {
    test('should create matcher with correct properties', () => {
      expect(matcher.method).toBe('tfidf');
      expect(matcher.threshold).toBe(0.3);
      expect(matcher.hashDimensions).toBe(64);
      expect(matcher.embeddingsCache).toBeInstanceOf(Map);
    });

    test('should use default values when not provided', () => {
      const defaultMatcher = new SemanticMatcher();
      expect(defaultMatcher.method).toBe('tfidf');
      expect(defaultMatcher.threshold).toBe(0.3);
      expect(defaultMatcher.hashDimensions).toBe(64);
    });
  });

  describe('computeSimilarity', () => {
    test('should compute similarity between matching need and offer', () => {
      const score = matcher.computeSimilarity('fresh apples', 'organic apples');
      expect(score).toBeGreaterThan(0);
      expect(score).toBeLessThanOrEqual(1);
    });

    test('should return high similarity for identical strings', () => {
      const score = matcher.computeSimilarity('apples', 'apples');
      expect(score).toBeGreaterThan(0.5);
    });

    test('should return low similarity for unrelated strings', () => {
      const score = matcher.computeSimilarity('apples', 'spaceships');
      expect(score).toBeLessThan(0.5);
    });

    test('should handle array input for need', () => {
      const score = matcher.computeSimilarity(['fresh', 'apples'], 'apples');
      expect(score).toBeGreaterThanOrEqual(0);
    });

    test('should handle array input for offer', () => {
      const score = matcher.computeSimilarity('apples', ['fresh', 'apples']);
      expect(score).toBeGreaterThanOrEqual(0);
    });

    test('should work with hash method', () => {
      const hashMatcher = new SemanticMatcher({ method: 'hash' });
      const score = hashMatcher.computeSimilarity('apples', 'apples');
      expect(score).toBeGreaterThan(0.5);
    });

    test('should work with combined method', () => {
      const combinedMatcher = new SemanticMatcher({ method: 'combined' });
      const score = combinedMatcher.computeSimilarity('apples', 'apples');
      expect(score).toBeGreaterThan(0.5);
    });

    test('should handle empty strings', () => {
      const score = matcher.computeSimilarity('', '');
      expect(score).toBeGreaterThanOrEqual(0);
    });
  });

  describe('computeSimilarityMatrix', () => {
    test('should compute average similarity for needs and offers', () => {
      const needs = ['fresh apples', 'bananas'];
      const offers = ['organic apples', 'tropical fruits'];
      const score = matcher.computeSimilarityMatrix(needs, offers);
      expect(score).toBeGreaterThanOrEqual(0);
      expect(score).toBeLessThanOrEqual(1);
    });

    test('should return 0 when no needs or offers', () => {
      const score = matcher.computeSimilarityMatrix([], ['apples']);
      expect(score).toBe(0);
      
      const score2 = matcher.computeSimilarityMatrix(['apples'], []);
      expect(score2).toBe(0);
    });

    test('should handle single need and offer', () => {
      const score = matcher.computeSimilarityMatrix(['apples'], ['apples']);
      expect(score).toBeGreaterThan(0.5);
    });
  });

  describe('findBestMatch', () => {
    test('should find best matching offer for a need', () => {
      const result = matcher.findBestMatch('apples', ['oranges', 'apples', 'bananas']);
      expect(result.offer).toBe('apples');
      expect(result.score).toBeGreaterThan(0.5);
      expect(result.meetsThreshold).toBe(true);
    });

    test('should indicate if best match meets threshold', () => {
      const result = matcher.findBestMatch('apples', ['spaceships', 'rockets']);
      expect(result.meetsThreshold).toBe(false);
    });

    test('should handle empty offers array', () => {
      const result = matcher.findBestMatch('apples', []);
      expect(result.offer).toBeNull();
      expect(result.score).toBe(-1);
    });

    test('should return null offer when no offers provided', () => {
      const result = matcher.findBestMatch('apples', []);
      expect(result.offer).toBeNull();
      expect(result.score).toBe(-1);
    });
  });

  describe('findCompatibleAgents', () => {
    test('should find compatible agents based on needs/offers', () => {
      const agent1 = new Agent('agent-1', {
        needs: ['apples'],
        offers: ['oranges']
      });
      const agent2 = new Agent('agent-2', {
        needs: ['oranges'],
        offers: ['apples']
      });
      
      const results = matcher.findCompatibleAgents(agent1, [agent2]);
      expect(results.length).toBeGreaterThan(0);
      expect(results[0].agent).toBe(agent2);
      expect(results[0].score).toBeGreaterThan(0);
    });

    test('should not include self in results', () => {
      const agent1 = new Agent('agent-1', {
        needs: ['apples'],
        offers: ['oranges']
      });
      
      const results = matcher.findCompatibleAgents(agent1, [agent1]);
      expect(results).toHaveLength(0);
    });

    test('should sort results by score descending', () => {
      const agent1 = new Agent('agent-1', {
        needs: ['apples'],
        offers: ['oranges']
      });
      const agent2 = new Agent('agent-2', {
        needs: ['oranges'],
        offers: ['apples']
      });
      const agent3 = new Agent('agent-3', {
        needs: ['bananas'],
        offers: ['grapes']
      });
      
      const results = matcher.findCompatibleAgents(agent1, [agent2, agent3]);
      if (results.length > 1) {
        expect(results[0].score).toBeGreaterThanOrEqual(results[1].score);
      }
    });

    test('should return empty array when no compatible agents', () => {
      const agent1 = new Agent('agent-1', {
        needs: ['apples'],
        offers: ['oranges']
      });
      const agent2 = new Agent('agent-2', {
        needs: ['spaceships'],
        offers: ['rockets']
      });
      
      const results = matcher.findCompatibleAgents(agent1, [agent2]);
      expect(results).toHaveLength(0);
    });

    test('should include bidirectional scores', () => {
      const agent1 = new Agent('agent-1', {
        needs: ['apples'],
        offers: ['oranges']
      });
      const agent2 = new Agent('agent-2', {
        needs: ['oranges'],
        offers: ['apples']
      });
      
      const results = matcher.findCompatibleAgents(agent1, [agent2]);
      expect(results[0].needOfferScore).toBeDefined();
      expect(results[0].offerNeedScore).toBeDefined();
    });
  });

  describe('buildCompatibilityMatrix', () => {
    test('should build matrix for all agent pairs', () => {
      const agent1 = new Agent('agent-1', { needs: ['apples'], offers: ['oranges'] });
      const agent2 = new Agent('agent-2', { needs: ['oranges'], offers: ['apples'] });
      
      const matrix = matcher.buildCompatibilityMatrix([agent1, agent2]);
      expect(matrix).toHaveLength(2);
      expect(matrix[0]).toHaveLength(2);
      expect(matrix[1]).toHaveLength(2);
    });

    test('should have self-match score of 1.0', () => {
      const agent1 = new Agent('agent-1', { needs: ['apples'], offers: ['oranges'] });
      const matrix = matcher.buildCompatibilityMatrix([agent1]);
      expect(matrix[0][0]).toBe(1.0);
    });

    test('should return empty matrix for empty agents list', () => {
      const matrix = matcher.buildCompatibilityMatrix([]);
      expect(matrix).toHaveLength(0);
    });

    test('should ensure non-negative scores', () => {
      const agent1 = new Agent('agent-1', { needs: ['apples'], offers: ['oranges'] });
      const agent2 = new Agent('agent-2', { needs: ['oranges'], offers: ['apples'] });
      
      const matrix = matcher.buildCompatibilityMatrix([agent1, agent2]);
      expect(matrix[0][1]).toBeGreaterThanOrEqual(0);
      expect(matrix[1][0]).toBeGreaterThanOrEqual(0);
    });
  });

  describe('clearCache', () => {
    test('should clear embeddings cache', () => {
      // Trigger cache by computing similarity
      matcher.computeSimilarity('apples', 'apples');
      expect(matcher.embeddingsCache.size).toBeGreaterThan(0);
      
      matcher.clearCache();
      expect(matcher.embeddingsCache.size).toBe(0);
    });
  });

  describe('getConfig', () => {
    test('should return configuration object', () => {
      const config = matcher.getConfig();
      expect(config).toEqual({
        method: 'tfidf',
        hashDimensions: 64,
        threshold: 0.3
      });
    });
  });

  describe('caching behavior', () => {
    test('should cache embeddings for repeated computations', () => {
      matcher.computeSimilarity('apples', 'oranges');
      const cacheSize = matcher.embeddingsCache.size;
      
      matcher.computeSimilarity('apples', 'oranges');
      expect(matcher.embeddingsCache.size).toBe(cacheSize);
    });

    test('should use different cache keys for different methods', () => {
      const tfidfMatcher = new SemanticMatcher({ method: 'tfidf' });
      const hashMatcher = new SemanticMatcher({ method: 'hash' });
      
      tfidfMatcher.computeSimilarity('apples', 'oranges');
      hashMatcher.computeSimilarity('apples', 'oranges');
      
      expect(tfidfMatcher.embeddingsCache.size).toBe(1);
      expect(hashMatcher.embeddingsCache.size).toBe(1);
    });
  });
});
