/**
 * Unit tests for SemanticMatcher class
 */

const SemanticMatcher = require('../src/SemanticMatcher');
const Agent = require('../src/Agent');

describe('SemanticMatcher', () => {
  describe('constructor', () => {
    it('should create with default options', () => {
      const matcher = new SemanticMatcher();
      
      expect(matcher.method).toBe('tfidf');
      expect(matcher.hashDimensions).toBe(64);
      expect(matcher.threshold).toBe(0.3);
    });

    it('should create with custom options', () => {
      const matcher = new SemanticMatcher({
        method: 'hash',
        hashDimensions: 128,
        threshold: 0.5
      });
      
      expect(matcher.method).toBe('hash');
      expect(matcher.hashDimensions).toBe(128);
      expect(matcher.threshold).toBe(0.5);
    });
  });

  describe('computeSimilarity', () => {
    it('should compute similarity between need and offer', () => {
      const matcher = new SemanticMatcher();
      const similarity = matcher.computeSimilarity('coffee', 'coffee beans');
      
      expect(similarity).toBeGreaterThanOrEqual(0);
      expect(similarity).toBeLessThanOrEqual(1);
    });

    it('should return high similarity for matching need/offer', () => {
      const matcher = new SemanticMatcher();
      const similarity = matcher.computeSimilarity('coffee', 'coffee');
      
      expect(similarity).toBe(1);
    });

    it('should return low similarity for non-matching need/offer', () => {
      const matcher = new SemanticMatcher();
      const similarity = matcher.computeSimilarity('coffee', 'automobile');
      
      expect(similarity).toBeLessThan(0.5);
    });

    it('should handle arrays as input', () => {
      const matcher = new SemanticMatcher();
      const similarity = matcher.computeSimilarity(['coffee', 'tea'], ['coffee beans']);
      
      expect(typeof similarity).toBe('number');
    });

    it('should work with hash method', () => {
      const matcher = new SemanticMatcher({ method: 'hash' });
      const similarity = matcher.computeSimilarity('coffee', 'coffee beans');
      
      expect(similarity).toBeGreaterThanOrEqual(0);
    });

    it('should work with combined method', () => {
      const matcher = new SemanticMatcher({ method: 'combined' });
      const similarity = matcher.computeSimilarity('coffee', 'coffee beans');
      
      expect(similarity).toBeGreaterThanOrEqual(0);
    });
  });

  describe('computeSimilarityMatrix', () => {
    it('should compute average similarity between needs and offers', () => {
      const matcher = new SemanticMatcher();
      const needs = ['coffee', 'tea'];
      const offers = ['coffee beans', 'green tea'];
      
      const similarity = matcher.computeSimilarityMatrix(needs, offers);
      
      expect(similarity).toBeGreaterThanOrEqual(0);
      expect(similarity).toBeLessThanOrEqual(1);
    });

    it('should return 0 for empty arrays', () => {
      const matcher = new SemanticMatcher();
      const similarity = matcher.computeSimilarityMatrix([], []);
      
      expect(similarity).toBe(0);
    });

    it('should handle one empty array', () => {
      const matcher = new SemanticMatcher();
      const similarity = matcher.computeSimilarityMatrix(['coffee'], []);
      
      expect(similarity).toBe(0);
    });
  });

  describe('findBestMatch', () => {
    it('should find the best matching offer', () => {
      const matcher = new SemanticMatcher();
      const result = matcher.findBestMatch('coffee', ['tea', 'coffee beans', 'water']);
      
      expect(result.offer).toBeDefined();
      expect(result.score).toBeGreaterThanOrEqual(0);
      expect(typeof result.meetsThreshold).toBe('boolean');
    });

    it('should return null offer for empty offers', () => {
      const matcher = new SemanticMatcher();
      const result = matcher.findBestMatch('coffee', []);
      
      expect(result.offer).toBeNull();
      expect(result.score).toBe(-1);
    });

    it('should correctly determine threshold', () => {
      const matcher = new SemanticMatcher({ threshold: 0.5 });
      const result = matcher.findBestMatch('coffee', ['coffee']);
      
      expect(result.meetsThreshold).toBe(true);
    });
  });

  describe('findCompatibleAgents', () => {
    it('should find compatible agents', () => {
      const matcher = new SemanticMatcher({ threshold: 0.1 });
      
      const agent1 = new Agent('agent1', {
        needs: ['coffee'],
        offers: ['tea']
      });
      
      const agent2 = new Agent('agent2', {
        needs: ['tea'],
        offers: ['coffee']
      });
      
      const agent3 = new Agent('agent3', {
        needs: ['water'],
        offers: ['juice']
      });
      
      const matches = matcher.findCompatibleAgents(agent1, [agent2, agent3]);
      
      expect(Array.isArray(matches)).toBe(true);
    });

    it('should not match agent with itself', () => {
      const matcher = new SemanticMatcher();
      const agent = new Agent('agent1', {
        needs: ['coffee'],
        offers: ['coffee']
      });
      
      const matches = matcher.findCompatibleAgents(agent, [agent]);
      
      expect(matches.length).toBe(0);
    });

    it('should sort matches by score descending', () => {
      const matcher = new SemanticMatcher({ threshold: 0 });
      
      const agent1 = new Agent('agent1', {
        needs: ['coffee'],
        offers: ['tea']
      });
      
      const agent2 = new Agent('agent2', {
        needs: ['tea'],
        offers: ['coffee']
      });
      
      const agent3 = new Agent('agent3', {
        needs: ['water'],
        offers: ['juice']
      });
      
      const matches = matcher.findCompatibleAgents(agent1, [agent2, agent3]);
      
      for (let i = 1; i < matches.length; i++) {
        expect(matches[i - 1].score).toBeGreaterThanOrEqual(matches[i].score);
      }
    });

    it('should include bidirectional scores', () => {
      const matcher = new SemanticMatcher({ threshold: 0 });
      
      const agent1 = new Agent('agent1', {
        needs: ['coffee'],
        offers: ['tea']
      });
      
      const agent2 = new Agent('agent2', {
        needs: ['tea'],
        offers: ['coffee']
      });
      
      const matches = matcher.findCompatibleAgents(agent1, [agent2]);
      
      if (matches.length > 0) {
        expect(matches[0].needOfferScore).toBeDefined();
        expect(matches[0].offerNeedScore).toBeDefined();
      }
    });
  });

  describe('buildCompatibilityMatrix', () => {
    it('should build a square matrix', () => {
      const matcher = new SemanticMatcher();
      
      const agents = [
        new Agent('a1', { needs: ['x'], offers: ['y'] }),
        new Agent('a2', { needs: ['y'], offers: ['z'] }),
        new Agent('a3', { needs: ['z'], offers: ['x'] })
      ];
      
      const matrix = matcher.buildCompatibilityMatrix(agents);
      
      expect(matrix.length).toBe(3);
      matrix.forEach(row => {
        expect(row.length).toBe(3);
      });
    });

    it('should set self-match to 1.0', () => {
      const matcher = new SemanticMatcher();
      
      const agents = [
        new Agent('a1', { needs: ['x'], offers: ['y'] })
      ];
      
      const matrix = matcher.buildCompatibilityMatrix(agents);
      
      expect(matrix[0][0]).toBe(1.0);
    });

    it('should return non-negative scores', () => {
      const matcher = new SemanticMatcher();
      
      const agents = [
        new Agent('a1', { needs: ['x'], offers: ['y'] }),
        new Agent('a2', { needs: ['y'], offers: ['z'] })
      ];
      
      const matrix = matcher.buildCompatibilityMatrix(agents);
      
      matrix.forEach(row => {
        row.forEach(score => {
          expect(score).toBeGreaterThanOrEqual(0);
        });
      });
    });

    it('should handle empty agents array', () => {
      const matcher = new SemanticMatcher();
      const matrix = matcher.buildCompatibilityMatrix([]);
      
      expect(matrix).toEqual([]);
    });
  });

  describe('caching', () => {
    it('should cache embeddings', () => {
      const matcher = new SemanticMatcher();
      
      // First call
      matcher.computeSimilarity('coffee', 'tea');
      expect(matcher.embeddingsCache.size).toBeGreaterThan(0);
      
      // Second call with same texts should use cache
      matcher.computeSimilarity('coffee', 'tea');
    });

    it('should clear cache', () => {
      const matcher = new SemanticMatcher();
      matcher.computeSimilarity('coffee', 'tea');
      
      matcher.clearCache();
      
      expect(matcher.embeddingsCache.size).toBe(0);
    });
  });

  describe('getConfig', () => {
    it('should return current configuration', () => {
      const matcher = new SemanticMatcher({
        method: 'hash',
        hashDimensions: 128,
        threshold: 0.5
      });
      
      const config = matcher.getConfig();
      
      expect(config.method).toBe('hash');
      expect(config.hashDimensions).toBe(128);
      expect(config.threshold).toBe(0.5);
    });
  });

  describe('edge cases', () => {
    it('should handle special characters in text', () => {
      const matcher = new SemanticMatcher();
      const similarity = matcher.computeSimilarity('c@ff33!', 'c@ff33!');
      
      expect(similarity).toBeCloseTo(1, 10);
    });

    it('should handle very long strings', () => {
      const matcher = new SemanticMatcher();
      const longString = 'word '.repeat(100);
      const similarity = matcher.computeSimilarity(longString, longString);
      
      expect(similarity).toBe(1);
    });

    it('should handle agents with no needs or offers', () => {
      const matcher = new SemanticMatcher();
      
      const agent1 = new Agent('agent1');
      const agent2 = new Agent('agent2');
      
      const matches = matcher.findCompatibleAgents(agent1, [agent2]);
      
      expect(Array.isArray(matches)).toBe(true);
    });
  });
});
