/**
 * Unit tests for Embedding class
 */

const Embedding = require('../src/Embedding');

describe('Embedding', () => {
  describe('tfidf', () => {
    it('should create embeddings for text array', () => {
      const texts = ['hello world', 'hello universe'];
      const embeddings = Embedding.tfidf(texts);
      
      expect(embeddings.length).toBe(2);
      expect(embeddings[0]).toBeInstanceOf(Array);
      expect(embeddings[1]).toBeInstanceOf(Array);
    });

    it('should create vectors of correct dimension', () => {
      const texts = ['hello world', 'hello universe'];
      const embeddings = Embedding.tfidf(texts);
      
      // All vectors should have the same dimension (vocabulary size)
      expect(embeddings[0].length).toBe(embeddings[1].length);
    });

    it('should handle empty texts array', () => {
      const embeddings = Embedding.tfidf([]);
      expect(embeddings).toEqual([]);
    });

    it('should handle single text', () => {
      const embeddings = Embedding.tfidf(['hello world']);
      expect(embeddings.length).toBe(1);
    });

    it('should handle text with repeated words', () => {
      const texts = ['hello hello hello world'];
      const embeddings = Embedding.tfidf(texts);
      
      expect(embeddings.length).toBe(1);
    });

    it('should produce different embeddings for different texts', () => {
      const texts = ['cat dog', 'car truck'];
      const embeddings = Embedding.tfidf(texts);
      
      // Different texts should produce different vectors
      const similarity = Embedding.cosineSimilarity(embeddings[0], embeddings[1]);
      expect(similarity).toBeLessThan(1);
    });

    it('should produce similar embeddings for similar texts', () => {
      const texts = ['hello world', 'hello world test'];
      const embeddings = Embedding.tfidf(texts);
      
      const similarity = Embedding.cosineSimilarity(embeddings[0], embeddings[1]);
      expect(similarity).toBeGreaterThan(0.5);
    });
  });

  describe('hash', () => {
    it('should create embeddings for text array', () => {
      const texts = ['hello world', 'hello universe'];
      const embeddings = Embedding.hash(texts);
      
      expect(embeddings.length).toBe(2);
      expect(embeddings[0]).toBeInstanceOf(Array);
    });

    it('should use default dimensions of 128', () => {
      const texts = ['hello world'];
      const embeddings = Embedding.hash(texts);
      
      expect(embeddings[0].length).toBe(128);
    });

    it('should use custom dimensions', () => {
      const texts = ['hello world'];
      const embeddings = Embedding.hash(texts, 64);
      
      expect(embeddings[0].length).toBe(64);
    });

    it('should handle empty texts array', () => {
      const embeddings = Embedding.hash([]);
      expect(embeddings).toEqual([]);
    });

    it('should produce normalized vectors', () => {
      const texts = ['hello world test'];
      const embeddings = Embedding.hash(texts);
      
      const magnitude = Math.sqrt(
        embeddings[0].reduce((sum, v) => sum + v * v, 0)
      );
      expect(magnitude).toBeCloseTo(1, 5);
    });

    it('should produce same embeddings for same text', () => {
      const texts = ['hello world'];
      const embeddings1 = Embedding.hash(texts);
      const embeddings2 = Embedding.hash(texts);
      
      expect(embeddings1[0]).toEqual(embeddings2[0]);
    });

    it('should produce similar embeddings for similar texts', () => {
      const texts = ['hello world', 'hello world test'];
      const embeddings = Embedding.hash(texts);
      
      const similarity = Embedding.cosineSimilarity(embeddings[0], embeddings[1]);
      expect(similarity).toBeGreaterThan(0.5);
    });
  });

  describe('combined', () => {
    it('should return both tfidf and hash embeddings', () => {
      const texts = ['hello world'];
      const embeddings = Embedding.combined(texts);
      
      expect(embeddings.tfidf).toBeDefined();
      expect(embeddings.hash).toBeDefined();
      expect(embeddings.tfidf.length).toBe(1);
      expect(embeddings.hash.length).toBe(1);
    });

    it('should use default hash dimensions', () => {
      const texts = ['hello world'];
      const embeddings = Embedding.combined(texts);
      
      expect(embeddings.hash[0].length).toBe(64);
    });

    it('should use custom hash dimensions', () => {
      const texts = ['hello world'];
      const embeddings = Embedding.combined(texts, 32);
      
      expect(embeddings.hash[0].length).toBe(32);
    });
  });

  describe('cosineSimilarity', () => {
    it('should return 1 for identical vectors', () => {
      const vec1 = [1, 0, 0];
      const vec2 = [1, 0, 0];
      
      expect(Embedding.cosineSimilarity(vec1, vec2)).toBe(1);
    });

    it('should return 0 for orthogonal vectors', () => {
      const vec1 = [1, 0, 0];
      const vec2 = [0, 1, 0];
      
      expect(Embedding.cosineSimilarity(vec1, vec2)).toBe(0);
    });

    it('should return -1 for opposite vectors', () => {
      const vec1 = [1, 0, 0];
      const vec2 = [-1, 0, 0];
      
      expect(Embedding.cosineSimilarity(vec1, vec2)).toBe(-1);
    });

    it('should return 0 for zero vectors', () => {
      const vec1 = [0, 0, 0];
      const vec2 = [1, 2, 3];
      
      expect(Embedding.cosineSimilarity(vec1, vec2)).toBe(0);
    });

    it('should throw error for different dimension vectors', () => {
      const vec1 = [1, 2];
      const vec2 = [1, 2, 3];
      
      expect(() => Embedding.cosineSimilarity(vec1, vec2)).toThrow('same dimension');
    });

    it('should calculate partial similarity correctly', () => {
      const vec1 = [1, 1, 0];
      const vec2 = [1, 0, 0];
      
      const similarity = Embedding.cosineSimilarity(vec1, vec2);
      expect(similarity).toBeCloseTo(Math.sqrt(2) / 2, 5);
    });
  });

  describe('euclideanDistance', () => {
    it('should return 0 for identical vectors', () => {
      const vec1 = [1, 2, 3];
      const vec2 = [1, 2, 3];
      
      expect(Embedding.euclideanDistance(vec1, vec2)).toBe(0);
    });

    it('should calculate distance correctly', () => {
      const vec1 = [0, 0, 0];
      const vec2 = [3, 4, 0];
      
      expect(Embedding.euclideanDistance(vec1, vec2)).toBe(5);
    });

    it('should throw error for different dimension vectors', () => {
      const vec1 = [1, 2];
      const vec2 = [1, 2, 3];
      
      expect(() => Embedding.euclideanDistance(vec1, vec2)).toThrow('same dimension');
    });

    it('should handle negative values', () => {
      const vec1 = [-1, -1];
      const vec2 = [2, 3];
      
      const distance = Embedding.euclideanDistance(vec1, vec2);
      expect(distance).toBe(5);
    });
  });

  describe('edge cases', () => {
    it('should handle text with only punctuation', () => {
      const texts = ['... ??? !!!'];
      const embeddings = Embedding.tfidf(texts);
      
      // Should handle gracefully
      expect(embeddings.length).toBe(1);
    });

    it('should handle text with numbers', () => {
      const texts = ['price is 100 dollars'];
      const embeddings = Embedding.tfidf(texts);
      
      expect(embeddings.length).toBe(1);
    });

    it('should handle very long text', () => {
      const longText = 'word '.repeat(1000);
      const texts = [longText];
      const embeddings = Embedding.tfidf(texts);
      
      expect(embeddings.length).toBe(1);
    });

    it('should handle unicode characters', () => {
      const texts = ['hello 世界', 'привет мир'];
      const embeddings = Embedding.tfidf(texts);
      
      expect(embeddings.length).toBe(2);
    });
  });
});
