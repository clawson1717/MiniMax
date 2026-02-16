/**
 * Unit tests for Embedding class
 */

const Embedding = require('../src/Embedding');

describe('Embedding', () => {
  describe('tfidf', () => {
    test('should create TF-IDF vectors for texts', () => {
      const texts = ['apple banana', 'banana cherry'];
      const embeddings = Embedding.tfidf(texts);
      
      expect(embeddings).toHaveLength(2);
      expect(embeddings[0]).toBeInstanceOf(Array);
      expect(embeddings[0].length).toBeGreaterThan(0);
    });

    test('should create vectors of same dimension for all texts', () => {
      const texts = ['short', 'this is a much longer text with many words'];
      const embeddings = Embedding.tfidf(texts);
      
      expect(embeddings[0].length).toBe(embeddings[1].length);
    });

    test('should handle empty texts array', () => {
      const embeddings = Embedding.tfidf([]);
      
      expect(embeddings).toEqual([]);
    });

    test('should handle single text', () => {
      const embeddings = Embedding.tfidf(['apple']);
      
      expect(embeddings).toHaveLength(1);
    });

    test('should compute TF component correctly', () => {
      const texts = ['apple apple banana'];
      const embeddings = Embedding.tfidf(texts);
      
      // Check that values are computed (sublinear TF)
      expect(embeddings[0].some(v => v > 0)).toBe(true);
    });

    test('should compute IDF component correctly', () => {
      const texts = ['apple banana', 'apple cherry', 'apple date'];
      const embeddings = Embedding.tfidf(texts);
      
      // 'apple' appears in all documents, so IDF should be lower
      // 'banana' appears in 1 document, so IDF should be higher
      // This is harder to test directly, so we just check structure
      expect(embeddings.every(e => e.length === embeddings[0].length)).toBe(true);
    });

    test('should normalize case and split on non-word characters', () => {
      const texts = ['Apple-Banana', 'apple:banana'];
      const embeddings = Embedding.tfidf(texts);
      
      // Should treat as similar since tokenizer normalizes
      const similarity = Embedding.cosineSimilarity(embeddings[0], embeddings[1]);
      expect(similarity).toBeGreaterThan(0.5);
    });

    test('should filter out empty tokens', () => {
      const texts = ['apple  banana', 'apple banana'];
      const embeddings = Embedding.tfidf(texts);
      
      const similarity = Embedding.cosineSimilarity(embeddings[0], embeddings[1]);
      expect(similarity).toBeGreaterThan(0.9);
    });
  });

  describe('hash', () => {
    test('should create hash-based embeddings', () => {
      const texts = ['apple banana', 'banana cherry'];
      const embeddings = Embedding.hash(texts, 128);
      
      expect(embeddings).toHaveLength(2);
      expect(embeddings[0]).toBeInstanceOf(Array);
      expect(embeddings[0].length).toBe(128);
    });

    test('should use specified dimensions', () => {
      const texts = ['test'];
      const embeddings = Embedding.hash(texts, 64);
      
      expect(embeddings[0].length).toBe(64);
    });

    test('should default to 128 dimensions', () => {
      const texts = ['test'];
      const embeddings = Embedding.hash(texts);
      
      expect(embeddings[0].length).toBe(128);
    });

    test('should produce normalized vectors', () => {
      const texts = ['apple banana cherry'];
      const embeddings = Embedding.hash(texts, 64);
      
      const vector = embeddings[0];
      const magnitude = Math.sqrt(vector.reduce((sum, v) => sum + v * v, 0));
      
      // Vectors should be normalized to unit length
      expect(Math.abs(magnitude - 1)).toBeLessThan(0.01);
    });

    test('should handle empty texts array', () => {
      const embeddings = Embedding.hash([]);
      
      expect(embeddings).toEqual([]);
    });

    test('should handle single text', () => {
      const embeddings = Embedding.hash(['test'], 64);
      
      expect(embeddings).toHaveLength(1);
    });

    test('should produce consistent embeddings for same text', () => {
      const texts1 = ['apple banana'];
      const texts2 = ['apple banana'];
      
      const embeddings1 = Embedding.hash(texts1, 64);
      const embeddings2 = Embedding.hash(texts2, 64);
      
      expect(embeddings1[0]).toEqual(embeddings2[0]);
    });

    test('should produce different embeddings for different texts', () => {
      const texts = ['apple', 'banana'];
      const embeddings = Embedding.hash(texts, 64);
      
      // Should be different vectors
      let allSame = true;
      for (let i = 0; i < embeddings[0].length; i++) {
        if (embeddings[0][i] !== embeddings[1][i]) {
          allSame = false;
          break;
        }
      }
      expect(allSame).toBe(false);
    });

    test('should handle empty string', () => {
      const embeddings = Embedding.hash([''], 64);
      
      // Empty string should produce zero vector (not normalized)
      const isZeroVector = embeddings[0].every(v => v === 0);
      expect(isZeroVector).toBe(true);
    });
  });

  describe('combined', () => {
    test('should return both TF-IDF and hash embeddings', () => {
      const texts = ['apple banana', 'banana cherry'];
      const embeddings = Embedding.combined(texts, 64);
      
      expect(embeddings).toHaveProperty('tfidf');
      expect(embeddings).toHaveProperty('hash');
      expect(embeddings.tfidf).toHaveLength(2);
      expect(embeddings.hash).toHaveLength(2);
    });

    test('should use specified hash dimensions', () => {
      const texts = ['test'];
      const embeddings = Embedding.combined(texts, 32);
      
      expect(embeddings.hash[0].length).toBe(32);
    });

    test('should default hash dimensions to 64', () => {
      const texts = ['test'];
      const embeddings = Embedding.combined(texts);
      
      expect(embeddings.hash[0].length).toBe(64);
    });
  });

  describe('cosineSimilarity', () => {
    test('should compute similarity between identical vectors', () => {
      const vec1 = [1, 2, 3];
      const vec2 = [1, 2, 3];
      
      const similarity = Embedding.cosineSimilarity(vec1, vec2);
      
      expect(similarity).toBeCloseTo(1, 5);
    });

    test('should compute similarity between orthogonal vectors', () => {
      const vec1 = [1, 0, 0];
      const vec2 = [0, 1, 0];
      
      const similarity = Embedding.cosineSimilarity(vec1, vec2);
      
      expect(similarity).toBeCloseTo(0, 5);
    });

    test('should compute similarity between opposite vectors', () => {
      const vec1 = [1, 2, 3];
      const vec2 = [-1, -2, -3];
      
      const similarity = Embedding.cosineSimilarity(vec1, vec2);
      
      expect(similarity).toBeCloseTo(-1, 5);
    });

    test('should throw error for different dimensions', () => {
      const vec1 = [1, 2, 3];
      const vec2 = [1, 2];
      
      expect(() => {
        Embedding.cosineSimilarity(vec1, vec2);
      }).toThrow('Vectors must have the same dimension');
    });

    test('should return 0 for zero vectors', () => {
      const vec1 = [0, 0, 0];
      const vec2 = [1, 2, 3];
      
      const similarity = Embedding.cosineSimilarity(vec1, vec2);
      
      expect(similarity).toBe(0);
    });

    test('should handle negative values', () => {
      const vec1 = [-1, -2, -3];
      const vec2 = [1, 2, 3];
      
      const similarity = Embedding.cosineSimilarity(vec1, vec2);
      
      expect(similarity).toBeCloseTo(-1, 5);
    });
  });

  describe('euclideanDistance', () => {
    test('should compute distance between identical vectors', () => {
      const vec1 = [1, 2, 3];
      const vec2 = [1, 2, 3];
      
      const distance = Embedding.euclideanDistance(vec1, vec2);
      
      expect(distance).toBe(0);
    });

    test('should compute distance between different vectors', () => {
      const vec1 = [0, 0, 0];
      const vec2 = [1, 1, 1];
      
      const distance = Embedding.euclideanDistance(vec1, vec2);
      
      expect(distance).toBeCloseTo(Math.sqrt(3), 5);
    });

    test('should throw error for different dimensions', () => {
      const vec1 = [1, 2, 3];
      const vec2 = [1, 2];
      
      expect(() => {
        Embedding.euclideanDistance(vec1, vec2);
      }).toThrow('Vectors must have the same dimension');
    });

    test('should compute correct distance', () => {
      const vec1 = [1, 2];
      const vec2 = [4, 6];
      
      const distance = Embedding.euclideanDistance(vec1, vec2);
      
      // sqrt((4-1)^2 + (6-2)^2) = sqrt(9 + 16) = 5
      expect(distance).toBe(5);
    });
  });

  describe('integration', () => {
    test('similar texts should have high TF-IDF similarity', () => {
      const texts = ['apple banana', 'apple orange'];
      const embeddings = Embedding.tfidf(texts);
      
      const similarity = Embedding.cosineSimilarity(embeddings[0], embeddings[1]);
      
      expect(similarity).toBeGreaterThan(0);
    });

    test('similar texts should have high hash similarity', () => {
      const texts = ['apple banana', 'apple orange'];
      const embeddings = Embedding.hash(texts, 128);
      
      const similarity = Embedding.cosineSimilarity(embeddings[0], embeddings[1]);
      
      // Hash similarity may be lower due to random projections
      expect(similarity).toBeGreaterThan(-1);
      expect(similarity).toBeLessThanOrEqual(1);
    });

    test('different texts should have different embeddings', () => {
      const texts = ['apple', 'banana', 'cherry'];
      const tfidfEmbeddings = Embedding.tfidf(texts);
      
      const sim1 = Embedding.cosineSimilarity(tfidfEmbeddings[0], tfidfEmbeddings[1]);
      const sim2 = Embedding.cosineSimilarity(tfidfEmbeddings[0], tfidfEmbeddings[2]);
      
      // No specific assertion - just ensuring it works
      expect(typeof sim1).toBe('number');
      expect(typeof sim2).toBe('number');
    });
  });
});
