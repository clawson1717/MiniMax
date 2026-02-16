/**
 * Embedding Module
 * 
 * Provides vector embeddings for semantic matching.
 * Implements both TF-IDF and hash-based approaches.
 */

class Embedding {
  /**
   * Create embeddings using TF-IDF approach
   * @param {string[]} texts - Array of text strings
   * @returns {number[][]} - Array of embedding vectors
   */
  static tfidf(texts) {
    const tokenizer = (text) => text.toLowerCase().split(/\W+/).filter(w => w.length > 0);
    
    // Build vocabulary
    const vocabulary = new Set();
    const tokenizedTexts = texts.map(tokenizer);
    tokenizedTexts.forEach(tokens => tokens.forEach(token => vocabulary.add(token)));
    
    const vocabArray = Array.from(vocabulary);
    const vocabSize = vocabArray.length;
    
    // Create TF-IDF vectors
    const vectors = tokenizedTexts.map(tokens => {
      const tf = {};
      tokens.forEach(token => {
        tf[token] = (tf[token] || 0) + 1;
      });
      
      // Normalize by text length
      const vector = new Array(vocabSize).fill(0);
      let maxTf = 0;
      Object.values(tf).forEach(count => {
        if (count > maxTf) maxTf = count;
      });
      
      vocabArray.forEach((word, i) => {
        if (tf[word]) {
          // TF normalization (sublinear)
          vector[i] = 1 + Math.log(tf[word]);
        }
      });
      
      return vector;
    });

    // Compute IDF
    const idf = new Array(vocabSize).fill(0);
    const N = texts.length;
    vocabArray.forEach((word, i) => {
      let df = 0;
      tokenizedTexts.forEach(tokens => {
        if (tokens.includes(word)) df++;
      });
      idf[i] = Math.log((N + 1) / (df + 1)) + 1;
    });

    // Apply IDF to vectors
    return vectors.map(vector => 
      vector.map((val, i) => val * idf[i])
    );
  }

  /**
   * Create embeddings using hash-based approach
   * @param {string[]} texts - Array of text strings
   * @param {number} dimensions - Number of dimensions for the hash (default 128)
   * @returns {number[][]} - Array of embedding vectors
   */
  static hash(texts, dimensions = 128) {
    const hashFunction = (str, seed = 0) => {
      let hash = seed;
      for (let i = 0; i < str.length; i++) {
        hash = ((hash << 5) - hash) + str.charCodeAt(i);
        hash = hash & hash; // Convert to 32bit integer
      }
      return Math.abs(hash);
    };

    return texts.map(text => {
      const tokens = text.toLowerCase().split(/\W+/).filter(w => w.length > 0);
      const vector = new Array(dimensions).fill(0);
      
      tokens.forEach(token => {
        const hash1 = hashFunction(token, 1);
        const hash2 = hashFunction(token, 2);
        
        for (let i = 0; i < dimensions; i++) {
          // Use multiple hash functions for better distribution
          const idx1 = hash1 % dimensions;
          const idx2 = hash2 % dimensions;
          
          // Signed random projection
          vector[idx1] += (hash1 % 2 === 0 ? 1 : -1) * 0.5;
          vector[idx2] += (hash2 % 2 === 0 ? 1 : -1) * 0.5;
        }
      });

      // Normalize vector
      const magnitude = Math.sqrt(vector.reduce((sum, v) => sum + v * v, 0));
      if (magnitude > 0) {
        return vector.map(v => v / magnitude);
      }
      return vector;
    });
  }

  /**
   * Create combined embeddings (TF-IDF + hash)
   * @param {string[]} texts
   * @param {number} hashDimensions
   * @returns {Object} - Combined embeddings with both methods
   */
  static combined(texts, hashDimensions = 64) {
    return {
      tfidf: Embedding.tfidf(texts),
      hash: Embedding.hash(texts, hashDimensions)
    };
  }

  /**
   * Compute cosine similarity between two vectors
   * @param {number[]} vec1
   * @param {number[]} vec2
   * @returns {number} - Similarity score between -1 and 1
   */
  static cosineSimilarity(vec1, vec2) {
    if (vec1.length !== vec2.length) {
      throw new Error('Vectors must have the same dimension');
    }

    let dotProduct = 0;
    let norm1 = 0;
    let norm2 = 0;

    for (let i = 0; i < vec1.length; i++) {
      dotProduct += vec1[i] * vec2[i];
      norm1 += vec1[i] * vec1[i];
      norm2 += vec2[i] * vec2[i];
    }

    const denominator = Math.sqrt(norm1) * Math.sqrt(norm2);
    if (denominator === 0) return 0;

    return dotProduct / denominator;
  }

  /**
   * Compute Euclidean distance between two vectors
   * @param {number[]} vec1
   * @param {number[]} vec2
   * @returns {number}
   */
  static euclideanDistance(vec1, vec2) {
    if (vec1.length !== vec2.length) {
      throw new Error('Vectors must have the same dimension');
    }

    let sum = 0;
    for (let i = 0; i < vec1.length; i++) {
      sum += Math.pow(vec1[i] - vec2[i], 2);
    }
    return Math.sqrt(sum);
  }
}

module.exports = Embedding;
