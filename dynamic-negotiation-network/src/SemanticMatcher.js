/**
 * SemanticMatcher Class
 * 
 * Matches agent needs to offers using semantic similarity.
 * Uses TF-IDF and hash-based embeddings for computing compatibility.
 */

const Embedding = require('./Embedding');

class SemanticMatcher {
  /**
   * Create a new SemanticMatcher
   * @param {Object} options - Configuration options
   * @param {string} options.method - Embedding method: 'tfidf', 'hash', or 'combined' (default: 'tfidf')
   * @param {number} options.hashDimensions - Dimensions for hash embeddings (default: 64)
   * @param {number} options.threshold - Minimum similarity threshold (default: 0.3)
   */
  constructor(options = {}) {
    this.method = options.method || 'tfidf';
    this.hashDimensions = options.hashDimensions || 64;
    this.threshold = options.threshold || 0.3;
    this.embeddingsCache = new Map();
  }

  /**
   * Compute similarity between a need and an offer
   * @param {string} need
   * @param {string} offer
   * @returns {number} - Similarity score between 0 and 1
   */
  computeSimilarity(need, offer) {
    const needText = Array.isArray(need) ? need.join(' ') : need;
    const offerText = Array.isArray(offer) ? offer.join(' ') : offer;

    const embeddings = this._getEmbeddings([needText, offerText]);
    
    if (this.method === 'hash') {
      return this._computeHashSimilarity(embeddings);
    } else if (this.method === 'combined') {
      const tfidfSim = this._computeTfidfSimilarity(embeddings.tfidf);
      const hashSim = this._computeHashSimilarity(embeddings.hash);
      return (tfidfSim + hashSim) / 2;
    }
    return this._computeTfidfSimilarity(embeddings);
  }

  /**
   * Compute similarity between array of needs and array of offers
   * @param {string[]} needs
   * @param {string[]} offers
   * @returns {number} - Average similarity score
   */
  computeSimilarityMatrix(needs, offers) {
    const allTexts = [...needs, ...offers];
    const embeddings = this._getEmbeddings(allTexts);
    
    const needEmbeddings = embeddings.slice(0, needs.length);
    const offerEmbeddings = embeddings.slice(needs.length);
    
    let totalSimilarity = 0;
    let matchCount = 0;
    
    needEmbeddings.forEach(needVec => {
      offerEmbeddings.forEach(offerVec => {
        let sim;
        if (this.method === 'hash' || this.method === 'combined') {
          sim = Embedding.cosineSimilarity(needVec, offerVec);
        } else {
          sim = Embedding.cosineSimilarity(needVec, offerVec);
        }
        totalSimilarity += Math.max(0, sim); // Ensure non-negative
        matchCount++;
      });
    });
    
    return matchCount > 0 ? totalSimilarity / matchCount : 0;
  }

  /**
   * Find the best matching offer for a given need
   * @param {string} need
   * @param {string[]} offers
   * @returns {Object} - Best match with offer and score
   */
  findBestMatch(need, offers) {
    let bestScore = -1;
    let bestOffer = null;
    
    offers.forEach(offer => {
      const score = this.computeSimilarity(need, offer);
      if (score > bestScore) {
        bestScore = score;
        bestOffer = offer;
      }
    });
    
    return {
      offer: bestOffer,
      score: bestScore,
      meetsThreshold: bestScore >= this.threshold
    };
  }

  /**
   * Find all compatible agents for a given agent based on their needs/offers
   * @param {Agent} agent - Source agent
   * @param {Agent[]} candidates - Potential target agents
   * @returns {Object[]} - Array of matches with scores
   */
  findCompatibleAgents(agent, candidates) {
    const results = [];
    
    candidates.forEach(candidate => {
      if (candidate.id === agent.id) return; // Don't match with self
      
      // Compute bidirectional compatibility
      const needOfferSim = this.computeSimilarityMatrix(agent.getNeeds(), candidate.getOffers());
      const offerNeedSim = this.computeSimilarityMatrix(candidate.getNeeds(), agent.getOffers());
      
      // Combined score (both must have compatible needs/offers)
      const combinedScore = (needOfferSim + offerNeedSim) / 2;
      
      if (combinedScore >= this.threshold) {
        results.push({
          agent: candidate,
          score: combinedScore,
          needOfferScore: needOfferSim,
          offerNeedScore: offerNeedSim,
          meetsThreshold: true
        });
      }
    });
    
    // Sort by score descending
    results.sort((a, b) => b.score - a.score);
    
    return results;
  }

  /**
   * Build a compatibility matrix for a list of agents
   * @param {Agent[]} agents
   * @returns {number[][]} - Matrix of compatibility scores
   */
  buildCompatibilityMatrix(agents) {
    const matrix = [];
    
    for (let i = 0; i < agents.length; i++) {
      matrix[i] = [];
      for (let j = 0; j < agents.length; j++) {
        if (i === j) {
          matrix[i][j] = 1.0; // Self-match
        } else {
          const compatibility = this.computeSimilarityMatrix(
            agents[i].getNeeds(),
            agents[j].getOffers()
          );
          matrix[i][j] = Math.max(0, compatibility);
        }
      }
    }
    
    return matrix;
  }

  /**
   * Get embeddings using the configured method
   * @private
   */
  _getEmbeddings(texts) {
    const cacheKey = `${this.method}:${texts.join('|')}`;
    
    if (this.embeddingsCache.has(cacheKey)) {
      return this.embeddingsCache.get(cacheKey);
    }
    
    let embeddings;
    if (this.method === 'hash') {
      embeddings = Embedding.hash(texts, this.hashDimensions);
    } else if (this.method === 'combined') {
      embeddings = Embedding.combined(texts, this.hashDimensions);
    } else {
      embeddings = Embedding.tfidf(texts);
    }
    
    this.embeddingsCache.set(cacheKey, embeddings);
    return embeddings;
  }

  /**
   * Compute TF-IDF based similarity
   * @private
   */
  _computeTfidfSimilarity(embeddings) {
    if (!embeddings[0] || !embeddings[1]) return 0;
    return Embedding.cosineSimilarity(embeddings[0], embeddings[1]);
  }

  /**
   * Compute hash-based similarity
   * @private
   */
  _computeHashSimilarity(embeddings) {
    if (!embeddings[0] || !embeddings[1]) return 0;
    return Embedding.cosineSimilarity(embeddings[0], embeddings[1]);
  }

  /**
   * Clear the embeddings cache
   */
  clearCache() {
    this.embeddingsCache.clear();
  }

  /**
   * Get configuration
   * @returns {Object}
   */
  getConfig() {
    return {
      method: this.method,
      hashDimensions: this.hashDimensions,
      threshold: this.threshold
    };
  }
}

module.exports = SemanticMatcher;
