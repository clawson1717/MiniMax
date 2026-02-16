/**
 * Demo Script
 * 
 * Demonstrates the core functionality of the Dynamic Negotiation Network
 */

const { Agent, SemanticMatcher, Embedding } = require('./index');

console.log('='.repeat(60));
console.log('Dynamic Negotiation Network - Demo');
console.log('='.repeat(60));

// Create agents with different needs and offers
console.log('\n--- Creating Agents ---\n');

const buyer = new Agent('buyer_1', {
  needs: ['discount', 'fast delivery', 'quality product', 'warranty'],
  offers: ['budget: $500', 'timeline: 2 weeks', 'payment: immediate'],
  strategy: 'flexible'
});

const seller = new Agent('seller_1', {
  needs: ['profit margin', 'quick payment', 'bulk order', 'good reviews'],
  offers: ['price: $600', 'product: premium quality', 'delivery: 1 week'],
  strategy: 'profit-focused'
});

const competitor = new Agent('competitor_1', {
  needs: ['market share', 'customer satisfaction'],
  offers: ['price: $550', 'product: good quality', 'delivery: 3 days'],
  strategy: 'aggressive'
});

console.log(`Created agent: ${buyer.id}`);
console.log(`  Needs: ${buyer.getNeeds().join(', ')}`);
console.log(`  Offers: ${buyer.getOffers().join(', ')}`);

console.log(`\nCreated agent: ${seller.id}`);
console.log(`  Needs: ${seller.getNeeds().join(', ')}`);
console.log(`  Offers: ${seller.getOffers().join(', ')}`);

// Demonstrate semantic matching
console.log('\n--- Semantic Matching ---\n');

const matcher = new SemanticMatcher({
  method: 'tfidf',
  threshold: 0.2
});

// Compute direct similarity between needs and offers
const need = 'fast delivery';
const offer = 'delivery: 1 week';
const similarity = matcher.computeSimilarity(need, offer);
console.log(`Similarity between "${need}" and "${offer}": ${similarity.toFixed(4)}`);

// Compute similarity between arrays
const buyerNeeds = buyer.getNeeds();
const sellerOffers = seller.getOffers();
const avgSimilarity = matcher.computeSimilarityMatrix(buyerNeeds, sellerOffers);
console.log(`Average similarity (buyer needs vs seller offers): ${avgSimilarity.toFixed(4)}`);

// Find best match
console.log('\n--- Finding Best Matches ---\n');

const bestMatch = matcher.findBestMatch('discount', [
  'price: $600', 
  'discount: 10%', 
  'warranty: 1 year'
]);
console.log(`Best match for "discount":`);
console.log(`  Offer: ${bestMatch.offer}`);
console.log(`  Score: ${bestMatch.score.toFixed(4)}`);
console.log(`  Meets threshold: ${bestMatch.meetsThreshold}`);

// Find compatible agents
console.log('\n--- Agent Compatibility ---\n');

const candidates = [seller, competitor];
const compatible = matcher.findCompatibleAgents(buyer, candidates);

compatible.forEach(match => {
  console.log(`Compatible with ${match.agent.id}:`);
  console.log(`  Score: ${match.score.toFixed(4)}`);
  console.log(`  Need->Offer: ${match.needOfferScore.toFixed(4)}`);
  console.log(`  Offer->Need: ${match.offerNeedScore.toFixed(4)}`);
});

// Build compatibility matrix
console.log('\n--- Compatibility Matrix ---\n');

const agents = [buyer, seller, competitor];
const matrix = matcher.buildCompatibilityMatrix(agents);

console.log('Agent'.padEnd(15), 'buyer_1'.padEnd(12), 'seller_1'.padEnd(12), 'competitor_1');
agents.forEach((agent, i) => {
  const row = matrix[i].map(v => v.toFixed(2).padStart(6)).join('  ');
  console.log(agent.id.padEnd(15), row);
});

// Demonstrate embedding methods
console.log('\n--- Embedding Methods ---\n');

const texts = ['fast delivery', 'discount price', 'quality product'];

console.log('TF-IDF Embeddings:');
const tfidfEmbeds = Embedding.tfidf(texts);
console.log(`  "${texts[0]}": [${tfidfEmbeds[0].slice(0, 5).join(', ')}...]`);

console.log('\nHash Embeddings:');
const hashEmbeds = Embedding.hash(texts, 32);
console.log(`  "${texts[0]}": [${hashEmbeds[0].slice(0, 5).join(', ')}...]`);

// Demonstrate agent belief updates (Reactive Knowledge Representation)
console.log('\n--- Belief Updates ---\n');

buyer.updateBelief('seller_1', {
  willingnessToNegotiate: 0.8,
  priceFlexibility: 'medium'
});

const belief = buyer.getBelief('seller_1');
console.log(`Buyer's belief about Seller:`);
console.log(`  ${JSON.stringify(belief)}`);

// Demonstrate offer response
console.log('\n--- Offer Response ---\n');

const response = buyer.respondToOffer('discount: 15%');
console.log(`Received offer: "discount: 15%"`);
console.log(`Response: ${response.action}`);
console.log(`Message: ${response.message}`);

console.log('\n' + '='.repeat(60));
console.log('Demo Complete!');
console.log('='.repeat(60));
