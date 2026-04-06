"""
FLOW-HEAL Semantic Noise Senser Module

The NoiseSenser class detects semantic drift and uncertainty in reasoning steps,
providing critical input to the FLOW-HEAL self-correcting pipeline.

This module implements techniques from the DenoiseFlow paper to:
1. Measure semantic similarity between consecutive reasoning steps using embedding distance
2. Quantify uncertainty through multiple indicators (probability scores, coherence, etc.)
3. Identify potential "broken" nodes in the Dynamic Interaction Graph (DIG)

The senser works by analyzing the reasoning pipeline in real-time, comparing each step
to its predecessors and detecting when the reasoning has veered off course (semantic drift).
It provides quantitative metrics that the FlowRegulator uses to pinpoint root causes for healing.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List, Tuple, Union, Set
from datetime import datetime
import numpy as np
from numpy.linalg import norm
import warnings

# Import FLOW-HEAL core components
from .payload import AgentRole

# Try to import embedding models, fallback to simple methods if unavailable
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_MODEL_AVAILABLE = True
except ImportError:
    EMBEDDING_MODEL_AVAILABLE = False
    warnings.warn("SentenceTransformers not available. Using TF-IDF as fallback.")

# Try to import nltk for text processing
try:
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
    # Download NLTK resources if needed
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')
except ImportError:
    NLTK_AVAILABLE = False
    warnings.warn("NLTK not available. Some text processing features will be disabled.")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2) + 1e-10)


def _jaccard_similarity(text1: str, text2: str) -> float:
    """Calculate Jaccard similarity between two texts based on word overlap."""
    if not NLTK_AVAILABLE:
        return 0.0
    
    words1 = set(word_tokenize(text1.lower()))
    words2 = set(word_tokenize(text2.lower()))
    
    if not words1 and not words2:
        return 1.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    return len(intersection) / len(union)


def _text_length_ratio(text1: str, text2: str) -> float:
    """Calculate the ratio of text lengths (normalized)."""
    len1 = len(text1)
    len2 = len(text2)
    
    if len1 == 0 or len2 == 0:
        return 0.0
    
    return min(len1, len2) / max(len1, len2)


# ============================================================================
# NOISE SENSER CLASS
# ============================================================================

class NoiseSenser:
    """
    Semantic Noise Senser for detecting drift and uncertainty in reasoning steps.
    
    The NoiseSenser uses embedding distance metrics and uncertainty quantification
    to identify when a reasoning step has deviated significantly from expected paths.
    
    Attributes:
        embedding_model: Sentence embedding model (default: SentenceTransformers)
        similarity_threshold: Minimum semantic similarity score to be considered on-topic
        uncertainty_threshold: Threshold for flagging high uncertainty
        use_tfidf_fallback: Whether to use TF-IDF if embedding model is unavailable
        history_window: Number of previous steps to consider for context
    """
    
    def __init__(
        self,
        embedding_model: Optional[Any] = None,
        similarity_threshold: float = 0.65,
        uncertainty_threshold: float = 0.7,
        use_tfidf_fallback: bool = True,
        history_window: int = 3
    ) -> None:
        """
        Initialize the NoiseSenser.
        
        Args:
            embedding_model: Pretrained embedding model for semantic similarity.
                If None, attempts to load a default model or uses TF-IDF fallback.
            similarity_threshold: Threshold for semantic similarity (0-1).
                Steps with similarity below this may be flagged as drift.
            uncertainty_threshold: Threshold for uncertainty scores.
                Steps above this may require healing attention.
            use_tfidf_fallback: If True, uses TF-IDF vectorizer when embeddings unavailable.
            history_window: Number of previous steps to consider for context analysis.
        """
        self.similarity_threshold = similarity_threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.use_tfidf_fallback = use_tfidf_fallback
        self.history_window = history_window
        
        # Load or initialize embedding model
        if embedding_model is not None:
            self.embedding_model = embedding_model
        elif EMBEDDING_MODEL_AVAILABLE:
            try:
                # Use a lightweight multilingual model by default
                self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            except:
                warnings.warn("Could not load default embedding model. Using TF-IDF fallback.")
                self.embedding_model = None
        else:
            self.embedding_model = None
        
        # Initialize TF-IDF vectorizer for fallback
        if use_tfidf_fallback:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.tfidf_vectorizer = TfidfVectorizer(stop_words='english' if NLTK_AVAILABLE else None)
        else:
            self.tfidf_vectorizer = None
        
        # Cache for embeddings to avoid recomputation
        self._embedding_cache: Dict[str, np.ndarray] = {}
    
    def _get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """
        Get embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors (numpy arrays)
        """
        embeddings = []
        
        for text in texts:
            if text in self._embedding_cache:
                embeddings.append(self._embedding_cache[text])
                continue
            
            if self.embedding_model:
                try:
                    embedding = self.embedding_model.encode(text, convert_to_numpy=True)
                    self._embedding_cache[text] = embedding
                    embeddings.append(embedding)
                except Exception as e:
                    warnings.warn(f"Embedding model error: {e}. Falling back to TF-IDF.")
                    self.embedding_model = None
                    return self._get_embeddings(texts)  # Retry with fallback
            else:
                # Fallback to TF-IDF or simple bag-of-words
                if self.tfidf_vectorizer:
                    # Fit on all texts if not already fitted
                    try:
                        self.tfidf_vectorizer.fit(texts)
                    except:
                        pass
                    vec = self.tfidf_vectorizer.transform([text]).toarray().flatten()
                    embeddings.append(vec)
                else:
                    # Simple count vectorization
                    words = text.lower().split()
                    vec = np.zeros(100)  # Fixed dimension
                    for i, word in enumerate(words[:100]):
                        vec[i] = hash(word) % 100
                    embeddings.append(vec)
        
        return embeddings
    
    def calculate_semantic_similarity(
        self,
        text1: str,
        text2: str,
        method: Literal['cosine', 'jaccard', 'combined'] = 'combined'
    ) -> float:
        """
        Calculate semantic similarity between two texts.
        
        Args:
            text1: First text string
            text2: Second text string
            method: Similarity calculation method
            
        Returns:
            Similarity score between 0 and 1 (1 = identical, 0 = completely different)
        """
        if method == 'cosine':
            # Use embedding-based cosine similarity
            embeddings = self._get_embeddings([text1, text2])
            if len(embeddings) == 2:
                return _cosine_similarity(embeddings[0], embeddings[1])
            else:
                return 0.0
        
        elif method == 'jaccard':
            # Use Jaccard index on words
            return _jaccard_similarity(text1, text2)
        
        else:  # combined
            # Combine multiple similarity measures
            cosine_sim = self.calculate_semantic_similarity(text1, text2, method='cosine')
            jaccard_sim = self.calculate_semantic_similarity(text1, text2, method='jaccard')
            length_ratio = _text_length_ratio(text1, text2)
            
            # Weighted combination
            weights = {'cosine': 0.6, 'jaccard': 0.3, 'length': 0.1}
            combined = (
                cosine_sim * weights['cosine'] +
                jaccard_sim * weights['jaccard'] +
                length_ratio * weights['length']
            )
            return combined
    
    def detect_semantic_drift(
        self,
        current_step: ReasoningStep,
        previous_step: Optional[ReasoningStep] = None,
        next_step: Optional[ReasoningStep] = None,
        context_steps: Optional[List[ReasoningStep]] = None
    ) -> Dict[str, Any]:
        """
        Detect semantic drift between reasoning steps.
        
        Args:
            current_step: The current reasoning step to analyze
            previous_step: The immediately preceding reasoning step
            next_step: The following reasoning step (optional)
            context_steps: Additional context steps (optional)
            
        Returns:
            Dictionary with drift detection results and metrics
        """
        results = {
            'drift_detected': False,
            'confidence': 0.0,
            'similarity_scores': {},
            'drift_indicators': [],
            'recommendations': []
        }
        
        current_content = current_step.content
        
        # Compare with previous step if available
        if previous_step:
            prev_content = previous_step.content
            similarity = self.calculate_semantic_similarity(current_content, prev_content)
            
            # Check if similarity is below threshold
            results['similarity_scores']['previous'] = similarity
            if similarity < self.similarity_threshold:
                drift_confidence = (self.similarity_threshold - similarity) / self.similarity_threshold
                results['drift_detected'] = True
                results['confidence'] = max(results['confidence'], drift_confidence)
                results['drift_indicators'].append('low_similarity_to_previous')
            
            # Check for topic shift using Jaccard similarity
            jaccard_sim = _jaccard_similarity(current_content, prev_content)
            results['similarity_scores']['jaccard_previous'] = jaccard_sim
            if jaccard_sim < 0.3:
                results['drift_detected'] = True
                results['confidence'] = max(results['confidence'], 0.3)
                results['drift_indicators'].append('low_jaccard_similarity')
        
        # Compare with next step if available (forward-looking drift)
        if next_step:
            next_content = next_step.content
            similarity = self.calculate_semantic_similarity(current_content, next_content)
            results['similarity_scores']['next'] = similarity
            
            # Check if current step is a poor precursor to next step
            if similarity < self.similarity_threshold * 0.8:
                drift_confidence = (self.similarity_threshold * 0.8 - similarity) / (self.similarity_threshold * 0.8)
                results['drift_detected'] = True
                results['confidence'] = max(results['confidence'], drift_confidence)
                results['drift_indicators'].append('poor_transition_to_next')
        
        # Compare with context steps if provided
        if context_steps:
            for i, context_step in enumerate(context_steps[:self.history_window]):
                context_content = context_step.content
                similarity = self.calculate_semantic_similarity(current_content, context_content)
                results['similarity_scores'][f'context_{i}'] = similarity
                
                if similarity < self.similarity_threshold * 0.9:
                    drift_confidence = (self.similarity_threshold * 0.9 - similarity) / (self.similarity_threshold * 0.9)
                    results['drift_detected'] = True
                    results['confidence'] = max(results['confidence'], drift_confidence)
                    results['drift_indicators'].append(f'low_context_similarity_{i}')
        
        # Check for abrupt content change (length-based)
        if previous_step:
            length_ratio = _text_length_ratio(current_content, previous_step.content)
            results['similarity_scores']['length_ratio'] = length_ratio
            if length_ratio < 0.3 or length_ratio > 3.0:
                results['drift_detected'] = True
                results['confidence'] = max(results['confidence'], 0.2)
                results['drift_indicators'].append('abrupt_length_change')
        
        # Generate recommendations based on drift indicators
        if results['drift_detected']:
            if 'low_similarity_to_previous' in results['drift_indicators']:
                results['recommendations'].append('recheck_premises')
            if 'poor_transition_to_next' in results['drift_indicators']:
                results['recommendations'].append('revisit_conclusion')
            if 'abrupt_length_change' in results['drift_indicators']:
                results['recommendations'].append('verify_step_completeness')
        
        # Normalize confidence to 0-1 range
        results['confidence'] = min(results['confidence'], 1.0)
        
        return results
    
    def quantify_uncertainty(
        self,
        step: ReasoningStep,
        model_probs: Optional[Dict[str, float]] = None,
        include_context: bool = True
    ) -> float:
        """
        Quantify uncertainty for a reasoning step using multiple indicators.
        
        Args:
            step: The reasoning step to analyze
            model_probs: Probability distribution from language model (if available)
            include_context: Whether to include contextual uncertainty indicators
            
        Returns:
            Uncertainty score between 0 (certain) and 1 (uncertain)
        """
        uncertainty_sources = []
        
        # 1. Built-in uncertainty score from the step itself
        uncertainty_sources.append(step.uncertainty_score)
        
        # 2. Coherence score (lower coherence = higher uncertainty)
        uncertainty_sources.append(1 - step.coherence_score)
        
        # 3. Logic score (lower logic = higher uncertainty)
        uncertainty_sources.append(1 - step.logic_score)
        
        # 4. Content-based uncertainty indicators
        content = step.content
        
        # Count hedging language (could, perhaps, maybe, etc.)
        hedging_words = ['could', 'perhaps', 'maybe', 'might', 'possibly', 'probable', 'likely']
        content_lower = content.lower()
        hedging_count = sum(1 for word in hedging_words if word in content_lower)
        hedging_score = min(hedging_count / 5, 1.0)  # Normalize to 0-1
        uncertainty_sources.append(hedging_score)
        
        # Count first-person uncertainty expressions
        uncertainty_phrases = ['i think', 'i believe', 'i guess', 'not sure', 'difficult to say']
        phrase_count = sum(1 for phrase in uncertainty_phrases if phrase in content_lower)
        phrase_score = min(phrase_count / 3, 1.0)
        uncertainty_sources.append(phrase_score)
        
        # 5. Check for question marks or self-correction
        question_count = content_lower.count('?')
        correction_count = content_lower.count('but i')
        self_correction_score = min((question_count + correction_count) / 5, 1.0)
        uncertainty_sources.append(self_correction_score)
        
        # 6. Model probability entropy (if available)
        if model_probs:
            # Calculate entropy of the probability distribution
            entropy = -sum(p * np.log(p + 1e-10) for p in model_probs.values())
            max_entropy = np.log(len(model_probs))
            entropy_ratio = entropy / max_entropy if max_entropy > 0 else 0.0
            uncertainty_sources.append(entropy_ratio)
        
        # 7. Contextual uncertainty from DIG (if requested)
        if include_context:
            # Check if this step is a leaf node with no children (might indicate incomplete reasoning)
            # This would be determined from the DIG, but we don't have DIG access here
            # We'll add a placeholder for this source
            pass
        
        # Combine sources with weights
        weights = {
            'step_uncertainty': 0.25,
            'coherence': 0.15,
            'logic': 0.15,
            'hedging': 0.15,
            'phrases': 0.10,
            'self_correction': 0.10,
            'entropy': 0.10
        }
        
        weighted_sum = 0.0
        for i, source in enumerate(uncertainty_sources):
            weight_name = list(weights.keys())[i] if i < len(weights) else 'other'
            if weight_name in weights:
                weighted_sum += source * weights[weight_name]
        
        return weighted_sum
    
    def analyze_step(
        self,
        current_step: ReasoningStep,
        previous_steps: List[ReasoningStep],
        dig: Optional[ReasoningDIG] = None,
        model_probs: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive analysis of a reasoning step in context.
        
        Args:
            current_step: The current reasoning step to analyze
            previous_steps: List of previous reasoning steps in order
            dig: Dynamic Interaction Graph (if available)
            model_probs: Probability distribution from language model (if available)
            
        Returns:
            Comprehensive analysis with drift detection, uncertainty quantification,
            and recommendations for the FLOW-HEAL pipeline.
        """
        analysis = {
            'step_id': current_step.step_id,
            'agent_id': current_step.agent_id,
            'agent_role': current_step.agent_role.value,
            'analysis_timestamp': datetime.now().isoformat(),
            'drift_analysis': {},
            'uncertainty_score': 0.0,
            'quality_metrics': {},
            'dig_context': {},
            'flags': [],
            'recommendations': []
        }
        
        # Get context steps (previous steps within history window)
        context_steps = previous_steps[-self.history_window:] if previous_steps else []
        
        # Detect semantic drift
        previous_step = previous_steps[-1] if previous_steps else None
        next_step = None  # Would need access to future steps, not typically available
        
        drift_results = self.detect_semantic_drift(
            current_step=current_step,
            previous_step=previous_step,
            next_step=next_step,
            context_steps=context_steps
        )
        analysis['drift_analysis'] = drift_results
        
        # Quantify uncertainty
        uncertainty_score = self.quantify_uncertainty(current_step, model_probs)
        analysis['uncertainty_score'] = uncertainty_score
        
        # Calculate quality metrics
        quality_metrics = {
            'built_in_quality': current_step.calculate_overall_quality(),
            'drift_confidence': drift_results['confidence'],
            'uncertainty_normalized': uncertainty_score,
            'semantic_coherence': 1 - drift_results['confidence'],  # Inverse relationship
        }
        analysis['quality_metrics'] = quality_metrics
        
        # Determine flags
        if drift_results['drift_detected']:
            analysis['flags'].append('semantic_drift')
            analysis['flags'].append(f'drift_confidence_{drift_results["confidence"]:.2f}')
        
        if uncertainty_score > self.uncertainty_threshold:
            analysis['flags'].append('high_uncertainty')
            analysis['flags'].append(f'uncertainty_{uncertainty_score:.2f}')
        
        # Generate recommendations
        analysis['recommendations'].extend(drift_results['recommendations'])
        
        # If step is from a healer or validator, add specific recommendations
        if current_step.agent_role in [AgentRole.HEALER, AgentRole.VALIDATOR]:
            analysis['recommendations'].append('targeted_healing')
        
        # Add DIG context if available
        if dig and current_step.step_id in dig.nodes:
            node = dig.get_node(current_step.step_id)
            if node:
                dig_info = {
                    'in_degree': node.causal_in_degree,
                    'out_degree': node.causal_out_degree,
                    'quality_score': node.quality_score,
                    'importance': dig.calculate_node_importance(current_step.step_id)
                }
                analysis['dig_context'] = dig_info
                
                # Add importance-based flags
                importance = dig_info.get('importance', {})
                if importance.get('betweenness_centrality', 0) > 0.5:
                    analysis['flags'].append('critical_node')
                if importance.get('connectivity_score', 0) > 0.7:
                    analysis['flags'].append('highly_connected')
        
        return analysis
    
    def update_step_with_analysis(
        self,
        step: ReasoningStep,
        analysis: Dict[str, Any],
        dig: Optional[ReasoningDIG] = None
    ) -> ReasoningStep:
        """
        Update a reasoning step with analysis results from the NoiseSenser.
        
        Args:
            step: The reasoning step to update
            analysis: Analysis results from analyze_step()
            dig: Dynamic Interaction Graph (if available)
            
        Returns:
            Updated ReasoningStep with new metrics and flags
        """
        # Update uncertainty score
        if 'uncertainty_score' in analysis:
            step.uncertainty_score = analysis['uncertainty_score']
        
        # Update coherence and logic scores based on drift detection
        drift_confidence = analysis['drift_analysis'].get('confidence', 0.0)
        if drift_confidence > 0.5:
            # Reduce coherence score if significant drift detected
            step.coherence_score = max(0.0, step.coherence_score - drift_confidence * 0.2)
        
        # Add metadata with analysis results
        step.metadata['noise_senser_analysis'] = {
            'timestamp': analysis.get('analysis_timestamp'),
            'drift_detected': analysis['drift_analysis'].get('drift_detected', False),
            'drift_confidence': float(drift_confidence),
            'uncertainty_score': float(analysis['uncertainty_score']),
            'flags': analysis.get('flags', []),
            'recommendations': analysis.get('recommendations', [])
        }
        
        # Update DIG node if available
        if dig and step.step_id in dig.nodes:
            node = dig.nodes[step.step_id]
            node.uncertainty_score = analysis['uncertainty_score']
            node.coherence_score = step.coherence_score
            node.logic_score = step.logic_score
            node.quality_score = step.calculate_overall_quality()
            # Update node in graph
            dig.graph.nodes[step.step_id]['node_data'] = node.to_dict()
        
        return step
    
    def batch_analyze(
        self,
        steps: List[ReasoningStep],
        dig: Optional[ReasoningDIG] = None,
        model_probs_list: Optional[List[Dict[str, float]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple reasoning steps in batch.
        
        Args:
            steps: List of reasoning steps to analyze
            dig: Dynamic Interaction Graph (if available)
            model_probs_list: List of probability distributions corresponding to each step
            
        Returns:
            List of analysis dictionaries for each step
        """
        analyses = []
        model_probs_list = model_probs_list or [None] * len(steps)
        
        for i, step in enumerate(steps):
            # Get previous steps for context
            previous_steps = steps[:i]
            model_probs = model_probs_list[i] if i < len(model_probs_list) else None
            
            analysis = self.analyze_step(
                current_step=step,
                previous_steps=previous_steps,
                dig=dig,
                model_probs=model_probs
            )
            analyses.append(analysis)
            
            # Update step in place
            self.update_step_with_analysis(step, analysis, dig)
        
        return analyses
    
    def visualize_drift_metrics(
        self,
        analysis_results: List[Dict[str, Any]],
        output_format: Literal['text', 'json', 'markdown'] = 'markdown'
    ) -> str:
        """
        Create a visual representation of drift metrics.
        
        Args:
            analysis_results: List of analysis results from analyze_step()
            output_format: Format for the output
            
        Returns:
            String representation of the drift metrics
        """
        if output_format == 'markdown':
            lines = []
            lines.append("# Semantic Drift Analysis Report")
            lines.append(f"Generated: {datetime.now().isoformat()}")
            lines.append(f"Total Steps Analyzed: {len(analysis_results)}")
            lines.append("")
            
            for i, analysis in enumerate(analysis_results):
                lines.append(f"## Step {i+1}: {analysis['step_id']}")
                lines.append(f"Agent: {analysis['agent_id']} ({analysis['agent_role']})")
                
                # Drift analysis
                drift = analysis['drift_analysis']
                lines.append(f"- **Drift Detected**: {drift['drift_detected']}")
                lines.append(f"- **Drift Confidence**: {drift['confidence']:.3f}")
                lines.append(f"- **Uncertainty Score**: {analysis['uncertainty_score']:.3f}")
                
                # Flags
                if analysis['flags']:
                    lines.append(f"- **Flags**: {', '.join(analysis['flags'])}")
                
                # Recommendations
                if analysis['recommendations']:
                    lines.append("- **Recommendations**:")
                    for rec in analysis['recommendations']:
                        lines.append(f"  - {rec}")
                
                lines.append("")
            
            return '\n'.join(lines)
        
        elif output_format == 'json':
            import json
            return json.dumps(analysis_results, indent=2, default=str)
        
        else:  # text
            lines = []
            for i, analysis in enumerate(analysis_results):
                lines.append(f"Step {i+1}: {analysis['step_id']} [{analysis['agent_role']}]")
                lines.append(f"  Drift: {analysis['drift_analysis']['drift_detected']} (conf: {analysis['drift_analysis']['confidence']:.2f})")
                lines.append(f"  Uncertainty: {analysis['uncertainty_score']:.2f}")
                if analysis['flags']:
                    lines.append(f"  Flags: {', '.join(analysis['flags'])}")
            return '\n'.join(lines)


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_noise_senser(
    model_name: Optional[str] = None,
    **kwargs: Any
) -> NoiseSenser:
    """
    Factory function to create a NoiseSenser instance.
    
    Args:
        model_name: Name of the embedding model to use (if None, uses default)
        **kwargs: Additional arguments to pass to NoiseSenser
        
    Returns:
        Configured NoiseSenser instance
    """
    if model_name:
        if EMBEDDING_MODEL_AVAILABLE:
            try:
                embedding_model = SentenceTransformer(model_name)
                return NoiseSenser(embedding_model=embedding_model, **kwargs)
            except:
                warnings.warn(f"Could not load model '{model_name}'. Using default or fallback.")
                pass
        else:
            warnings.warn("SentenceTransformers not available. Using TF-IDF fallback.")
    
    return NoiseSenser(**kwargs)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example 1: Basic usage
    print("=" * 60)
    print("EXAMPLE 1: Basic NoiseSenser Usage")
    print("=" * 60)
    
    # Create a NoiseSenser instance
    senser = NoiseSenser(similarity_threshold=0.7, uncertainty_threshold=0.7)
    
    # Create sample reasoning steps
    from uuid import uuid4
    
    step1 = ReasoningStep(
        step_id="step_1",
        agent_id="planner",
        agent_role=AgentRole.PLANNER,
        intent_hash="plan_initial",
        content="I will create an initial project plan for FLOW-HEAL.",
        uncertainty_score=0.2,
        coherence_score=0.9,
        logic_score=0.8
    )
    
    step2 = ReasoningStep(
        step_id="step_2",
        agent_id="researcher",
        agent_role=AgentRole.RESEARCHER,
        intent_hash="research_requirements",
        content="I will research the requirements for the FLOW-HEAL system from the three source papers.",
        uncertainty_score=0.3,
        coherence_score=0.8,
        logic_score=0.7
    )
    
    step3 = ReasoningStep(
        step_id="step_3",
        agent_id="thinker",
        agent_role=AgentRole.THINKER,
        intent_hash="design_architecture",
        content="Based on the research, I will design the overall architecture for FLOW-HEAL.",
        uncertainty_score=0.5,
        coherence_score=0.7,
        logic_score=0.6
    )
    
    # Analyze steps
    print("\nAnalyzing semantic drift between steps...")
    drift_result = senser.detect_semantic_drift(
        current_step=step2,
        previous_step=step1,
        next_step=step3
    )
    print(f"Drift detected: {drift_result['drift_detected']}")
    print(f"Confidence: {drift_result['confidence']:.3f}")
    print(f"Similarity to previous: {drift_result['similarity_scores'].get('previous', 'N/A'):.3f}")
    
    # Quantify uncertainty for a step
    print("\nUncertainty quantification for step 3:")
    uncertainty = senser.quantify_uncertainty(step3)
    print(f"Uncertainty score: {uncertainty:.3f}")
    
    # Batch analysis
    print("\nBatch analysis of all steps:")
    analyses = senser.batch_analyze([step1, step2, step3])
    for i, analysis in enumerate(analyses):
        print(f"Step {i+1}: Drift={analysis['drift_analysis']['drift_detected']}, "
              f"Uncertainty={analysis['uncertainty_score']:.2f}")
    
    # Generate report
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Drift Metrics Visualization")
    print("=" * 60)
    report = senser.visualize_drift_metrics(analyses, output_format='markdown')
    print(report)