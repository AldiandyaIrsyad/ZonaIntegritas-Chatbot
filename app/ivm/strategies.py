"""
Relevance evaluation strategies for the Input Validation Module (IVM).
"""
import math
from typing import List

from app.core.interfaces.infra import SearchResult
from app.core.interfaces.ivm import IRelevanceStrategy


class TopOneStrategy(IRelevanceStrategy):
    """Original strategy: checks if the best single score meets the threshold."""

    def evaluate(self, results: List[SearchResult], similarity_threshold: float) -> bool:
        if not results:
            return True # Empty KB case
        return results[0].score >= similarity_threshold


class SilhouetteKNNStrategy(IRelevanceStrategy):
    """k-NN voting strategy that calculates a Silhouette-inspired density score.
    
    Since we only have the distance (similarity) from the query to the top-K chunks,
    we compute a score based on the mean similarity penalized by the variance among
    the top-K. A query floating in empty space far from a tight "safe" cluster will
    have a lower mean score and potentially higher variance if it's roughly equidistant 
    from multiple disparate clusters.
    """

    def evaluate(self, results: List[SearchResult], similarity_threshold: float) -> bool:
        if not results:
            return True # Empty KB case
            
        scores = [res.score for res in results]
        k = len(scores)
        
        if k == 1:
            return scores[0] >= similarity_threshold
            
        mean_score = sum(scores) / k
        variance = sum((s - mean_score) ** 2 for s in scores) / k
        std_dev = math.sqrt(variance)
        
        # Silhouette-inspired penalty: 
        # If std_dev is high, the results are spread out (query is between clusters).
        # We penalize the mean score by the standard deviation to ensure it's near a tight cluster.
        # This effectively acts as a lower-bound confidence check.
        effective_score = mean_score - (0.5 * std_dev)
        
        return effective_score >= similarity_threshold
