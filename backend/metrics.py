"""Metrics collection utilities for LLM Council."""

from collections import defaultdict
from typing import Dict, List, Any, Optional
from datetime import datetime
import statistics


class ResponseTimeTracker:
    """
    Track response times for models.
    """

    def __init__(self):
        """Initialize the tracker."""
        self.times: Dict[str, List[float]] = defaultdict(list)

    def record(self, model: str, duration_seconds: float):
        """
        Record a response time for a model.

        Args:
            model: Model identifier
            duration_seconds: Response time in seconds
        """
        self.times[model].append(duration_seconds)

    def get_stats(self, model: str) -> Dict[str, Any]:
        """
        Get statistics for a model.

        Args:
            model: Model identifier

        Returns:
            Dict with timing statistics
        """
        times = self.times.get(model, [])

        if not times:
            return {
                "count": 0,
                "average": 0.0,
                "min": 0.0,
                "max": 0.0,
                "median": 0.0,
                "std_dev": 0.0
            }

        return {
            "count": len(times),
            "average": sum(times) / len(times),
            "min": min(times),
            "max": max(times),
            "median": statistics.median(times) if times else 0.0,
            "std_dev": statistics.stdev(times) if len(times) > 1 else 0.0
        }

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all models.

        Returns:
            Dict mapping model to statistics
        """
        return {model: self.get_stats(model) for model in self.times}

    def clear(self, model: Optional[str] = None):
        """
        Clear recorded times.

        Args:
            model: Optional model to clear (None clears all)
        """
        if model:
            self.times[model] = []
        else:
            self.times.clear()


class CouncilMetrics:
    """
    Track metrics for council deliberations.
    """

    def __init__(self):
        """Initialize the metrics tracker."""
        self.deliberations: List[Dict[str, Any]] = []
        self.response_tracker = ResponseTimeTracker()

    def record_deliberation(
        self,
        query: str,
        stage1_count: int,
        stage2_count: int,
        stage3_success: bool,
        total_duration: float
    ):
        """
        Record a completed deliberation.

        Args:
            query: User query
            stage1_count: Number of Stage 1 responses
            stage2_count: Number of Stage 2 rankings
            stage3_success: Whether Stage 3 succeeded
            total_duration: Total deliberation time in seconds
        """
        self.deliberations.append({
            "timestamp": datetime.utcnow().isoformat(),
            "query_length": len(query),
            "stage1_count": stage1_count,
            "stage2_count": stage2_count,
            "stage3_success": stage3_success,
            "total_duration": total_duration
        })

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of all metrics.

        Returns:
            Dict with metric summary
        """
        if not self.deliberations:
            return {
                "total_deliberations": 0,
                "average_duration": 0.0,
                "success_rate": 0.0,
                "response_times": {}
            }

        total = len(self.deliberations)
        avg_duration = sum(d['total_duration'] for d in self.deliberations) / total
        success_count = sum(1 for d in self.deliberations if d['stage3_success'])

        return {
            "total_deliberations": total,
            "average_duration": round(avg_duration, 2),
            "success_rate": round(success_count / total, 3),
            "response_times": self.response_tracker.get_all_stats()
        }
