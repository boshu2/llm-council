"""Council deliberation utilities for LLM Council."""

from typing import List, Dict, Any, Optional
from collections import Counter


def filter_successful_models(
    all_models: List[str],
    stage1_results: List[Dict[str, Any]]
) -> List[str]:
    """
    Filter to only models that succeeded in Stage 1.

    Args:
        all_models: List of all council models
        stage1_results: Results from Stage 1

    Returns:
        List of successful model identifiers
    """
    successful_models = {r['model'] for r in stage1_results if r.get('response')}
    return [m for m in all_models if m in successful_models]


def identify_minority_opinions(
    stage2_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Identify models with minority opinions in rankings.

    Args:
        stage2_results: Rankings from each model

    Returns:
        Dict with minority opinion information
    """
    if not stage2_results or len(stage2_results) < 2:
        return {
            "has_minority": False,
            "dissenting_models": [],
            "consensus_ranking": []
        }

    # Get first-place votes
    first_place_votes = []
    for result in stage2_results:
        parsed = result.get('parsed_ranking', [])
        if parsed:
            first_place_votes.append((result['model'], parsed[0]))

    if not first_place_votes:
        return {
            "has_minority": False,
            "dissenting_models": [],
            "consensus_ranking": []
        }

    # Find the majority first-place choice
    first_choices = [v[1] for v in first_place_votes]
    vote_counts = Counter(first_choices)
    majority_choice = vote_counts.most_common(1)[0][0]

    # Find dissenting models
    dissenting = [
        model for model, choice in first_place_votes
        if choice != majority_choice
    ]

    # Determine consensus ranking (most common order)
    all_rankings = [r.get('parsed_ranking', []) for r in stage2_results if r.get('parsed_ranking')]

    return {
        "has_minority": len(dissenting) > 0,
        "dissenting_models": dissenting,
        "majority_choice": majority_choice,
        "vote_distribution": dict(vote_counts)
    }


def extract_reasoning(response: Dict[str, Any]) -> Optional[str]:
    """
    Extract chain-of-thought reasoning from a response.

    Args:
        response: Model response dict

    Returns:
        Reasoning text if available
    """
    if not response:
        return None

    # Check for explicit reasoning field
    reasoning = response.get('reasoning_details')
    if reasoning:
        return reasoning

    # Check for reasoning in content
    content = response.get('content', '')
    if '<thinking>' in content:
        # Extract thinking tags
        import re
        match = re.search(r'<thinking>(.*?)</thinking>', content, re.DOTALL)
        if match:
            return match.group(1).strip()

    return None


def calculate_model_credibility(
    historical_rankings: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Calculate credibility scores for models based on historical performance.

    Args:
        historical_rankings: List of historical ranking data

    Returns:
        Dict mapping model to credibility score
    """
    from collections import defaultdict

    if not historical_rankings:
        return {}

    model_scores = defaultdict(list)

    for ranking_data in historical_rankings:
        aggregate = ranking_data.get('aggregate_rankings', [])
        total_models = len(aggregate)

        for rank_info in aggregate:
            model = rank_info.get('model')
            avg_rank = rank_info.get('average_rank', total_models)

            # Convert rank to score (1st place = 1.0, last = 0.0)
            if total_models > 1:
                score = 1.0 - (avg_rank - 1) / (total_models - 1)
            else:
                score = 1.0

            model_scores[model].append(score)

    # Calculate average credibility
    credibility = {}
    for model, scores in model_scores.items():
        if scores:
            credibility[model] = sum(scores) / len(scores)

    return credibility


def merge_rankings(
    rankings: List[List[str]],
    method: str = "borda"
) -> List[str]:
    """
    Merge multiple rankings into a single consensus ranking.

    Args:
        rankings: List of ranking lists
        method: Merging method (borda, plurality, etc.)

    Returns:
        Merged ranking list
    """
    if not rankings:
        return []

    if method == "plurality":
        # Count first-place votes
        first_places = [r[0] for r in rankings if r]
        vote_counts = Counter(first_places)
        return [item for item, _ in vote_counts.most_common()]

    # Default: Borda count
    all_items = set()
    for ranking in rankings:
        all_items.update(ranking)

    n_items = len(all_items)
    scores = Counter()

    for ranking in rankings:
        for position, item in enumerate(ranking):
            # Borda: n-1 points for 1st, n-2 for 2nd, etc.
            points = n_items - position - 1
            scores[item] += max(0, points)

    return [item for item, _ in scores.most_common()]


def validate_ranking_format(ranking_text: str) -> bool:
    """
    Validate that a ranking follows the expected format.

    Args:
        ranking_text: The ranking text to validate

    Returns:
        True if valid format
    """
    import re

    # Check for FINAL RANKING section
    if "FINAL RANKING:" not in ranking_text:
        return False

    # Check for numbered list format
    parts = ranking_text.split("FINAL RANKING:")
    if len(parts) < 2:
        return False

    ranking_section = parts[1]

    # Should have at least one "Response X" entry
    matches = re.findall(r'\d+\.\s*Response [A-Z]', ranking_section)

    return len(matches) > 0
