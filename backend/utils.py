"""Utility functions for LLM Council."""

import re
from typing import List, Dict, Any, Optional
from collections import Counter
import hashlib


# Feature 31: Input sanitization
def sanitize_input(text: str) -> str:
    """
    Sanitize user input by removing potentially harmful characters.

    Args:
        text: The input text to sanitize

    Returns:
        Sanitized text
    """
    if not text:
        return ""
    # Remove null bytes and other control characters except newlines and tabs
    sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    return sanitized.strip()


# Feature 32: Maximum message length validation
def validate_message_length(text: str, max_length: int = 50000) -> bool:
    """
    Validate that a message is within the allowed length.

    Args:
        text: The message text
        max_length: Maximum allowed length (default 50000 chars)

    Returns:
        True if valid, False if too long
    """
    return len(text) <= max_length


# Feature 33: Basic content filtering
PROFANITY_PATTERNS = []  # Placeholder - in production would have actual patterns

def check_content_appropriate(text: str) -> bool:
    """
    Basic content check (placeholder for more sophisticated filtering).

    Args:
        text: The content to check

    Returns:
        True if content passes basic checks
    """
    if not text:
        return True
    # Basic check - just ensure it's not empty after stripping
    return len(text.strip()) > 0


# Feature 36: Empty message validation
def validate_not_empty(text: str) -> bool:
    """
    Validate that a message is not empty or whitespace-only.

    Args:
        text: The message text

    Returns:
        True if not empty, False otherwise
    """
    return bool(text and text.strip())


# Feature 37: Conversation ID format validation
def validate_conversation_id(conversation_id: str) -> bool:
    """
    Validate that a conversation ID is in valid UUID format.

    Args:
        conversation_id: The ID to validate

    Returns:
        True if valid UUID format
    """
    uuid_pattern = re.compile(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
        re.IGNORECASE
    )
    return bool(uuid_pattern.match(conversation_id))


# Feature 39: Model name validation
def validate_model_name(model_name: str) -> bool:
    """
    Validate that a model name follows the expected format (provider/model).

    Args:
        model_name: The model name to validate

    Returns:
        True if valid format
    """
    if not model_name:
        return False
    # Format: provider/model-name (e.g., openai/gpt-4)
    pattern = re.compile(r'^[a-z0-9-]+/[a-z0-9.-]+$', re.IGNORECASE)
    return bool(pattern.match(model_name))


# Feature 40: Duplicate message detection
def compute_message_hash(content: str) -> str:
    """
    Compute a hash for a message to detect duplicates.

    Args:
        content: The message content

    Returns:
        SHA256 hash of the normalized content
    """
    normalized = content.lower().strip()
    return hashlib.sha256(normalized.encode()).hexdigest()


def is_duplicate_message(content: str, recent_hashes: List[str]) -> bool:
    """
    Check if a message is a duplicate of recent messages.

    Args:
        content: The message content
        recent_hashes: List of hashes from recent messages

    Returns:
        True if duplicate detected
    """
    message_hash = compute_message_hash(content)
    return message_hash in recent_hashes


# Feature 41: Token count estimation
def estimate_token_count(text: str) -> int:
    """
    Estimate the number of tokens in a text.
    Uses simple word-based estimation (roughly 1.3 tokens per word).

    Args:
        text: The text to estimate tokens for

    Returns:
        Estimated token count
    """
    if not text:
        return 0
    # Simple estimation: ~4 characters per token on average
    return max(1, len(text) // 4)


# Feature 42: Response word count
def count_words(text: str) -> int:
    """
    Count the number of words in a text.

    Args:
        text: The text to count words in

    Returns:
        Word count
    """
    if not text:
        return 0
    words = text.split()
    return len(words)


# Feature 43: Response similarity detection
def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate simple similarity between two texts using Jaccard similarity.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Similarity score between 0 and 1
    """
    if not text1 or not text2:
        return 0.0

    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if not words1 and not words2:
        return 1.0
    if not words1 or not words2:
        return 0.0

    intersection = words1 & words2
    union = words1 | words2

    return len(intersection) / len(union)


# Feature 44: Ranking agreement score
def calculate_ranking_agreement(rankings: List[List[str]]) -> float:
    """
    Calculate how much the rankings agree with each other.
    Uses a simplified agreement score based on position similarity.

    Args:
        rankings: List of ranking lists (each list is ordered best to worst)

    Returns:
        Agreement score between 0 (no agreement) and 1 (perfect agreement)
    """
    if not rankings or len(rankings) < 2:
        return 1.0

    # Get all unique items
    all_items = set()
    for ranking in rankings:
        all_items.update(ranking)

    if not all_items:
        return 1.0

    n_items = len(all_items)

    if n_items < 2:
        return 1.0

    # Check for perfect agreement first
    first_ranking = rankings[0]
    all_identical = all(r == first_ranking for r in rankings)
    if all_identical:
        return 1.0

    # Calculate agreement based on position differences
    total_agreement = 0.0
    pair_count = 0

    for i in range(len(rankings)):
        for j in range(i + 1, len(rankings)):
            # Compare rankings i and j
            r1 = rankings[i]
            r2 = rankings[j]

            # Calculate position-based similarity
            position_matches = 0.0
            for item in all_items:
                if item in r1 and item in r2:
                    pos1 = r1.index(item)
                    pos2 = r2.index(item)
                    # Max diff is n_items - 1
                    diff = abs(pos1 - pos2)
                    max_diff = n_items - 1
                    if max_diff > 0:
                        position_matches += 1.0 - (diff / max_diff)
                    else:
                        position_matches += 1.0

            if len(all_items) > 0:
                total_agreement += position_matches / len(all_items)
            pair_count += 1

    if pair_count == 0:
        return 1.0

    return total_agreement / pair_count


# Feature 45: Average response time calculation
def calculate_average_response_time(response_times: List[float]) -> float:
    """
    Calculate average response time from a list of times.

    Args:
        response_times: List of response times in seconds

    Returns:
        Average time in seconds
    """
    if not response_times:
        return 0.0
    return sum(response_times) / len(response_times)


# Feature 46: Model performance statistics
def calculate_model_statistics(
    model_rankings: Dict[str, List[int]]
) -> Dict[str, Dict[str, float]]:
    """
    Calculate performance statistics for each model.

    Args:
        model_rankings: Dict mapping model name to list of rank positions

    Returns:
        Dict with statistics for each model
    """
    stats = {}
    for model, rankings in model_rankings.items():
        if rankings:
            stats[model] = {
                "average_rank": sum(rankings) / len(rankings),
                "best_rank": min(rankings),
                "worst_rank": max(rankings),
                "total_evaluations": len(rankings),
                "first_place_count": rankings.count(1)
            }
        else:
            stats[model] = {
                "average_rank": 0,
                "best_rank": 0,
                "worst_rank": 0,
                "total_evaluations": 0,
                "first_place_count": 0
            }
    return stats


# Feature 47: Conversation duration calculation
def calculate_conversation_duration(
    start_time: str,
    end_time: str
) -> float:
    """
    Calculate the duration of a conversation in seconds.

    Args:
        start_time: ISO format start timestamp
        end_time: ISO format end timestamp

    Returns:
        Duration in seconds
    """
    from datetime import datetime

    try:
        start = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        end = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
        return (end - start).total_seconds()
    except (ValueError, AttributeError):
        return 0.0


# Feature 48: Response quality metrics
def calculate_quality_metrics(response: str) -> Dict[str, Any]:
    """
    Calculate various quality metrics for a response.

    Args:
        response: The response text

    Returns:
        Dict with quality metrics
    """
    if not response:
        return {
            "word_count": 0,
            "sentence_count": 0,
            "avg_sentence_length": 0,
            "has_code_blocks": False,
            "has_lists": False,
            "estimated_tokens": 0
        }

    words = response.split()
    word_count = len(words)

    # Count sentences (simple approximation)
    sentences = re.split(r'[.!?]+', response)
    sentence_count = len([s for s in sentences if s.strip()])

    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0

    # Check for code blocks
    has_code_blocks = '```' in response or '`' in response

    # Check for lists
    has_lists = bool(re.search(r'^\s*[-*•]\s', response, re.MULTILINE)) or \
                bool(re.search(r'^\s*\d+\.\s', response, re.MULTILINE))

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_length": round(avg_sentence_length, 2),
        "has_code_blocks": has_code_blocks,
        "has_lists": has_lists,
        "estimated_tokens": estimate_token_count(response)
    }


# Feature 49: Model diversity score
def calculate_model_diversity(responses: List[str]) -> float:
    """
    Calculate how diverse the model responses are.
    Higher score means more diverse responses.

    Args:
        responses: List of response texts

    Returns:
        Diversity score between 0 and 1
    """
    if not responses or len(responses) < 2:
        return 0.0

    # Calculate pairwise similarities
    similarities = []
    for i in range(len(responses)):
        for j in range(i + 1, len(responses)):
            sim = calculate_similarity(responses[i], responses[j])
            similarities.append(sim)

    if not similarities:
        return 0.0

    # Diversity is inverse of average similarity
    avg_similarity = sum(similarities) / len(similarities)
    return 1.0 - avg_similarity


# Feature 50: Council consensus calculation
def calculate_consensus_score(
    stage2_results: List[Dict[str, Any]],
    label_to_model: Dict[str, str]
) -> Dict[str, Any]:
    """
    Calculate the consensus level of the council.

    Args:
        stage2_results: Rankings from each model
        label_to_model: Mapping from labels to model names

    Returns:
        Dict with consensus metrics
    """
    if not stage2_results:
        return {
            "consensus_score": 0.0,
            "top_choice_agreement": 0.0,
            "most_agreed_model": None
        }

    # Extract first-place votes
    first_place_votes = []
    all_rankings = []

    for result in stage2_results:
        parsed = result.get('parsed_ranking', [])
        if parsed:
            first_place_votes.append(parsed[0])
            all_rankings.append(parsed)

    if not first_place_votes:
        return {
            "consensus_score": 0.0,
            "top_choice_agreement": 0.0,
            "most_agreed_model": None
        }

    # Calculate top choice agreement
    vote_counts = Counter(first_place_votes)
    most_common = vote_counts.most_common(1)[0]
    top_choice_agreement = most_common[1] / len(first_place_votes)

    # Get the most agreed upon model
    most_agreed_label = most_common[0]
    most_agreed_model = label_to_model.get(most_agreed_label, most_agreed_label)

    # Calculate overall consensus using ranking agreement
    consensus_score = calculate_ranking_agreement(all_rankings)

    return {
        "consensus_score": round(consensus_score, 3),
        "top_choice_agreement": round(top_choice_agreement, 3),
        "most_agreed_model": most_agreed_model
    }


# Feature 11: Weighted ranking calculation
def calculate_weighted_rankings(
    stage2_results: List[Dict[str, Any]],
    label_to_model: Dict[str, str],
    model_weights: Optional[Dict[str, float]] = None
) -> List[Dict[str, Any]]:
    """
    Calculate weighted aggregate rankings based on model credibility.

    Args:
        stage2_results: Rankings from each model
        label_to_model: Mapping from labels to model names
        model_weights: Optional dict of model weights (default: equal weights)

    Returns:
        List of models with weighted average ranks
    """
    from collections import defaultdict

    if model_weights is None:
        model_weights = {}

    # Track weighted positions for each model
    model_scores = defaultdict(lambda: {"total_weight": 0, "weighted_sum": 0})

    for ranking in stage2_results:
        ranker_model = ranking.get('model', '')
        weight = model_weights.get(ranker_model, 1.0)
        parsed = ranking.get('parsed_ranking', [])

        for position, label in enumerate(parsed, start=1):
            if label in label_to_model:
                model_name = label_to_model[label]
                model_scores[model_name]["weighted_sum"] += position * weight
                model_scores[model_name]["total_weight"] += weight

    # Calculate weighted averages
    results = []
    for model, scores in model_scores.items():
        if scores["total_weight"] > 0:
            weighted_avg = scores["weighted_sum"] / scores["total_weight"]
            results.append({
                "model": model,
                "weighted_average_rank": round(weighted_avg, 3),
                "total_weight": round(scores["total_weight"], 3)
            })

    results.sort(key=lambda x: x["weighted_average_rank"])
    return results


# Feature 12: Borda count voting
def calculate_borda_count(
    stage2_results: List[Dict[str, Any]],
    label_to_model: Dict[str, str]
) -> List[Dict[str, Any]]:
    """
    Calculate rankings using Borda count voting method.

    Args:
        stage2_results: Rankings from each model
        label_to_model: Mapping from labels to model names

    Returns:
        List of models with Borda scores (higher is better)
    """
    from collections import defaultdict

    # Determine number of candidates
    n_candidates = len(label_to_model)

    # Calculate Borda scores
    borda_scores = defaultdict(int)

    for ranking in stage2_results:
        parsed = ranking.get('parsed_ranking', [])
        for position, label in enumerate(parsed):
            if label in label_to_model:
                model_name = label_to_model[label]
                # Borda count: n-1 points for 1st, n-2 for 2nd, etc.
                points = n_candidates - position - 1
                borda_scores[model_name] += max(0, points)

    # Convert to sorted list
    results = [
        {"model": model, "borda_score": score}
        for model, score in borda_scores.items()
    ]
    results.sort(key=lambda x: x["borda_score"], reverse=True)

    return results


# Feature 13: Tie-breaking logic
def break_ties(
    rankings: List[Dict[str, Any]],
    stage1_results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Break ties in rankings using response length as tiebreaker.

    Args:
        rankings: List of ranked models with scores
        stage1_results: Original stage 1 responses

    Returns:
        Rankings with ties broken
    """
    # Create response length lookup
    response_lengths = {
        r['model']: len(r.get('response', ''))
        for r in stage1_results
    }

    # Sort with tiebreaker
    def sort_key(item):
        # Primary: score (lower is better for rank, higher for other scores)
        primary = item.get('average_rank') or item.get('weighted_average_rank') or \
                  -item.get('borda_score', 0)
        # Tiebreaker: longer responses ranked higher (negative for ascending sort)
        secondary = -response_lengths.get(item['model'], 0)
        return (primary, secondary)

    return sorted(rankings, key=sort_key)


# Feature 14: Confidence score calculation
def calculate_confidence_score(
    stage2_results: List[Dict[str, Any]],
    label_to_model: Dict[str, str]
) -> float:
    """
    Calculate confidence score for the council's final answer.
    Based on ranking agreement and number of successful responses.

    Args:
        stage2_results: Rankings from each model
        label_to_model: Mapping from labels to model names

    Returns:
        Confidence score between 0 and 1
    """
    if not stage2_results or not label_to_model:
        return 0.0

    # Factor 1: How many models participated
    participation_score = len(stage2_results) / max(len(label_to_model), 1)

    # Factor 2: Ranking agreement
    all_rankings = [r.get('parsed_ranking', []) for r in stage2_results]
    agreement_score = calculate_ranking_agreement(all_rankings)

    # Factor 3: Consistency of top choice
    consensus = calculate_consensus_score(stage2_results, label_to_model)
    top_agreement = consensus.get('top_choice_agreement', 0)

    # Weighted combination
    confidence = (
        participation_score * 0.3 +
        agreement_score * 0.4 +
        top_agreement * 0.3
    )

    return round(min(1.0, max(0.0, confidence)), 3)


# Feature 20: Response length validation
def validate_response_length(response: str, min_length: int = 10, max_length: int = 100000) -> bool:
    """
    Validate that a response is within acceptable length bounds.

    Args:
        response: The response text
        min_length: Minimum acceptable length
        max_length: Maximum acceptable length

    Returns:
        True if valid length
    """
    if not response:
        return min_length == 0
    length = len(response)
    return min_length <= length <= max_length
