"""Prompt management utilities for LLM Council."""

from typing import List, Optional, Dict

# Default system prompts
_system_prompts: Dict[str, str] = {
    "council": "You are a helpful AI assistant participating in a council deliberation.",
    "chairman": "You are the chairman of an AI council, responsible for synthesizing responses.",
    "ranking": "You are evaluating and ranking responses from different AI models."
}


def get_system_prompt(prompt_type: str) -> str:
    """
    Get a system prompt by type.

    Args:
        prompt_type: Type of prompt (council, chairman, ranking)

    Returns:
        The system prompt string
    """
    return _system_prompts.get(prompt_type, "")


def set_system_prompt(prompt_type: str, prompt: str):
    """
    Set a custom system prompt.

    Args:
        prompt_type: Type of prompt (council, chairman, ranking)
        prompt: The prompt string
    """
    _system_prompts[prompt_type] = prompt


def build_ranking_prompt(
    user_query: str,
    responses_text: str,
    criteria: Optional[List[str]] = None
) -> str:
    """
    Build a ranking prompt with optional custom criteria.

    Args:
        user_query: The original user query
        responses_text: Formatted text of all responses
        criteria: Optional list of ranking criteria

    Returns:
        The complete ranking prompt
    """
    if criteria is None:
        criteria = ["accuracy", "completeness", "clarity", "helpfulness"]

    criteria_text = "\n".join(f"- {c}" for c in criteria)

    return f"""You are evaluating different responses to the following question:

Question: {user_query}

Here are the responses from different models (anonymized):

{responses_text}

Please evaluate each response based on the following criteria:
{criteria_text}

Your task:
1. First, evaluate each response individually. For each response, explain what it does well and what it does poorly.
2. Then, at the very end of your response, provide a final ranking.

IMPORTANT: Your final ranking MUST be formatted EXACTLY as follows:
- Start with the line "FINAL RANKING:" (all caps, with colon)
- Then list the responses from best to worst as a numbered list
- Each line should be: number, period, space, then ONLY the response label (e.g., "1. Response A")
- Do not add any other text or explanations in the ranking section

Now provide your evaluation and ranking:"""


def build_chairman_prompt(
    user_query: str,
    stage1_text: str,
    stage2_text: str,
    include_reasoning: bool = True
) -> str:
    """
    Build the chairman synthesis prompt.

    Args:
        user_query: The original user query
        stage1_text: Formatted Stage 1 responses
        stage2_text: Formatted Stage 2 rankings
        include_reasoning: Whether to ask for reasoning explanation

    Returns:
        The complete chairman prompt
    """
    reasoning_instruction = ""
    if include_reasoning:
        reasoning_instruction = """
Before your final answer, briefly explain:
- Which responses were most highly rated
- Any key points of agreement or disagreement
- How you weighted different perspectives
"""

    return f"""You are the Chairman of an LLM Council. Multiple AI models have provided responses to a user's question, and then ranked each other's responses.

Original Question: {user_query}

STAGE 1 - Individual Responses:
{stage1_text}

STAGE 2 - Peer Rankings:
{stage2_text}

Your task as Chairman is to synthesize all of this information into a single, comprehensive, accurate answer to the user's original question. Consider:
- The individual responses and their insights
- The peer rankings and what they reveal about response quality
- Any patterns of agreement or disagreement
{reasoning_instruction}
Provide a clear, well-reasoned final answer that represents the council's collective wisdom:"""


def build_title_generation_prompt(user_query: str) -> str:
    """
    Build a prompt for generating conversation titles.

    Args:
        user_query: The first user message

    Returns:
        The title generation prompt
    """
    return f"""Generate a very short title (3-5 words maximum) that summarizes the following question.
The title should be concise and descriptive. Do not use quotes or punctuation in the title.

Question: {user_query}

Title:"""
