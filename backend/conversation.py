"""Conversation management utilities for LLM Council."""

from typing import List, Dict, Any, Optional


def build_conversation_history(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Build a conversation history suitable for model context.

    Args:
        messages: List of conversation messages

    Returns:
        List of message dicts with role and content
    """
    history = []

    for msg in messages:
        if msg['role'] == 'user':
            history.append({
                'role': 'user',
                'content': msg['content']
            })
        elif msg['role'] == 'assistant':
            # Extract the final response from stage3
            stage3 = msg.get('stage3', {})
            response = stage3.get('response', '') if isinstance(stage3, dict) else ''
            if response:
                history.append({
                    'role': 'assistant',
                    'content': response
                })

    return history


def get_conversation_context(
    messages: List[Dict[str, Any]],
    max_messages: int = 10
) -> List[Dict[str, str]]:
    """
    Get recent conversation context for model queries.

    Args:
        messages: Full list of conversation messages
        max_messages: Maximum number of messages to include

    Returns:
        List of recent message dicts
    """
    history = build_conversation_history(messages)
    return history[-max_messages:] if len(history) > max_messages else history


def format_message_for_display(message: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format a message for display in the UI.

    Args:
        message: Raw message dict

    Returns:
        Formatted message dict
    """
    if message['role'] == 'user':
        return {
            'role': 'user',
            'content': message['content'],
            'display_type': 'text'
        }
    else:
        return {
            'role': 'assistant',
            'stage1': message.get('stage1', []),
            'stage2': message.get('stage2', []),
            'stage3': message.get('stage3', {}),
            'display_type': 'council'
        }


def summarize_conversation(messages: List[Dict[str, Any]], max_length: int = 500) -> str:
    """
    Generate a summary of a conversation.

    Args:
        messages: List of conversation messages
        max_length: Maximum summary length

    Returns:
        Summary string
    """
    if not messages:
        return "Empty conversation"

    parts = []
    for i, msg in enumerate(messages[:5]):  # First 5 messages
        if msg['role'] == 'user':
            content = msg.get('content', '')[:100]
            parts.append(f"User: {content}...")

    summary = "\n".join(parts)
    if len(summary) > max_length:
        summary = summary[:max_length] + "..."

    return summary
