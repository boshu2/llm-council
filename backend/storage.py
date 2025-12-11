"""JSON-based storage for conversations."""

import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from .config import DATA_DIR


def ensure_data_dir():
    """Ensure the data directory exists."""
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)


def get_conversation_path(conversation_id: str) -> str:
    """Get the file path for a conversation."""
    return os.path.join(DATA_DIR, f"{conversation_id}.json")


def create_conversation(conversation_id: str) -> Dict[str, Any]:
    """
    Create a new conversation.

    Args:
        conversation_id: Unique identifier for the conversation

    Returns:
        New conversation dict
    """
    ensure_data_dir()

    conversation = {
        "id": conversation_id,
        "created_at": datetime.utcnow().isoformat(),
        "title": "New Conversation",
        "messages": []
    }

    # Save to file
    path = get_conversation_path(conversation_id)
    with open(path, 'w') as f:
        json.dump(conversation, f, indent=2)

    return conversation


def get_conversation(conversation_id: str) -> Optional[Dict[str, Any]]:
    """
    Load a conversation from storage.

    Args:
        conversation_id: Unique identifier for the conversation

    Returns:
        Conversation dict or None if not found
    """
    path = get_conversation_path(conversation_id)

    if not os.path.exists(path):
        return None

    with open(path, 'r') as f:
        return json.load(f)


def save_conversation(conversation: Dict[str, Any]):
    """
    Save a conversation to storage.

    Args:
        conversation: Conversation dict to save
    """
    ensure_data_dir()

    path = get_conversation_path(conversation['id'])
    with open(path, 'w') as f:
        json.dump(conversation, f, indent=2)


def list_conversations() -> List[Dict[str, Any]]:
    """
    List all conversations (metadata only).

    Returns:
        List of conversation metadata dicts
    """
    ensure_data_dir()

    conversations = []
    for filename in os.listdir(DATA_DIR):
        if filename.endswith('.json'):
            path = os.path.join(DATA_DIR, filename)
            with open(path, 'r') as f:
                data = json.load(f)
                # Return metadata only
                conversations.append({
                    "id": data["id"],
                    "created_at": data["created_at"],
                    "title": data.get("title", "New Conversation"),
                    "message_count": len(data["messages"])
                })

    # Sort by creation time, newest first
    conversations.sort(key=lambda x: x["created_at"], reverse=True)

    return conversations


def add_user_message(conversation_id: str, content: str):
    """
    Add a user message to a conversation.

    Args:
        conversation_id: Conversation identifier
        content: User message content
    """
    conversation = get_conversation(conversation_id)
    if conversation is None:
        raise ValueError(f"Conversation {conversation_id} not found")

    conversation["messages"].append({
        "role": "user",
        "content": content
    })

    save_conversation(conversation)


def add_assistant_message(
    conversation_id: str,
    stage1: List[Dict[str, Any]],
    stage2: List[Dict[str, Any]],
    stage3: Dict[str, Any]
):
    """
    Add an assistant message with all 3 stages to a conversation.

    Args:
        conversation_id: Conversation identifier
        stage1: List of individual model responses
        stage2: List of model rankings
        stage3: Final synthesized response
    """
    conversation = get_conversation(conversation_id)
    if conversation is None:
        raise ValueError(f"Conversation {conversation_id} not found")

    conversation["messages"].append({
        "role": "assistant",
        "stage1": stage1,
        "stage2": stage2,
        "stage3": stage3
    })

    save_conversation(conversation)


def update_conversation_title(conversation_id: str, title: str):
    """
    Update the title of a conversation.

    Args:
        conversation_id: Conversation identifier
        title: New title for the conversation
    """
    conversation = get_conversation(conversation_id)
    if conversation is None:
        raise ValueError(f"Conversation {conversation_id} not found")

    conversation["title"] = title
    save_conversation(conversation)


# Feature 1: Delete conversation
def delete_conversation(conversation_id: str) -> bool:
    """
    Delete a conversation.

    Args:
        conversation_id: Conversation identifier

    Returns:
        True if deleted, False if not found
    """
    path = get_conversation_path(conversation_id)

    if not os.path.exists(path):
        return False

    os.remove(path)
    return True


# Feature 21: Delete all conversations
def delete_all_conversations() -> int:
    """
    Delete all conversations.

    Returns:
        Number of conversations deleted
    """
    ensure_data_dir()

    count = 0
    for filename in os.listdir(DATA_DIR):
        if filename.endswith('.json'):
            path = os.path.join(DATA_DIR, filename)
            os.remove(path)
            count += 1

    return count


# Feature 22: Export conversation to JSON
def export_conversation_json(conversation_id: str) -> Optional[Dict[str, Any]]:
    """
    Export a conversation to JSON format.

    Args:
        conversation_id: Conversation identifier

    Returns:
        Conversation dict or None if not found
    """
    return get_conversation(conversation_id)


# Feature 23: Search conversations by title
def search_conversations(query: str) -> List[Dict[str, Any]]:
    """
    Search conversations by title.

    Args:
        query: Search query

    Returns:
        List of matching conversation metadata
    """
    all_convs = list_conversations()
    query_lower = query.lower()

    return [
        conv for conv in all_convs
        if query_lower in conv.get('title', '').lower()
    ]


# Feature 24: Pagination for conversations
def list_conversations_paginated(
    page: int = 1,
    page_size: int = 10
) -> Dict[str, Any]:
    """
    List conversations with pagination.

    Args:
        page: Page number (1-indexed)
        page_size: Number of items per page

    Returns:
        Dict with items, total, page, and pages
    """
    all_convs = list_conversations()
    total = len(all_convs)
    pages = (total + page_size - 1) // page_size if total > 0 else 1

    start = (page - 1) * page_size
    end = start + page_size

    return {
        "items": all_convs[start:end],
        "total": total,
        "page": page,
        "pages": pages,
        "page_size": page_size
    }


# Feature 25: Archive conversation
def archive_conversation(conversation_id: str) -> bool:
    """
    Archive a conversation.

    Args:
        conversation_id: Conversation identifier

    Returns:
        True if archived successfully
    """
    conversation = get_conversation(conversation_id)
    if conversation is None:
        return False

    conversation['archived'] = True
    conversation['archived_at'] = datetime.utcnow().isoformat()
    save_conversation(conversation)
    return True


def unarchive_conversation(conversation_id: str) -> bool:
    """
    Unarchive a conversation.

    Args:
        conversation_id: Conversation identifier

    Returns:
        True if unarchived successfully
    """
    conversation = get_conversation(conversation_id)
    if conversation is None:
        return False

    conversation['archived'] = False
    conversation.pop('archived_at', None)
    save_conversation(conversation)
    return True


# Feature 26: Message timestamps (updated add_user_message)
def add_user_message(conversation_id: str, content: str, add_timestamp: bool = False):
    """
    Add a user message to a conversation.

    Args:
        conversation_id: Conversation identifier
        content: User message content
        add_timestamp: Whether to add a timestamp
    """
    conversation = get_conversation(conversation_id)
    if conversation is None:
        raise ValueError(f"Conversation {conversation_id} not found")

    message = {
        "role": "user",
        "content": content
    }

    if add_timestamp:
        message["timestamp"] = datetime.utcnow().isoformat()

    conversation["messages"].append(message)
    save_conversation(conversation)


# Feature 27: Conversation tags
def add_tags(conversation_id: str, tags: List[str]) -> bool:
    """
    Add tags to a conversation.

    Args:
        conversation_id: Conversation identifier
        tags: List of tags to add

    Returns:
        True if successful
    """
    conversation = get_conversation(conversation_id)
    if conversation is None:
        return False

    existing_tags = set(conversation.get('tags', []))
    existing_tags.update(tags)
    conversation['tags'] = list(existing_tags)
    save_conversation(conversation)
    return True


def remove_tags(conversation_id: str, tags: List[str]) -> bool:
    """
    Remove tags from a conversation.

    Args:
        conversation_id: Conversation identifier
        tags: List of tags to remove

    Returns:
        True if successful
    """
    conversation = get_conversation(conversation_id)
    if conversation is None:
        return False

    existing_tags = set(conversation.get('tags', []))
    existing_tags -= set(tags)
    conversation['tags'] = list(existing_tags)
    save_conversation(conversation)
    return True


def get_conversations_by_tag(tag: str) -> List[Dict[str, Any]]:
    """
    Get conversations with a specific tag.

    Args:
        tag: Tag to filter by

    Returns:
        List of conversation metadata
    """
    all_convs = list_conversations()
    result = []

    for conv_meta in all_convs:
        full_conv = get_conversation(conv_meta['id'])
        if full_conv and tag in full_conv.get('tags', []):
            result.append(conv_meta)

    return result


# Feature 28: Export to markdown
def export_conversation_markdown(conversation_id: str) -> Optional[str]:
    """
    Export a conversation to markdown format.

    Args:
        conversation_id: Conversation identifier

    Returns:
        Markdown string or None if not found
    """
    conversation = get_conversation(conversation_id)
    if conversation is None:
        return None

    lines = [
        f"# {conversation.get('title', 'Conversation')}",
        f"",
        f"*Created: {conversation.get('created_at', 'Unknown')}*",
        f"",
        "---",
        ""
    ]

    for msg in conversation.get('messages', []):
        if msg['role'] == 'user':
            lines.append(f"## User")
            lines.append("")
            lines.append(msg.get('content', ''))
            lines.append("")
        else:
            lines.append(f"## Council Response")
            lines.append("")

            # Stage 3 final response
            stage3 = msg.get('stage3', {})
            if isinstance(stage3, dict):
                lines.append(f"**Final Answer ({stage3.get('model', 'Unknown')}):**")
                lines.append("")
                lines.append(stage3.get('response', ''))
                lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


# Feature 29: Import from JSON
def import_conversation_json(data: Dict[str, Any]) -> Optional[str]:
    """
    Import a conversation from JSON data.

    Args:
        data: Conversation data dict

    Returns:
        Conversation ID if successful, None if failed
    """
    try:
        # Validate required fields
        if 'id' not in data:
            return None

        conversation_id = data['id']

        # Set defaults
        conversation = {
            'id': conversation_id,
            'created_at': data.get('created_at', datetime.utcnow().isoformat()),
            'title': data.get('title', 'Imported Conversation'),
            'messages': data.get('messages', []),
            'tags': data.get('tags', []),
            'archived': data.get('archived', False)
        }

        ensure_data_dir()
        path = get_conversation_path(conversation_id)

        with open(path, 'w') as f:
            json.dump(conversation, f, indent=2)

        return conversation_id
    except Exception:
        return None


# Feature 30: Conversation statistics
def get_conversation_statistics(conversation_id: str) -> Optional[Dict[str, Any]]:
    """
    Get statistics for a conversation.

    Args:
        conversation_id: Conversation identifier

    Returns:
        Statistics dict or None if not found
    """
    conversation = get_conversation(conversation_id)
    if conversation is None:
        return None

    messages = conversation.get('messages', [])
    user_messages = [m for m in messages if m.get('role') == 'user']
    assistant_messages = [m for m in messages if m.get('role') == 'assistant']

    # Calculate total response lengths
    total_response_length = 0
    for msg in assistant_messages:
        stage3 = msg.get('stage3', {})
        if isinstance(stage3, dict):
            total_response_length += len(stage3.get('response', ''))

    return {
        'message_count': len(messages),
        'user_message_count': len(user_messages),
        'assistant_message_count': len(assistant_messages),
        'total_response_length': total_response_length,
        'created_at': conversation.get('created_at'),
        'title': conversation.get('title'),
        'archived': conversation.get('archived', False),
        'tags': conversation.get('tags', [])
    }
