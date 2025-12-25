"""Rate limiting utilities for LLM Council."""

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List
import time


class RateLimiter:
    """
    Simple rate limiter using sliding window.
    """

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        """
        Initialize rate limiter.

        Args:
            max_requests: Maximum requests allowed in the window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[float]] = defaultdict(list)

    def is_allowed(self, client_id: str) -> bool:
        """
        Check if a request from a client is allowed.

        Args:
            client_id: Unique identifier for the client

        Returns:
            True if request is allowed, False if rate limited
        """
        now = time.time()
        window_start = now - self.window_seconds

        # Clean up old requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if req_time > window_start
        ]

        # Check if under limit
        if len(self.requests[client_id]) >= self.max_requests:
            return False

        # Record this request
        self.requests[client_id].append(now)
        return True

    def get_remaining(self, client_id: str) -> int:
        """
        Get remaining requests for a client.

        Args:
            client_id: Unique identifier for the client

        Returns:
            Number of remaining requests in current window
        """
        now = time.time()
        window_start = now - self.window_seconds

        valid_requests = [
            req_time for req_time in self.requests[client_id]
            if req_time > window_start
        ]

        return max(0, self.max_requests - len(valid_requests))

    def reset(self, client_id: str = None):
        """
        Reset rate limit for a client or all clients.

        Args:
            client_id: Optional client ID to reset (None resets all)
        """
        if client_id:
            self.requests[client_id] = []
        else:
            self.requests.clear()


class ConversationRateLimiter:
    """
    Rate limiter specific to conversations.
    """

    def __init__(self, max_messages: int = 10, window_seconds: int = 60):
        """
        Initialize conversation rate limiter.

        Args:
            max_messages: Maximum messages allowed per conversation in window
            window_seconds: Time window in seconds
        """
        self.max_messages = max_messages
        self.window_seconds = window_seconds
        self.messages: Dict[str, List[float]] = defaultdict(list)

    def is_allowed(self, conversation_id: str) -> bool:
        """
        Check if a message in a conversation is allowed.

        Args:
            conversation_id: Unique identifier for the conversation

        Returns:
            True if message is allowed, False if rate limited
        """
        now = time.time()
        window_start = now - self.window_seconds

        # Clean up old messages
        self.messages[conversation_id] = [
            msg_time for msg_time in self.messages[conversation_id]
            if msg_time > window_start
        ]

        # Check if under limit
        if len(self.messages[conversation_id]) >= self.max_messages:
            return False

        # Record this message
        self.messages[conversation_id].append(now)
        return True

    def get_remaining(self, conversation_id: str) -> int:
        """
        Get remaining messages for a conversation.

        Args:
            conversation_id: Unique identifier for the conversation

        Returns:
            Number of remaining messages in current window
        """
        now = time.time()
        window_start = now - self.window_seconds

        valid_messages = [
            msg_time for msg_time in self.messages[conversation_id]
            if msg_time > window_start
        ]

        return max(0, self.max_messages - len(valid_messages))
