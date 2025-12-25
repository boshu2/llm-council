"""Pytest configuration and fixtures for LLM Council tests."""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def sample_stage1_results():
    """Sample Stage 1 results for testing."""
    return [
        {'model': 'openai/gpt-4', 'response': 'This is a detailed answer about Python.'},
        {'model': 'anthropic/claude-3', 'response': 'Python is a programming language.'},
        {'model': 'google/gemini-pro', 'response': 'Here is information about Python programming.'},
    ]


@pytest.fixture
def sample_stage2_results():
    """Sample Stage 2 results for testing."""
    return [
        {
            'model': 'openai/gpt-4',
            'ranking': 'FINAL RANKING:\n1. Response A\n2. Response B\n3. Response C',
            'parsed_ranking': ['Response A', 'Response B', 'Response C']
        },
        {
            'model': 'anthropic/claude-3',
            'ranking': 'FINAL RANKING:\n1. Response A\n2. Response C\n3. Response B',
            'parsed_ranking': ['Response A', 'Response C', 'Response B']
        },
    ]


@pytest.fixture
def sample_label_to_model():
    """Sample label to model mapping."""
    return {
        'Response A': 'openai/gpt-4',
        'Response B': 'anthropic/claude-3',
        'Response C': 'google/gemini-pro'
    }


@pytest.fixture
def sample_conversation():
    """Sample conversation for testing."""
    return {
        'id': 'test-conv-123',
        'created_at': '2024-01-01T12:00:00',
        'title': 'Test Conversation',
        'messages': [
            {'role': 'user', 'content': 'What is Python?'},
            {
                'role': 'assistant',
                'stage1': [{'model': 'gpt-4', 'response': 'Python is...'}],
                'stage2': [],
                'stage3': {'model': 'gemini', 'response': 'Final answer...'}
            }
        ],
        'tags': ['python', 'programming'],
        'archived': False
    }
