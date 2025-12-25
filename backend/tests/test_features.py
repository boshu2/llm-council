"""
Comprehensive tests for 50 features of LLM Council.

Features:
1-10: Core Backend Features
11-20: Council Logic Features
21-30: Storage Features
31-40: Validation & Security Features
41-50: Utility Features
"""

import pytest
import json
import os
import sys
import uuid
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend import utils
from backend import storage
from backend import config


# ============================================================================
# Features 1-10: Core Backend Features
# ============================================================================

class TestFeature1DeleteConversation:
    """Feature 1: Delete conversation endpoint"""

    def test_delete_existing_conversation(self, tmp_path):
        """Should delete an existing conversation"""
        # Setup
        with patch.object(config, 'DATA_DIR', str(tmp_path)):
            storage.DATA_DIR = str(tmp_path)
            conv_id = str(uuid.uuid4())
            storage.create_conversation(conv_id)
            assert storage.get_conversation(conv_id) is not None

            # Test deletion
            result = storage.delete_conversation(conv_id)
            assert result is True
            assert storage.get_conversation(conv_id) is None

    def test_delete_nonexistent_conversation(self, tmp_path):
        """Should return False for non-existent conversation"""
        with patch.object(config, 'DATA_DIR', str(tmp_path)):
            storage.DATA_DIR = str(tmp_path)
            result = storage.delete_conversation("nonexistent-id")
            assert result is False


class TestFeature2ConfigurableTimeout:
    """Feature 2: Configurable timeout for model queries"""

    def test_default_timeout(self):
        """Should have default timeout configured"""
        from backend.llm_client import query_model
        from backend.config import DEFAULT_TIMEOUT
        # Check function signature has timeout parameter
        import inspect
        sig = inspect.signature(query_model)
        timeout_param = sig.parameters.get('timeout')
        assert timeout_param is not None
        # Default is None (uses config value)
        assert DEFAULT_TIMEOUT == 120.0

    def test_custom_timeout_parameter(self):
        """Should accept custom timeout parameter"""
        from backend.llm_client import query_model
        import inspect
        sig = inspect.signature(query_model)
        assert 'timeout' in sig.parameters


class TestFeature3HealthCheck:
    """Feature 3: Model health check endpoint"""

    def test_health_check_structure(self):
        """Health check should return proper structure"""
        from backend.health import check_api_health
        result = check_api_health()
        assert 'status' in result
        assert 'litellm_configured' in result
        assert 'timestamp' in result


class TestFeature4AvailableModels:
    """Feature 4: Get available models endpoint"""

    def test_get_council_models(self):
        """Should return list of council models"""
        from backend.config import COUNCIL_MODELS, CHAIRMAN_MODEL
        assert isinstance(COUNCIL_MODELS, list)
        assert len(COUNCIL_MODELS) > 0
        assert CHAIRMAN_MODEL is not None

    def test_get_available_models(self):
        """Should return available models info"""
        from backend.health import get_available_models
        result = get_available_models()
        assert 'council_models' in result
        assert 'chairman_model' in result


class TestFeature5RetryLogic:
    """Feature 5: Retry logic for failed model queries"""

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """Should retry on transient failures"""
        from backend.llm_client import query_model_with_retry

        call_count = 0
        async def mock_query(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return None
            return {'content': 'success'}

        with patch('backend.llm_client.query_model', mock_query):
            result = await query_model_with_retry('test/model', [], max_retries=3)
            assert result is not None
            assert call_count == 3


class TestFeature6RateLimiting:
    """Feature 6: Rate limiting for API endpoints"""

    def test_rate_limiter_allows_requests(self):
        """Should allow requests within limit"""
        from backend.rate_limiter import RateLimiter
        limiter = RateLimiter(max_requests=10, window_seconds=60)
        assert limiter.is_allowed("test_client") is True

    def test_rate_limiter_blocks_excess(self):
        """Should block requests exceeding limit"""
        from backend.rate_limiter import RateLimiter
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        limiter.is_allowed("test_client")
        limiter.is_allowed("test_client")
        assert limiter.is_allowed("test_client") is False


class TestFeature7RequestLogging:
    """Feature 7: Request logging middleware"""

    def test_log_request_structure(self):
        """Should log requests with proper structure"""
        from backend.logging_utils import format_request_log
        log = format_request_log(
            method="POST",
            path="/api/test",
            status_code=200,
            duration_ms=150.5
        )
        assert "POST" in log
        assert "/api/test" in log
        assert "200" in log


class TestFeature8ResponseTimeTracking:
    """Feature 8: Model response time tracking"""

    def test_track_response_time(self):
        """Should track model response times"""
        from backend.metrics import ResponseTimeTracker
        tracker = ResponseTimeTracker()
        tracker.record("openai/gpt-4", 1.5)
        tracker.record("openai/gpt-4", 2.0)

        stats = tracker.get_stats("openai/gpt-4")
        assert stats['count'] == 2
        assert stats['average'] == 1.75


class TestFeature9CustomSystemPrompts:
    """Feature 9: Custom system prompts for council models"""

    def test_system_prompt_config(self):
        """Should support custom system prompts"""
        from backend.prompts import get_system_prompt, set_system_prompt

        custom_prompt = "You are a helpful council member."
        set_system_prompt("council", custom_prompt)
        assert get_system_prompt("council") == custom_prompt


class TestFeature10TemperatureConfig:
    """Feature 10: Temperature parameter configuration"""

    def test_temperature_in_config(self):
        """Should support temperature configuration"""
        from backend.model_config import ModelConfig

        config = ModelConfig(temperature=0.7, max_tokens=4096)
        assert config.temperature == 0.7
        assert config.max_tokens == 4096


# ============================================================================
# Features 11-20: Council Logic Features
# ============================================================================

class TestFeature11WeightedRankings:
    """Feature 11: Weighted ranking calculation"""

    def test_weighted_rankings_equal_weights(self):
        """Should calculate correctly with equal weights"""
        stage2_results = [
            {'model': 'model_a', 'parsed_ranking': ['Response A', 'Response B']},
            {'model': 'model_b', 'parsed_ranking': ['Response B', 'Response A']},
        ]
        label_to_model = {'Response A': 'openai/gpt-4', 'Response B': 'anthropic/claude'}

        result = utils.calculate_weighted_rankings(stage2_results, label_to_model)
        assert len(result) == 2
        # Both should have average rank of 1.5
        for r in result:
            assert r['weighted_average_rank'] == 1.5

    def test_weighted_rankings_with_weights(self):
        """Should apply model weights correctly"""
        stage2_results = [
            {'model': 'model_a', 'parsed_ranking': ['Response A', 'Response B']},
            {'model': 'model_b', 'parsed_ranking': ['Response B', 'Response A']},
        ]
        label_to_model = {'Response A': 'openai/gpt-4', 'Response B': 'anthropic/claude'}
        model_weights = {'model_a': 2.0, 'model_b': 1.0}

        result = utils.calculate_weighted_rankings(stage2_results, label_to_model, model_weights)
        # model_a's rankings should count more
        gpt4_rank = next(r for r in result if r['model'] == 'openai/gpt-4')
        assert gpt4_rank['weighted_average_rank'] < 1.5  # Should be better due to weight


class TestFeature12BordaCount:
    """Feature 12: Borda count voting"""

    def test_borda_count_calculation(self):
        """Should calculate Borda count correctly"""
        stage2_results = [
            {'model': 'model_a', 'parsed_ranking': ['Response A', 'Response B', 'Response C']},
            {'model': 'model_b', 'parsed_ranking': ['Response A', 'Response C', 'Response B']},
        ]
        label_to_model = {
            'Response A': 'gpt-4',
            'Response B': 'claude',
            'Response C': 'gemini'
        }

        result = utils.calculate_borda_count(stage2_results, label_to_model)
        # Response A: 2+2=4, Response B: 1+0=1, Response C: 0+1=1
        assert result[0]['model'] == 'gpt-4'
        assert result[0]['borda_score'] == 4


class TestFeature13TieBreaking:
    """Feature 13: Tie-breaking logic"""

    def test_tie_breaking_by_response_length(self):
        """Should break ties using response length"""
        rankings = [
            {'model': 'gpt-4', 'average_rank': 1.5},
            {'model': 'claude', 'average_rank': 1.5},
        ]
        stage1_results = [
            {'model': 'gpt-4', 'response': 'Short'},
            {'model': 'claude', 'response': 'This is a much longer response'},
        ]

        result = utils.break_ties(rankings, stage1_results)
        # Longer response should win
        assert result[0]['model'] == 'claude'


class TestFeature14ConfidenceScore:
    """Feature 14: Confidence score calculation"""

    def test_high_confidence_unanimous(self):
        """Should have high confidence for unanimous rankings"""
        stage2_results = [
            {'model': 'a', 'parsed_ranking': ['Response A', 'Response B']},
            {'model': 'b', 'parsed_ranking': ['Response A', 'Response B']},
            {'model': 'c', 'parsed_ranking': ['Response A', 'Response B']},
        ]
        label_to_model = {'Response A': 'gpt-4', 'Response B': 'claude'}

        confidence = utils.calculate_confidence_score(stage2_results, label_to_model)
        assert confidence > 0.7

    def test_low_confidence_disagreement(self):
        """Should have lower confidence for disagreement"""
        stage2_results = [
            {'model': 'a', 'parsed_ranking': ['Response A', 'Response B']},
            {'model': 'b', 'parsed_ranking': ['Response B', 'Response A']},
        ]
        label_to_model = {'Response A': 'gpt-4', 'Response B': 'claude'}

        confidence = utils.calculate_confidence_score(stage2_results, label_to_model)
        assert confidence < 0.8


class TestFeature15MultiTurnConversation:
    """Feature 15: Multi-turn conversation support"""

    def test_conversation_history_building(self):
        """Should build conversation history correctly"""
        from backend.conversation import build_conversation_history

        messages = [
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'stage3': {'response': 'Hi there!'}},
            {'role': 'user', 'content': 'How are you?'},
        ]

        history = build_conversation_history(messages)
        assert len(history) == 3
        assert history[0]['role'] == 'user'
        assert history[1]['role'] == 'assistant'


class TestFeature16SkipFailedModels:
    """Feature 16: Skip failed models in Stage 2"""

    def test_filter_failed_models(self):
        """Should filter out models that failed in Stage 1"""
        from backend.council_utils import filter_successful_models

        stage1_results = [
            {'model': 'gpt-4', 'response': 'answer'},
            {'model': 'claude', 'response': 'answer'},
        ]
        all_models = ['gpt-4', 'claude', 'gemini']  # gemini failed

        filtered = filter_successful_models(all_models, stage1_results)
        assert 'gemini' not in filtered
        assert len(filtered) == 2


class TestFeature17CustomRankingCriteria:
    """Feature 17: Custom ranking criteria configuration"""

    def test_ranking_criteria_config(self):
        """Should support custom ranking criteria"""
        from backend.prompts import build_ranking_prompt

        criteria = ["accuracy", "clarity", "completeness"]
        prompt = build_ranking_prompt("test query", "responses", criteria)
        assert "accuracy" in prompt
        assert "clarity" in prompt


class TestFeature18MinorityOpinion:
    """Feature 18: Minority opinion tracking"""

    def test_identify_minority_opinions(self):
        """Should identify minority opinions"""
        from backend.council_utils import identify_minority_opinions

        stage2_results = [
            {'model': 'a', 'parsed_ranking': ['Response A', 'Response B']},
            {'model': 'b', 'parsed_ranking': ['Response A', 'Response B']},
            {'model': 'c', 'parsed_ranking': ['Response B', 'Response A']},  # minority
        ]

        minority = identify_minority_opinions(stage2_results)
        assert 'c' in minority['dissenting_models']


class TestFeature19ReasoningExtraction:
    """Feature 19: Chain-of-thought reasoning extraction"""

    def test_extract_reasoning(self):
        """Should extract reasoning from response"""
        from backend.council_utils import extract_reasoning

        response = {
            'content': 'Final answer',
            'reasoning_details': 'Step 1... Step 2... Step 3...'
        }

        reasoning = extract_reasoning(response)
        assert reasoning == 'Step 1... Step 2... Step 3...'


class TestFeature20ResponseLengthValidation:
    """Feature 20: Response length validation"""

    def test_valid_response_length(self):
        """Should accept valid length responses"""
        response = "This is a valid response with enough content."
        assert utils.validate_response_length(response) is True

    def test_too_short_response(self):
        """Should reject too short responses"""
        assert utils.validate_response_length("Hi", min_length=10) is False

    def test_too_long_response(self):
        """Should reject too long responses"""
        long_response = "x" * 200000
        assert utils.validate_response_length(long_response, max_length=100000) is False


# ============================================================================
# Features 21-30: Storage Features
# ============================================================================

class TestFeature21DeleteAllConversations:
    """Feature 21: Delete all conversations endpoint"""

    def test_delete_all_conversations(self, tmp_path):
        """Should delete all conversations"""
        with patch.object(config, 'DATA_DIR', str(tmp_path)):
            storage.DATA_DIR = str(tmp_path)
            # Create multiple conversations
            for _ in range(3):
                storage.create_conversation(str(uuid.uuid4()))

            count = storage.delete_all_conversations()
            assert count == 3
            assert len(storage.list_conversations()) == 0


class TestFeature22ExportJSON:
    """Feature 22: Export conversation to JSON"""

    def test_export_conversation_json(self, tmp_path):
        """Should export conversation to JSON"""
        with patch.object(config, 'DATA_DIR', str(tmp_path)):
            storage.DATA_DIR = str(tmp_path)
            conv_id = str(uuid.uuid4())
            storage.create_conversation(conv_id)

            exported = storage.export_conversation_json(conv_id)
            assert 'id' in exported
            assert 'messages' in exported


class TestFeature23SearchByTitle:
    """Feature 23: Conversation search by title"""

    def test_search_conversations_by_title(self, tmp_path):
        """Should search conversations by title"""
        with patch.object(config, 'DATA_DIR', str(tmp_path)):
            storage.DATA_DIR = str(tmp_path)
            conv_id = str(uuid.uuid4())
            storage.create_conversation(conv_id)
            storage.update_conversation_title(conv_id, "Python Programming Help")

            results = storage.search_conversations("Python")
            assert len(results) == 1
            assert results[0]['id'] == conv_id


class TestFeature24Pagination:
    """Feature 24: Pagination for conversations list"""

    def test_paginated_conversations(self, tmp_path):
        """Should paginate conversation list"""
        with patch.object(config, 'DATA_DIR', str(tmp_path)):
            storage.DATA_DIR = str(tmp_path)
            for _ in range(10):
                storage.create_conversation(str(uuid.uuid4()))

            page1 = storage.list_conversations_paginated(page=1, page_size=3)
            assert len(page1['items']) == 3
            assert page1['total'] == 10
            assert page1['pages'] == 4


class TestFeature25Archiving:
    """Feature 25: Conversation archiving"""

    def test_archive_conversation(self, tmp_path):
        """Should archive a conversation"""
        with patch.object(config, 'DATA_DIR', str(tmp_path)):
            storage.DATA_DIR = str(tmp_path)
            conv_id = str(uuid.uuid4())
            storage.create_conversation(conv_id)

            storage.archive_conversation(conv_id)
            conv = storage.get_conversation(conv_id)
            assert conv['archived'] is True


class TestFeature26MessageTimestamps:
    """Feature 26: Message timestamps"""

    def test_message_has_timestamp(self, tmp_path):
        """Should add timestamp to messages"""
        with patch.object(config, 'DATA_DIR', str(tmp_path)):
            storage.DATA_DIR = str(tmp_path)
            conv_id = str(uuid.uuid4())
            storage.create_conversation(conv_id)
            storage.add_user_message(conv_id, "Hello", add_timestamp=True)

            conv = storage.get_conversation(conv_id)
            assert 'timestamp' in conv['messages'][0]


class TestFeature27ConversationTags:
    """Feature 27: Conversation tags/categories"""

    def test_add_tags_to_conversation(self, tmp_path):
        """Should add tags to conversation"""
        with patch.object(config, 'DATA_DIR', str(tmp_path)):
            storage.DATA_DIR = str(tmp_path)
            conv_id = str(uuid.uuid4())
            storage.create_conversation(conv_id)

            storage.add_tags(conv_id, ["python", "help"])
            conv = storage.get_conversation(conv_id)
            assert "python" in conv['tags']


class TestFeature28ExportMarkdown:
    """Feature 28: Export to markdown format"""

    def test_export_conversation_markdown(self, tmp_path):
        """Should export conversation to markdown"""
        with patch.object(config, 'DATA_DIR', str(tmp_path)):
            storage.DATA_DIR = str(tmp_path)
            conv_id = str(uuid.uuid4())
            storage.create_conversation(conv_id)
            storage.add_user_message(conv_id, "What is Python?")

            markdown = storage.export_conversation_markdown(conv_id)
            assert "# " in markdown  # Has header
            assert "What is Python?" in markdown


class TestFeature29ImportJSON:
    """Feature 29: Import conversation from JSON"""

    def test_import_conversation_json(self, tmp_path):
        """Should import conversation from JSON"""
        with patch.object(config, 'DATA_DIR', str(tmp_path)):
            storage.DATA_DIR = str(tmp_path)

            json_data = {
                'id': str(uuid.uuid4()),
                'created_at': datetime.utcnow().isoformat(),
                'title': 'Imported Conversation',
                'messages': []
            }

            conv_id = storage.import_conversation_json(json_data)
            assert conv_id is not None
            conv = storage.get_conversation(conv_id)
            assert conv['title'] == 'Imported Conversation'


class TestFeature30ConversationStats:
    """Feature 30: Conversation statistics endpoint"""

    def test_conversation_statistics(self, tmp_path):
        """Should return conversation statistics"""
        with patch.object(config, 'DATA_DIR', str(tmp_path)):
            storage.DATA_DIR = str(tmp_path)
            conv_id = str(uuid.uuid4())
            storage.create_conversation(conv_id)
            storage.add_user_message(conv_id, "Hello")

            stats = storage.get_conversation_statistics(conv_id)
            assert 'message_count' in stats
            assert 'user_message_count' in stats


# ============================================================================
# Features 31-40: Validation & Security Features
# ============================================================================

class TestFeature31InputSanitization:
    """Feature 31: Input sanitization"""

    def test_removes_null_bytes(self):
        """Should remove null bytes from input"""
        result = utils.sanitize_input("Hello\x00World")
        assert "\x00" not in result
        assert "HelloWorld" in result

    def test_preserves_normal_text(self):
        """Should preserve normal text"""
        text = "Hello, World! How are you?"
        assert utils.sanitize_input(text) == text

    def test_handles_empty_input(self):
        """Should handle empty input"""
        assert utils.sanitize_input("") == ""
        assert utils.sanitize_input(None) == ""


class TestFeature32MessageLengthValidation:
    """Feature 32: Maximum message length validation"""

    def test_valid_length(self):
        """Should accept valid length messages"""
        assert utils.validate_message_length("Hello") is True

    def test_exceeds_max_length(self):
        """Should reject messages exceeding max length"""
        long_message = "x" * 60000
        assert utils.validate_message_length(long_message, max_length=50000) is False


class TestFeature33ContentFiltering:
    """Feature 33: Content filtering"""

    def test_appropriate_content(self):
        """Should accept appropriate content"""
        assert utils.check_content_appropriate("How do I write Python code?") is True

    def test_empty_content(self):
        """Should handle empty content"""
        assert utils.check_content_appropriate("") is True


class TestFeature34APIKeyValidation:
    """Feature 34: API key validation"""

    def test_litellm_configured(self):
        """Should check if LiteLLM is configured"""
        from backend.health import is_litellm_configured
        # Just verify the function exists and returns bool
        result = is_litellm_configured()
        assert isinstance(result, bool)


class TestFeature35RequestBodySize:
    """Feature 35: Request body size limit"""

    def test_body_size_validation(self):
        """Should validate request body size"""
        from backend.validation import validate_body_size

        small_body = {"content": "Hello"}
        large_body = {"content": "x" * 2000000}

        assert validate_body_size(small_body) is True
        assert validate_body_size(large_body, max_size=1000000) is False


class TestFeature36EmptyMessageValidation:
    """Feature 36: Empty message validation"""

    def test_non_empty_valid(self):
        """Should accept non-empty messages"""
        assert utils.validate_not_empty("Hello") is True

    def test_empty_invalid(self):
        """Should reject empty messages"""
        assert utils.validate_not_empty("") is False
        assert utils.validate_not_empty("   ") is False
        assert utils.validate_not_empty(None) is False


class TestFeature37ConversationIDValidation:
    """Feature 37: Conversation ID format validation"""

    def test_valid_uuid(self):
        """Should accept valid UUIDs"""
        valid_id = str(uuid.uuid4())
        assert utils.validate_conversation_id(valid_id) is True

    def test_invalid_format(self):
        """Should reject invalid formats"""
        assert utils.validate_conversation_id("invalid") is False
        assert utils.validate_conversation_id("123") is False


class TestFeature38RateLimitPerConversation:
    """Feature 38: Rate limit per conversation"""

    def test_conversation_rate_limit(self):
        """Should rate limit per conversation"""
        from backend.rate_limiter import ConversationRateLimiter

        limiter = ConversationRateLimiter(max_messages=5, window_seconds=60)
        conv_id = str(uuid.uuid4())

        for _ in range(5):
            assert limiter.is_allowed(conv_id) is True
        assert limiter.is_allowed(conv_id) is False


class TestFeature39ModelNameValidation:
    """Feature 39: Model name validation"""

    def test_valid_model_names(self):
        """Should accept valid model name formats"""
        assert utils.validate_model_name("openai/gpt-4") is True
        assert utils.validate_model_name("anthropic/claude-3") is True
        assert utils.validate_model_name("google/gemini-pro") is True

    def test_invalid_model_names(self):
        """Should reject invalid model name formats"""
        assert utils.validate_model_name("invalid") is False
        assert utils.validate_model_name("") is False
        assert utils.validate_model_name(None) is False


class TestFeature40DuplicateMessageDetection:
    """Feature 40: Duplicate message detection"""

    def test_detect_duplicate(self):
        """Should detect duplicate messages"""
        message = "What is Python?"
        hash1 = utils.compute_message_hash(message)

        assert utils.is_duplicate_message(message, [hash1]) is True

    def test_non_duplicate(self):
        """Should not flag non-duplicates"""
        message = "What is JavaScript?"
        hash1 = utils.compute_message_hash("What is Python?")

        assert utils.is_duplicate_message(message, [hash1]) is False


# ============================================================================
# Features 41-50: Utility Features
# ============================================================================

class TestFeature41TokenEstimation:
    """Feature 41: Token count estimation"""

    def test_estimate_tokens(self):
        """Should estimate token count"""
        text = "This is a test message with several words."
        tokens = utils.estimate_token_count(text)
        assert tokens > 0
        assert tokens < len(text)  # Tokens should be fewer than characters

    def test_empty_text_tokens(self):
        """Should handle empty text"""
        assert utils.estimate_token_count("") == 0


class TestFeature42WordCount:
    """Feature 42: Response word count"""

    def test_count_words(self):
        """Should count words correctly"""
        text = "This is a test message"
        assert utils.count_words(text) == 5

    def test_empty_word_count(self):
        """Should handle empty text"""
        assert utils.count_words("") == 0


class TestFeature43SimilarityDetection:
    """Feature 43: Response similarity detection"""

    def test_identical_texts(self):
        """Should return 1.0 for identical texts"""
        text = "This is a test"
        assert utils.calculate_similarity(text, text) == 1.0

    def test_completely_different(self):
        """Should return low similarity for different texts"""
        text1 = "cats dogs birds"
        text2 = "python javascript ruby"
        similarity = utils.calculate_similarity(text1, text2)
        assert similarity < 0.5

    def test_partial_overlap(self):
        """Should detect partial similarity"""
        text1 = "I like Python programming"
        text2 = "Python programming is great"
        similarity = utils.calculate_similarity(text1, text2)
        assert 0 < similarity < 1


class TestFeature44RankingAgreement:
    """Feature 44: Ranking agreement score"""

    def test_perfect_agreement(self):
        """Should return high score for perfect agreement"""
        rankings = [
            ['A', 'B', 'C'],
            ['A', 'B', 'C'],
            ['A', 'B', 'C'],
        ]
        score = utils.calculate_ranking_agreement(rankings)
        assert score == 1.0

    def test_complete_disagreement(self):
        """Should return low score for disagreement"""
        rankings = [
            ['A', 'B', 'C'],
            ['C', 'B', 'A'],
        ]
        score = utils.calculate_ranking_agreement(rankings)
        assert score < 0.8


class TestFeature45AverageResponseTime:
    """Feature 45: Average response time calculation"""

    def test_calculate_average(self):
        """Should calculate average correctly"""
        times = [1.0, 2.0, 3.0]
        assert utils.calculate_average_response_time(times) == 2.0

    def test_empty_times(self):
        """Should handle empty list"""
        assert utils.calculate_average_response_time([]) == 0.0


class TestFeature46ModelStatistics:
    """Feature 46: Model performance statistics"""

    def test_calculate_statistics(self):
        """Should calculate model statistics"""
        model_rankings = {
            'gpt-4': [1, 2, 1, 3],
            'claude': [2, 1, 2, 1],
        }

        stats = utils.calculate_model_statistics(model_rankings)
        assert stats['gpt-4']['average_rank'] == 1.75
        assert stats['gpt-4']['best_rank'] == 1
        assert stats['gpt-4']['first_place_count'] == 2


class TestFeature47ConversationDuration:
    """Feature 47: Conversation duration calculation"""

    def test_calculate_duration(self):
        """Should calculate duration correctly"""
        start = "2024-01-01T10:00:00"
        end = "2024-01-01T10:05:00"
        duration = utils.calculate_conversation_duration(start, end)
        assert duration == 300.0  # 5 minutes in seconds


class TestFeature48QualityMetrics:
    """Feature 48: Response quality metrics"""

    def test_quality_metrics(self):
        """Should calculate quality metrics"""
        response = "This is a test. It has multiple sentences. And some code: `print('hello')`."

        metrics = utils.calculate_quality_metrics(response)
        assert metrics['word_count'] > 0
        assert metrics['sentence_count'] >= 2
        assert metrics['has_code_blocks'] is True

    def test_empty_response_metrics(self):
        """Should handle empty response"""
        metrics = utils.calculate_quality_metrics("")
        assert metrics['word_count'] == 0


class TestFeature49ModelDiversity:
    """Feature 49: Model diversity score"""

    def test_high_diversity(self):
        """Should detect high diversity"""
        responses = [
            "Python is a programming language",
            "JavaScript runs in browsers",
            "Rust is memory safe",
        ]
        diversity = utils.calculate_model_diversity(responses)
        assert diversity > 0.5

    def test_low_diversity(self):
        """Should detect low diversity"""
        responses = [
            "Python is great for programming",
            "Python is wonderful for programming",
        ]
        diversity = utils.calculate_model_diversity(responses)
        assert diversity < 0.5


class TestFeature50ConsensusScore:
    """Feature 50: Council consensus calculation"""

    def test_high_consensus(self):
        """Should calculate high consensus for agreement"""
        stage2_results = [
            {'model': 'a', 'parsed_ranking': ['Response A', 'Response B']},
            {'model': 'b', 'parsed_ranking': ['Response A', 'Response B']},
        ]
        label_to_model = {'Response A': 'gpt-4', 'Response B': 'claude'}

        result = utils.calculate_consensus_score(stage2_results, label_to_model)
        assert result['top_choice_agreement'] == 1.0
        assert result['most_agreed_model'] == 'gpt-4'

    def test_split_consensus(self):
        """Should handle split consensus"""
        stage2_results = [
            {'model': 'a', 'parsed_ranking': ['Response A', 'Response B']},
            {'model': 'b', 'parsed_ranking': ['Response B', 'Response A']},
        ]
        label_to_model = {'Response A': 'gpt-4', 'Response B': 'claude'}

        result = utils.calculate_consensus_score(stage2_results, label_to_model)
        assert result['top_choice_agreement'] == 0.5
