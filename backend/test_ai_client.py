"""
Test suite for OpenRouterClient.
"""

import asyncio
import json
from unittest.mock import patch, Mock

import httpx
import pytest
import pytest_asyncio

from ai_client import OpenRouterClient, OpenRouterError

pytestmark = pytest.mark.asyncio

class TestOpenRouterClient:
    """Test cases for OpenRouterClient."""
    
    @pytest_asyncio.fixture(autouse=True, scope="function")
    async def setup_client(self):
        """Set up test fixtures."""
        self.api_key = "test_key"
        self.client = OpenRouterClient(self.api_key)
        yield
        await self.client.client.aclose()
    
    async def test_init_no_api_key(self):
        """Test initialization without API key."""
        with pytest.raises(ValueError):
            OpenRouterClient("")
    
    async def test_init_with_api_key(self):
        """Test initialization with API key."""
        client = OpenRouterClient(self.api_key)
        assert client.api_key == self.api_key
        assert isinstance(client.client, httpx.AsyncClient)
        await client.client.aclose()
    
    async def test_summarize_success(self):
        """Test successful summarization."""
        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": "Test summary"
                    }
                }
            ]
        }
        
        with patch.object(self.client, '_make_request', return_value=mock_response):
            summary = await self.client.summarize(
                "Test transcript",
                "test/model",
                length="short"
            )
            assert summary == "Test summary"
    
    async def test_summarize_failure(self):
        """Test failed summarization."""
        with patch.object(self.client, '_make_request', side_effect=OpenRouterError("Test error")):
            summary = await self.client.summarize(
                "Test transcript",
                "test/model"
            )
            assert summary is None
    
    async def test_generate_study_guide_success(self):
        """Test successful study guide generation."""
        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": "Test study guide"
                    }
                }
            ]
        }
        
        with patch.object(self.client, '_make_request', return_value=mock_response):
            guide = await self.client.generate_study_guide(
                "Test content",
                "test/model",
                guide_type="outline"
            )
            assert guide == "Test study guide"
    
    async def test_generate_study_guide_failure(self):
        """Test failed study guide generation."""
        with patch.object(self.client, '_make_request', side_effect=OpenRouterError("Test error")):
            guide = await self.client.generate_study_guide(
                "Test content",
                "test/model"
            )
            assert guide is None
    
    async def test_make_request_rate_limit(self):
        """Test rate limit handling."""
        mock_response = Mock()
        mock_response.status_code = 429
        
        with patch.object(self.client.client, 'post', return_value=mock_response):
            with pytest.raises(OpenRouterError):
                await self.client._make_request("Test prompt", "test/model")
    
    async def test_make_request_network_error(self):
        """Test network error handling."""
        with patch.object(self.client.client, 'post', side_effect=httpx.NetworkError("Test error")):
            with pytest.raises(OpenRouterError):
                await self.client._make_request("Test prompt", "test/model")
    
    async def test_construct_summary_prompt(self):
        """Test summary prompt construction."""
        transcript = "Test transcript"
        
        # Test short length
        prompt = self.client._construct_summary_prompt(transcript, "short")
        assert "2-3 sentences" in prompt
        assert transcript in prompt
        
        # Test with timestamps
        prompt = self.client._construct_summary_prompt(transcript, include_timestamps=True)
        assert "[HH:MM:SS]" in prompt
    
    async def test_construct_study_guide_prompt(self):
        """Test study guide prompt construction."""
        content = "Test content"
        
        # Test different guide types
        prompt = self.client._construct_study_guide_prompt(content, "questions")
        assert "Generate practice questions" in prompt
        assert content in prompt
        
        # Test custom prompt
        custom = "Custom prompt for {text}"
        prompt = self.client._construct_study_guide_prompt(content, custom_prompt=custom)
        assert prompt == custom.format(text=content)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
