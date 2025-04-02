"""
AI Client for OpenRouter API integration.
Handles communication with OpenRouter for text generation tasks.
"""

import json
import logging
from typing import Optional, Dict, Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Configure logging
logger = logging.getLogger(__name__)

class OpenRouterError(Exception):
    """Base exception for OpenRouter API errors."""
    pass

class OpenRouterClient:
    """Client for interacting with OpenRouter API."""
    
    BASE_URL = "https://openrouter.ai/api/v1"
    
    def __init__(self, api_key: str):
        """
        Initialize OpenRouter client.
        
        Args:
            api_key: OpenRouter API key
        """
        if not api_key:
            raise ValueError("API key is required")
            
        self.api_key = api_key
        self.client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://github.com/your-repo/video-summarizer-pro",  # Update with actual repo
                "X-Title": "Video Summarizer Pro"
            },
            timeout=60.0  # 60 second timeout
        )
        logger.info("Initialized OpenRouter client")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.client.aclose()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError))
    )
    async def _make_request(self, prompt: str, model: str, **kwargs) -> Dict[str, Any]:
        """
        Make a request to OpenRouter API.
        
        Args:
            prompt: The prompt to send
            model: Model identifier (e.g., 'anthropic/claude-2')
            **kwargs: Additional parameters for the API
            
        Returns:
            dict: API response
            
        Raises:
            OpenRouterError: If the API request fails
        """
        try:
            response = await self.client.post(
                f"{self.BASE_URL}/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    **kwargs
                }
            )
            
            if response.status_code == 429:
                logger.warning("Rate limit exceeded")
                raise OpenRouterError("Rate limit exceeded")
                
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error occurred: {str(e)}")
            raise OpenRouterError(f"HTTP error: {str(e)}")
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse response: {str(e)}")
            raise OpenRouterError(f"Invalid JSON response: {str(e)}")
            
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise OpenRouterError(f"Unexpected error: {str(e)}")
    
    def _construct_summary_prompt(self, transcript: str, length: str = "medium", include_timestamps: bool = False) -> str:
        """
        Construct a prompt for video summarization.
        
        Args:
            transcript: Video transcript to summarize
            length: Desired summary length ('short', 'medium', or 'long')
            include_timestamps: Whether to include timestamps in the summary
            
        Returns:
            str: Constructed prompt
        """
        length_guide = {
            "short": "2-3 sentences",
            "medium": "4-6 sentences",
            "long": "7-10 sentences"
        }.get(length, "4-6 sentences")
        
        timestamp_guide = "Include key timestamps from the transcript in [HH:MM:SS] format." if include_timestamps else ""
        
        return f"""Summarize the following video transcript in {length_guide}. Focus on the main points and key takeaways. {timestamp_guide}

Transcript:
{transcript}

Summary:"""
    
    def _construct_study_guide_prompt(self, text_input: str, guide_type: str = "outline", difficulty: str = "intermediate", custom_prompt: Optional[str] = None) -> str:
        """
        Construct a prompt for study guide generation.
        
        Args:
            text_input: Text to create study guide from
            guide_type: Type of study guide ('outline', 'questions', 'flashcards', or 'notes')
            difficulty: Difficulty level ('beginner', 'intermediate', or 'advanced')
            custom_prompt: Optional custom prompt template
            
        Returns:
            str: Constructed prompt
        """
        if custom_prompt:
            return custom_prompt.format(text=text_input)
            
        guide_formats = {
            "outline": "Create a hierarchical outline",
            "questions": "Generate practice questions with answers",
            "flashcards": "Create flashcard pairs (term::definition)",
            "notes": "Write detailed study notes"
        }
        
        difficulty_guides = {
            "beginner": "Focus on fundamental concepts and basic terminology",
            "intermediate": "Include both basic and advanced concepts",
            "advanced": "Emphasize complex relationships and technical details"
        }
        
        format_guide = guide_formats.get(guide_type, guide_formats["outline"])
        difficulty_guide = difficulty_guides.get(difficulty, difficulty_guides["intermediate"])
        
        return f"""Create a study guide from the following content. {format_guide} that {difficulty_guide}.

Content:
{text_input}

Study Guide:"""
    
    async def summarize(
        self,
        transcript: str,
        model: str,
        custom_prompt: Optional[str] = None,
        length: str = "medium",
        include_timestamps: bool = False
    ) -> Optional[str]:
        """
        Generate a summary of the video transcript.
        
        Args:
            transcript: Video transcript to summarize
            model: Model to use for summarization
            custom_prompt: Optional custom prompt template
            length: Desired summary length ('short', 'medium', or 'long')
            include_timestamps: Whether to include timestamps
            
        Returns:
            str: Generated summary or None if failed
        """
        try:
            prompt = custom_prompt or self._construct_summary_prompt(transcript, length, include_timestamps)
            
            logger.info(f"Requesting summary using model: {model}")
            response = await self._make_request(prompt, model)
            
            if not response.get("choices"):
                logger.error("No choices in response")
                return None
                
            summary = response["choices"][0]["message"]["content"].strip()
            logger.info("Successfully generated summary")
            return summary
            
        except OpenRouterError as e:
            logger.error(f"Failed to generate summary: {str(e)}")
            return None
    
    async def generate_study_guide(
        self,
        text_input: str,
        model: str,
        guide_type: str = "outline",
        difficulty: str = "intermediate",
        custom_prompt: Optional[str] = None
    ) -> Optional[str]:
        """
        Generate a study guide from the input text.
        
        Args:
            text_input: Text to create study guide from
            model: Model to use for generation
            guide_type: Type of study guide ('outline', 'questions', 'flashcards', or 'notes')
            difficulty: Difficulty level ('beginner', 'intermediate', or 'advanced')
            custom_prompt: Optional custom prompt template
            
        Returns:
            str: Generated study guide or None if failed
        """
        try:
            prompt = self._construct_study_guide_prompt(text_input, guide_type, difficulty, custom_prompt)
            
            logger.info(f"Requesting study guide using model: {model}")
            response = await self._make_request(prompt, model)
            
            if not response.get("choices"):
                logger.error("No choices in response")
                return None
                
            guide = response["choices"][0]["message"]["content"].strip()
            logger.info("Successfully generated study guide")
            return guide
            
        except OpenRouterError as e:
            logger.error(f"Failed to generate study guide: {str(e)}")
            return None
