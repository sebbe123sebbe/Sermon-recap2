"""
Settings manager for Video Summarizer Pro.
Handles loading and saving application settings.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

class SettingsManager:
    """Manages application settings."""
    
    def __init__(self, settings_file: str = None):
        """
        Initialize settings manager.
        
        Args:
            settings_file: Optional path to settings file
        """
        if settings_file is None:
            # Use default location in user's home directory
            settings_dir = os.path.join(str(Path.home()), ".video_summarizer_pro")
            os.makedirs(settings_dir, exist_ok=True)
            settings_file = os.path.join(settings_dir, "settings.json")
        
        self.settings_file = settings_file
        self._settings = self._load_default_settings()
    
    def _load_default_settings(self) -> Dict[str, Any]:
        """
        Load default settings.
        
        Returns:
            dict: Default settings
        """
        return {
            "transcriber": {
                "model_size": "base",
                "device": "auto",
                "compute_type": "float16"
            },
            "ai": {
                "api_key": "",
                "models": [
                    "anthropic/claude-2",
                    "anthropic/claude-instant-1",
                    "google/palm-2",
                    "meta-llama/llama-2-70b-chat"
                ],
                "summary_length": "medium",
                "guide_type": "outline",
                "difficulty": "intermediate"
            },
            "video": {
                "trim_preview_duration": 10,
                "burn_subtitles": False,
                "recap_duration": 60
            },
            "output": {
                "formats": ["txt", "srt", "vtt"],
                "save_transcripts": True,
                "save_summary": True,
                "save_study_guide": True
            }
        }
    
    def load_settings(self) -> Dict[str, Any]:
        """
        Load settings from file.
        
        Returns:
            dict: Current settings
        """
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, "r") as f:
                    stored_settings = json.load(f)
                    
                # Update default settings with stored settings
                self._settings = self._merge_settings(self._settings, stored_settings)
                logger.info("Settings loaded successfully")
            else:
                logger.info("No settings file found, using defaults")
                
        except Exception as e:
            logger.error(f"Failed to load settings: {str(e)}")
        
        return self._settings
    
    def save_settings(self, settings: Dict[str, Any]) -> None:
        """
        Save settings to file.
        
        Args:
            settings: Settings to save
        """
        try:
            # Update current settings
            self._settings = self._merge_settings(self._settings, settings)
            
            # Save to file
            with open(self.settings_file, "w") as f:
                json.dump(self._settings, f, indent=4)
            
            logger.info("Settings saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save settings: {str(e)}")
    
    def _merge_settings(self, base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two settings dictionaries.
        
        Args:
            base: Base settings
            update: Settings to update with
            
        Returns:
            dict: Merged settings
        """
        result = base.copy()
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_settings(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """
        Get a specific setting value.
        
        Args:
            key: Setting key (dot notation supported)
            default: Default value if not found
            
        Returns:
            Setting value or default
        """
        try:
            value = self._settings
            for part in key.split("."):
                value = value[part]
            return value
        except (KeyError, TypeError):
            return default
    
    def set_setting(self, key: str, value: Any) -> None:
        """
        Set a specific setting value.
        
        Args:
            key: Setting key (dot notation supported)
            value: Value to set
        """
        try:
            parts = key.split(".")
            target = self._settings
            
            # Navigate to the target dictionary
            for part in parts[:-1]:
                if part not in target:
                    target[part] = {}
                target = target[part]
            
            # Set the value
            target[parts[-1]] = value
            
            # Save settings
            self.save_settings(self._settings)
            
        except Exception as e:
            logger.error(f"Failed to set setting {key}: {str(e)}")
    
    def reset_settings(self) -> None:
        """Reset settings to defaults."""
        self._settings = self._load_default_settings()
        self.save_settings(self._settings)
