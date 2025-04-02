"""
Test suite for SettingsManager.
"""

import json
import os
import tempfile
from unittest.mock import patch

import pytest

from settings_manager import SettingsManager

class TestSettingsManager:
    """Test cases for SettingsManager."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        # Create temporary settings file
        self.temp_dir = tempfile.mkdtemp()
        self.settings_file = os.path.join(self.temp_dir, "test_settings.json")
        
        # Create settings manager
        self.manager = SettingsManager(self.settings_file)
        
        yield
        
        # Cleanup
        try:
            os.remove(self.settings_file)
            os.rmdir(self.temp_dir)
        except:
            pass
    
    def test_init_default_settings(self):
        """Test initialization with default settings."""
        settings = self.manager.load_settings()
        
        assert "transcriber" in settings
        assert "ai" in settings
        assert "video" in settings
        assert "output" in settings
        
        assert settings["transcriber"]["model_size"] == "base"
        assert settings["ai"]["summary_length"] == "medium"
        assert settings["video"]["burn_subtitles"] is False
        assert settings["output"]["save_transcripts"] is True
    
    def test_save_load_settings(self):
        """Test saving and loading settings."""
        # Modify settings
        settings = self.manager.load_settings()
        settings["transcriber"]["model_size"] = "large"
        settings["ai"]["api_key"] = "test_key"
        
        # Save settings
        self.manager.save_settings(settings)
        
        # Create new manager and load settings
        new_manager = SettingsManager(self.settings_file)
        loaded_settings = new_manager.load_settings()
        
        assert loaded_settings["transcriber"]["model_size"] == "large"
        assert loaded_settings["ai"]["api_key"] == "test_key"
    
    def test_merge_settings(self):
        """Test settings merging."""
        base = {
            "transcriber": {
                "model_size": "base",
                "device": "auto"
            },
            "ai": {
                "api_key": ""
            }
        }
        
        update = {
            "transcriber": {
                "model_size": "large"
            },
            "video": {
                "burn_subtitles": True
            }
        }
        
        merged = self.manager._merge_settings(base, update)
        
        assert merged["transcriber"]["model_size"] == "large"
        assert merged["transcriber"]["device"] == "auto"
        assert merged["ai"]["api_key"] == ""
        assert merged["video"]["burn_subtitles"] is True
    
    def test_get_setting(self):
        """Test getting specific settings."""
        # Set some settings
        self.manager.set_setting("transcriber.model_size", "large")
        self.manager.set_setting("ai.api_key", "test_key")
        
        # Test getting settings
        assert self.manager.get_setting("transcriber.model_size") == "large"
        assert self.manager.get_setting("ai.api_key") == "test_key"
        assert self.manager.get_setting("nonexistent.key", "default") == "default"
    
    def test_set_setting(self):
        """Test setting specific settings."""
        # Set nested setting
        self.manager.set_setting("transcriber.new_option", "value")
        
        # Verify setting was saved
        settings = self.manager.load_settings()
        assert settings["transcriber"]["new_option"] == "value"
        
        # Set setting in new section
        self.manager.set_setting("new_section.option", "value")
        settings = self.manager.load_settings()
        assert settings["new_section"]["option"] == "value"
    
    def test_reset_settings(self):
        """Test resetting settings to defaults."""
        # Modify settings
        self.manager.set_setting("transcriber.model_size", "large")
        self.manager.set_setting("ai.api_key", "test_key")
        
        # Reset settings
        self.manager.reset_settings()
        
        # Verify defaults
        settings = self.manager.load_settings()
        assert settings["transcriber"]["model_size"] == "base"
        assert settings["ai"]["api_key"] == ""
    
    def test_invalid_settings_file(self):
        """Test handling invalid settings file."""
        # Write invalid JSON
        with open(self.settings_file, "w") as f:
            f.write("invalid json")
        
        # Load settings should return defaults
        settings = self.manager.load_settings()
        assert settings["transcriber"]["model_size"] == "base"
    
    def test_missing_settings_file(self):
        """Test handling missing settings file."""
        # Use non-existent file
        manager = SettingsManager(os.path.join(self.temp_dir, "nonexistent.json"))
        
        # Load settings should return defaults
        settings = manager.load_settings()
        assert settings["transcriber"]["model_size"] == "base"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
