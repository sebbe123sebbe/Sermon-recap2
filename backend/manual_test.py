"""
Manual test script for the Video Summarizer Pro pipeline.

This script exercises the full pipeline with various inputs and configurations
to verify functionality in a real-world setting.
"""

import asyncio
import logging
import sys
from pathlib import Path

from main_controller import MainController

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('manual_test.log')
    ]
)
logger = logging.getLogger(__name__)

# Test configurations
TEST_CONFIGS = [
    {
        "name": "Basic Test (CPU)",
        "config": {
            "model_size": "tiny",
            "device": "cpu",
            "compute_type": "int8",
            "language": "en",
            "summary_length": "short"
        }
    },
    {
        "name": "CUDA Test",
        "config": {
            "model_size": "base",
            "device": "cuda",
            "compute_type": "float16",
            "language": None,  # Auto-detect
            "summary_length": "medium"
        }
    },
    {
        "name": "High Quality",
        "config": {
            "model_size": "large-v3",
            "device": "cuda",
            "compute_type": "float16",
            "language": None,
            "summary_length": "long"
        }
    }
]

def status_callback(status: str):
    """Handle status updates."""
    logger.info(f"Status: {status}")

def progress_callback(progress: float):
    """Handle progress updates."""
    logger.info(f"Progress: {progress:.1f}%")

def completion_callback(success: bool, result: dict):
    """Handle completion."""
    if success:
        logger.info("Pipeline completed successfully!")
        logger.info(f"Results: {result}")
    else:
        logger.error(f"Pipeline failed: {result.get('error')}")

async def run_test(test_name: str, input_path: Path, output_dir: Path, config: dict):
    """Run a single test configuration."""
    logger.info(f"\n{'='*80}\nRunning test: {test_name}\n{'='*80}")
    
    try:
        # Initialize controller
        controller = MainController()
        
        # Update config with paths
        full_config = {
            **config,
            "video_path": str(input_path),  
            "output_dir": str(output_dir / test_name.lower().replace(" ", "_"))
        }
        
        # Create output directory
        Path(full_config["output_dir"]).mkdir(parents=True, exist_ok=True)
        
        # Log configuration
        logger.info(f"Configuration:\n{full_config}")
        
        # Run pipeline
        await controller.run_full_pipeline(
            full_config,
            status_callback=status_callback,
            progress_callback=progress_callback,
            completion_callback=completion_callback
        )
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}", exc_info=True)

async def main():
    """Run all test configurations."""
    if len(sys.argv) != 2:
        print("Usage: python manual_test.py <input_file>")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    if not input_path.exists():
        logger.error(f"Test video not found: {input_path}")
        sys.exit(1)
    
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    for test in TEST_CONFIGS:
        await run_test(test["name"], input_path, output_dir, test["config"])

if __name__ == "__main__":
    asyncio.run(main())
