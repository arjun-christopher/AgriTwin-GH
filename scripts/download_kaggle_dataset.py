"""
Download Kaggle Dataset for AgriTwin-GH Project

This script downloads the tomato greenhouse dataset from Kaggle and organizes
it into the appropriate directories in the project structure.

Dataset: arjunchristopher/tomato-greenhouse-environment-growth-and-disease

Usage:
    python scripts/download_kaggle_dataset.py [--force] [--dry-run]

Requirements:
    - Kaggle API credentials configured (~/.kaggle/kaggle.json)
    - Install: pip install kaggle

Dataset Structure:
    - Tomato Diseases (6 disease types) -> data/external/Tomato Diseases/
    - Tomato Growth Stages (6 stages) -> data/external/Tomato Growth Stages/
    - Tomato Healthy Leaves -> data/external/Tomato Healthy Leaves/
    - Weather Data (CSV files) -> data/external/Weather Data/
    - Greenhouse Indoor Conditions (CSV) -> data/processed/Greenhouse Indoor Conditions/
"""

import os
import sys
import argparse
import logging
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import zipfile

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/kaggle_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TEMP_DOWNLOAD_DIR = PROJECT_ROOT / "temp_kaggle_download"

# Kaggle dataset identifier
KAGGLE_DATASET = "arjunchristopher/tomato-greenhouse-environment-growth-and-disease"

# Directory mapping from Kaggle dataset to local repository
# Format: {kaggle_folder: local_destination}
DIRECTORY_MAPPING = {
    # Disease folders
    "Tomato Diseases/Tomato_Early_Blight": "data/external/Tomato Diseases/Tomato_Early_Blight",
    "Tomato Diseases/Tomato_Late_Blight": "data/external/Tomato Diseases/Tomato_Late_Blight",
    "Tomato Diseases/Tomato_Leaf_Mold": "data/external/Tomato Diseases/Tomato_Leaf_Mold",
    "Tomato Diseases/Tomato_Powdery_Mildew": "data/external/Tomato Diseases/Tomato_Powdery_Mildew",
    "Tomato Diseases/Tomato_Septoria_Leaf_Spot": "data/external/Tomato Diseases/Tomato_Septoria_Leaf_Spot",
    "Tomato Diseases/Tomato_Spider_Mites": "data/external/Tomato Diseases/Tomato_Spider_Mites",
    
    # Growth stage folders
    "Tomato Growth Stages/Stage1_Seedling": "data/external/Tomato Growth Stages/Stage1_Seedling",
    "Tomato Growth Stages/Stage2_Early_Vegetative": "data/external/Tomato Growth Stages/Stage2_Early_Vegetative",
    "Tomato Growth Stages/Stage3_Flowering_Initiation": "data/external/Tomato Growth Stages/Stage3_Flowering_Initiation",
    "Tomato Growth Stages/Stage4_Flowering": "data/external/Tomato Growth Stages/Stage4_Flowering",
    "Tomato Growth Stages/Stage5_Unripe": "data/external/Tomato Growth Stages/Stage5_Unripe",
    "Tomato Growth Stages/Stage6_Ripe": "data/external/Tomato Growth Stages/Stage6_Ripe",
    
    # Healthy leaves
    "Tomato Healthy Leaves": "data/external/Tomato Healthy Leaves",
    
    # Weather data
    "Weather Data": "data/external/Weather Data",
    
    # Greenhouse indoor conditions (processed data)
    "Greenhouse Indoor Conditions": "data/processed/Greenhouse Indoor Conditions",
}


def check_kaggle_credentials() -> bool:
    """Check if Kaggle API credentials are configured."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    
    if not kaggle_json.exists():
        logger.error(f"Kaggle credentials not found at {kaggle_json}")
        logger.error("Please follow these steps:")
        logger.error("1. Go to https://www.kaggle.com/account")
        logger.error("2. Click 'Create New API Token'")
        logger.error("3. Place the downloaded kaggle.json in ~/.kaggle/")
        logger.error("4. On Windows: C:\\Users\\<username>\\.kaggle\\kaggle.json")
        return False
    
    logger.info(f"✓ Kaggle credentials found at {kaggle_json}")
    return True


def download_dataset(force: bool = False) -> Path:
    """
    Download the Kaggle dataset.
    
    Args:
        force: If True, re-download even if already exists
        
    Returns:
        Path to the downloaded dataset directory
    """
    try:
        import kaggle
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        logger.error("Kaggle package not installed. Install with: pip install kaggle")
        sys.exit(1)
    
    # Create temp directory
    TEMP_DOWNLOAD_DIR.mkdir(exist_ok=True)
    
    # Check if already downloaded
    if not force and (TEMP_DOWNLOAD_DIR / "dataset.zip").exists():
        logger.info(f"Dataset already downloaded at {TEMP_DOWNLOAD_DIR}")
        return TEMP_DOWNLOAD_DIR
    
    logger.info(f"Downloading dataset: {KAGGLE_DATASET}")
    logger.info(f"Download location: {TEMP_DOWNLOAD_DIR}")
    
    try:
        # Initialize Kaggle API
        api = KaggleApi()
        api.authenticate()
        
        # Download dataset
        api.dataset_download_files(
            KAGGLE_DATASET,
            path=str(TEMP_DOWNLOAD_DIR),
            unzip=True
        )
        
        logger.info("✓ Dataset downloaded successfully")
        return TEMP_DOWNLOAD_DIR
        
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        sys.exit(1)


def organize_files(source_dir: Path, dry_run: bool = False) -> Dict[str, int]:
    """
    Organize downloaded files into project structure.
    
    Args:
        source_dir: Directory containing downloaded dataset
        dry_run: If True, only show what would be done
        
    Returns:
        Dictionary with copy statistics
    """
    stats = {
        "folders_created": 0,
        "files_copied": 0,
        "files_skipped": 0,
        "errors": 0
    }
    
    logger.info("Organizing files into project structure...")
    
    if dry_run:
        logger.info("DRY RUN MODE - No files will be copied")
    
    for kaggle_path, local_path in DIRECTORY_MAPPING.items():
        source = source_dir / kaggle_path
        destination = PROJECT_ROOT / local_path
        
        if not source.exists():
            logger.warning(f"Source not found: {kaggle_path}")
            continue
        
        logger.info(f"Processing: {kaggle_path}")
        logger.info(f"  → {local_path}")
        
        if not dry_run:
            destination.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                if source.is_dir():
                    # Copy directory contents
                    if not destination.exists():
                        shutil.copytree(source, destination)
                        stats["folders_created"] += 1
                        logger.info(f"  ✓ Created directory: {destination.name}")
                    else:
                        # Copy files into existing directory
                        for item in source.rglob("*"):
                            if item.is_file():
                                rel_path = item.relative_to(source)
                                dest_file = destination / rel_path
                                
                                if dest_file.exists():
                                    stats["files_skipped"] += 1
                                else:
                                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                                    shutil.copy2(item, dest_file)
                                    stats["files_copied"] += 1
                        
                        logger.info(f"  ✓ Updated directory: {destination.name}")
                else:
                    # Copy single file
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if destination.exists():
                        stats["files_skipped"] += 1
                    else:
                        shutil.copy2(source, destination)
                        stats["files_copied"] += 1
                    
            except Exception as e:
                logger.error(f"  ✗ Error copying {kaggle_path}: {e}")
                stats["errors"] += 1
        else:
            # Dry run - just count files
            if source.is_dir():
                file_count = sum(1 for _ in source.rglob("*") if _.is_file())
                logger.info(f"  Would copy {file_count} files")
            else:
                logger.info(f"  Would copy 1 file")
    
    return stats


def cleanup_temp_files(keep_download: bool = False):
    """
    Clean up temporary download directory.
    
    Args:
        keep_download: If True, keep the downloaded files
    """
    if keep_download:
        logger.info(f"Keeping temporary files at {TEMP_DOWNLOAD_DIR}")
        return
    
    try:
        if TEMP_DOWNLOAD_DIR.exists():
            shutil.rmtree(TEMP_DOWNLOAD_DIR)
            logger.info(f"✓ Cleaned up temporary files at {TEMP_DOWNLOAD_DIR}")
    except Exception as e:
        logger.warning(f"Could not clean up temporary files: {e}")


def verify_dataset_structure():
    """Verify that all expected directories are in place."""
    logger.info("Verifying dataset structure...")
    
    missing = []
    present = []
    
    for local_path in DIRECTORY_MAPPING.values():
        full_path = PROJECT_ROOT / local_path
        if full_path.exists():
            file_count = sum(1 for _ in full_path.rglob("*") if _.is_file())
            present.append((local_path, file_count))
        else:
            missing.append(local_path)
    
    # Report present directories
    if present:
        logger.info(f"✓ Found {len(present)} directories:")
        for path, count in present:
            logger.info(f"  - {path} ({count} files)")
    
    # Report missing directories
    if missing:
        logger.warning(f"Missing {len(missing)} directories:")
        for path in missing:
            logger.warning(f"  - {path}")
    else:
        logger.info("✓ All expected directories are present")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Download and organize Kaggle dataset for AgriTwin-GH"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if dataset exists"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually copying files"
    )
    parser.add_argument(
        "--keep-download",
        action="store_true",
        help="Keep the temporary download directory after organizing"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify the current dataset structure without downloading"
    )
    
    args = parser.parse_args()
    
    # Create logs directory if it doesn't exist
    (PROJECT_ROOT / "logs").mkdir(exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("AgriTwin-GH Kaggle Dataset Download Script")
    logger.info(f"Dataset: {KAGGLE_DATASET}")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)
    
    # Verify only mode
    if args.verify_only:
        verify_dataset_structure()
        return
    
    # Check prerequisites
    if not check_kaggle_credentials():
        sys.exit(1)
    
    # Download dataset
    download_dir = download_dataset(force=args.force)
    
    # Organize files
    stats = organize_files(download_dir, dry_run=args.dry_run)
    
    # Print statistics
    logger.info("=" * 70)
    logger.info("Summary:")
    logger.info(f"  Folders created: {stats['folders_created']}")
    logger.info(f"  Files copied: {stats['files_copied']}")
    logger.info(f"  Files skipped (already exist): {stats['files_skipped']}")
    logger.info(f"  Errors: {stats['errors']}")
    logger.info("=" * 70)
    
    # Verify structure
    if not args.dry_run:
        verify_dataset_structure()
    
    # Cleanup
    if not args.dry_run:
        cleanup_temp_files(keep_download=args.keep_download)
    
    logger.info(f"✓ Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if args.dry_run:
        logger.info("\nThis was a DRY RUN. Re-run without --dry-run to actually copy files.")


if __name__ == "__main__":
    main()
