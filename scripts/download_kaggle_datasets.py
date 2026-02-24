"""
Script to download multiple datasets from Kaggle.

Requirements:
    - Kaggle API credentials (~/.kaggle/kaggle.json)
    - Install kaggle package: pip install kaggle

Usage:
    python scripts/download_kaggle_datasets.py
"""

import os
import sys
from pathlib import Path
from typing import List, Dict

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError:
    print("Error: Kaggle package not installed.")
    print("Please install it using: pip install kaggle")
    sys.exit(1)
except OSError as e:
    print("="*70)
    print("ERROR: Kaggle API credentials not found!")
    print("="*70)
    print("\nTo set up Kaggle API credentials:")
    print("\n1. Go to https://www.kaggle.com/")
    print("2. Log in and go to your Account settings")
    print("3. Scroll to 'API' section and click 'Create New Token'")
    print("4. This will download 'kaggle.json' file")
    print("\n5. Place the kaggle.json file in one of these locations:")
    print(f"   - Windows: C:\\Users\\{os.environ.get('USERNAME', 'YourUsername')}\\.kaggle\\kaggle.json")
    print("   - Linux/Mac: ~/.kaggle/kaggle.json")
    print("\n6. Make sure the directory exists (create it if needed)")
    print("\nOR set environment variables:")
    print("   - KAGGLE_USERNAME=your_username")
    print("   - KAGGLE_KEY=your_api_key")
    print("\n" + "="*70)
    sys.exit(1)


class KaggleDatasetDownloader:
    """Handle downloading multiple datasets from Kaggle."""
    
    def __init__(self, download_dir: str = None):
        """
        Initialize the Kaggle dataset downloader.
        
        Args:
            download_dir: Directory to download datasets to. 
                         Defaults to data/external/ relative to project root.
        """
        # Get project root (parent of scripts directory)
        self.project_root = Path(__file__).parent.parent
        
        # Set download directory
        if download_dir:
            self.download_dir = Path(download_dir)
        else:
            self.download_dir = self.project_root / "data" / "external"
        
        # Create download directory if it doesn't exist
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Kaggle API
        self.api = KaggleApi()
        try:
            self.api.authenticate()
        except OSError as e:
            print("="*70)
            print("ERROR: Failed to authenticate with Kaggle API")
            print("="*70)
            print(f"\n{str(e)}")
            print("\nPlease ensure kaggle.json is in the correct location.")
            print("="*70)
            raise
        
    def download_dataset(self, dataset_name: str, unzip: bool = True, 
                        force: bool = False, quiet: bool = False) -> bool:
        """
        Download a single dataset from Kaggle.
        
        Args:
            dataset_name: Kaggle dataset identifier (e.g., 'username/dataset-name')
            unzip: Whether to unzip the downloaded files
            force: Force download even if files exist
            quiet: Suppress output messages
            
        Returns:
            bool: True if download successful, False otherwise
        """
        try:
            # Create subdirectory for this dataset
            dataset_folder = dataset_name.replace('/', '_')
            target_path = self.download_dir / dataset_folder
            target_path.mkdir(parents=True, exist_ok=True)
            
            if not quiet:
                print(f"\n{'='*60}")
                print(f"Downloading: {dataset_name}")
                print(f"Target path: {target_path}")
                print(f"{'='*60}")
            
            # Download dataset
            self.api.dataset_download_files(
                dataset=dataset_name,
                path=str(target_path),
                unzip=unzip,
                force=force,
                quiet=quiet
            )
            
            if not quiet:
                print(f"✓ Successfully downloaded: {dataset_name}")
            
            return True
            
        except Exception as e:
            print(f"✗ Error downloading {dataset_name}: {str(e)}")
            return False
    
    def download_multiple_datasets(self, datasets: List[Dict[str, any]], 
                                   quiet: bool = False) -> Dict[str, bool]:
        """
        Download multiple datasets from Kaggle.
        
        Args:
            datasets: List of dataset configurations. Each dict should contain:
                     - 'name': Dataset identifier (required)
                     - 'unzip': Whether to unzip (optional, default True)
                     - 'force': Force download (optional, default False)
            quiet: Suppress output messages
            
        Returns:
            dict: Dataset names mapped to success status
        """
        results = {}
        
        print(f"\n{'#'*60}")
        print(f"Starting download of {len(datasets)} dataset(s)")
        print(f"{'#'*60}\n")
        
        for dataset_config in datasets:
            dataset_name = dataset_config.get('name')
            if not dataset_name:
                print("Warning: Skipping dataset config without 'name' field")
                continue
            
            unzip = dataset_config.get('unzip', True)
            force = dataset_config.get('force', False)
            
            success = self.download_dataset(
                dataset_name=dataset_name,
                unzip=unzip,
                force=force,
                quiet=quiet
            )
            results[dataset_name] = success
        
        # Print summary
        print(f"\n{'#'*60}")
        print("Download Summary")
        print(f"{'#'*60}")
        successful = sum(1 for v in results.values() if v)
        failed = len(results) - successful
        print(f"Total: {len(results)} | Successful: {successful} | Failed: {failed}")
        print(f"{'#'*60}\n")
        
        return results


def main():
    """Main function to download configured datasets."""
    
    # ========================================================================
    # CONFIGURE YOUR DATASETS HERE
    # ========================================================================
    # Add datasets you want to download in the list below
    # Format: {'name': 'owner/dataset-name', 'unzip': True, 'force': False}
    
    datasets_to_download = [

        {
            'name': 'trainingdatapro/ripe-strawberries-detection',
            'unzip': True,
            'force': False
        },
        {
            'name': 'cookiefinder/tomato-disease-multiple-sources',
            'unzip': True,
            'force': False
        },
        {
            'name': 'jawadulkarim117/tomato-flower-3-class',
            'unzip': True,
            'force': False
        },
        {
            'name': 'arjunsudheer326/tomato-plant-stages-dataset',
            'unzip': True,
            'force': False
        },
        {
            'name': 'kotameyan/strawberry-growth-stage-datasets',
            'unzip': True,
            'force': False
        },
        {
            'name': 'zakariamuhammad/strawberry',
            'unzip': True,
            'force': False
        },
        {
            'name': 'abdallahalidev/plantvillage-dataset',
            'unzip': True,
            'force': False
        },
    ]
    
    # ========================================================================
    # DOWNLOAD EXECUTION
    # ========================================================================
    
    if not datasets_to_download:
        print("No datasets configured for download.")
        print("Please add dataset names to the 'datasets_to_download' list.")
        return
    
    # Initialize downloader
    downloader = KaggleDatasetDownloader()
    
    print(f"Download directory: {downloader.download_dir.absolute()}")
    
    # Download all configured datasets
    results = downloader.download_multiple_datasets(datasets_to_download)
    
    # Exit with appropriate status code
    if all(results.values()):
        print("\n✓ All datasets downloaded successfully!")
        sys.exit(0)
    else:
        print("\n⚠ Some datasets failed to download. Check the logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
