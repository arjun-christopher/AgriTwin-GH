"""
Upload Agricultural Images to MinIO and Store Metadata in PostgreSQL

This script processes the AgriTwin-GH image datasets and:
1. Uploads images to MinIO object storage
2. Stores metadata in PostgreSQL database
3. Organizes images by category (disease, growth_stage, healthy)
4. Handles errors and provides progress tracking

Usage:
    python upload_images_to_minio.py [--dry-run] [--batch-size 100]
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import hashlib
import json

# Third-party imports
from minio import Minio
from minio.error import S3Error
import psycopg2
from psycopg2.extras import execute_batch
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.minio_config import MinIOConfig, MinIOPathBuilder

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/image_upload.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ImageUploader:
    """Handles uploading images to MinIO and storing metadata in PostgreSQL"""
    
    def __init__(self, config: MinIOConfig, dry_run: bool = False):
        """
        Initialize the uploader
        
        Args:
            config: MinIO configuration
            dry_run: If True, don't actually upload or store data
        """
        self.config = config
        self.dry_run = dry_run
        self.path_builder = MinIOPathBuilder()
        
        # Statistics
        self.stats = {
            'total_files': 0,
            'uploaded': 0,
            'failed': 0,
            'skipped': 0,
            'total_size_bytes': 0
        }
        
        # Initialize connections
        self.minio_client = None
        self.db_conn = None
        
        if not dry_run:
            self._init_minio()
            self._init_database()
    
    def _init_minio(self):
        """Initialize MinIO client and create buckets if needed"""
        try:
            logger.info(f"Connecting to MinIO at {self.config.endpoint}")
            self.minio_client = Minio(
                self.config.endpoint,
                access_key=self.config.access_key,
                secret_key=self.config.secret_key,
                secure=self.config.secure
            )
            
            # Create buckets if they don't exist
            for bucket_name in self.config.buckets.values():
                if not self.minio_client.bucket_exists(bucket_name):
                    logger.info(f"Creating bucket: {bucket_name}")
                    self.minio_client.make_bucket(bucket_name)
                else:
                    logger.info(f"Bucket exists: {bucket_name}")
            
            logger.info("MinIO client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize MinIO client: {e}")
            raise
    
    def _init_database(self):
        """Initialize PostgreSQL connection"""
        try:
            logger.info("Connecting to PostgreSQL")
            self.db_conn = psycopg2.connect(
                host=os.getenv("DB_HOST", "localhost"),
                port=os.getenv("DB_PORT", "5432"),
                database=os.getenv("DB_NAME", "agritwin_db"),
                user=os.getenv("DB_USER", "postgres"),
                password=os.getenv("DB_PASSWORD", "")
            )
            self.db_conn.autocommit = False
            logger.info("PostgreSQL connection established")
            
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise
    
    def _get_image_info(self, file_path: Path) -> Dict:
        """
        Extract image information
        
        Args:
            file_path: Path to image file
            
        Returns:
            dict: Image information (width, height, format, etc.)
        """
        try:
            with Image.open(file_path) as img:
                return {
                    'width': img.width,
                    'height': img.height,
                    'format': img.format,
                    'mode': img.mode,
                    'size_bytes': file_path.stat().st_size
                }
        except Exception as e:
            logger.warning(f"Could not read image info for {file_path}: {e}")
            return {
                'width': None,
                'height': None,
                'format': None,
                'mode': None,
                'size_bytes': file_path.stat().st_size
            }
    
    def _parse_folder_structure(self, file_path: Path, base_path: Path) -> Dict:
        """
        Parse category and subcategory from folder structure
        
        Args:
            file_path: Full path to image file
            base_path: Base data directory path
            
        Returns:
            dict: Parsed metadata (category, subcategory, label, etc.)
        """
        relative_path = file_path.relative_to(base_path)
        parts = relative_path.parts
        
        metadata = {
            'category': None,
            'subcategory': None,
            'label': None,
            'source_dataset': None
        }
        
        # Determine category based on folder name
        for part in parts:
            part_lower = part.lower()
            
            if 'disease' in part_lower:
                metadata['category'] = 'disease'
                metadata['source_dataset'] = 'Tomato Diseases'
                # Next part should be specific disease
                idx = parts.index(part)
                if idx + 1 < len(parts):
                    disease_name = parts[idx + 1]
                    metadata['subcategory'] = disease_name.lower().replace(' ', '_')
                    metadata['label'] = disease_name.replace('_', ' ').title()
                break
                
            elif 'growth' in part_lower or 'stage' in part_lower:
                metadata['category'] = 'growth_stage'
                metadata['source_dataset'] = 'Tomato Growth Stages'
                # Look for stage folder
                for p in parts:
                    if p.lower().startswith('stage'):
                        metadata['subcategory'] = p.lower().replace(' ', '_')
                        metadata['label'] = p.replace('_', ' ').title()
                        break
                break
                
            elif 'healthy' in part_lower:
                metadata['category'] = 'healthy'
                metadata['subcategory'] = 'healthy_leaf'
                metadata['label'] = 'Healthy Leaf'
                metadata['source_dataset'] = 'Tomato Healthy Leaves'
                break
        
        return metadata
    
    def upload_image(
        self,
        file_path: Path,
        category: str,
        subcategory: str,
        metadata: Dict
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Upload a single image to MinIO
        
        Args:
            file_path: Path to image file
            category: Image category
            subcategory: Image subcategory
            metadata: Additional metadata
            
        Returns:
            tuple: (success, image_key, etag)
        """
        if self.dry_run:
            image_key = self.path_builder.build_image_key(
                category, subcategory, file_path.name
            )
            return True, image_key, "dry-run-etag"
        
        try:
            # Build MinIO object key
            image_key = self.path_builder.build_image_key(
                category,
                subcategory,
                file_path.name,
                year=datetime.now().year,
                month=datetime.now().month
            )
            
            # Get bucket name
            bucket_name = self.config.get_bucket_name(category)
            
            # Upload file
            result = self.minio_client.fput_object(
                bucket_name,
                image_key,
                str(file_path),
                content_type=f"image/{file_path.suffix[1:]}"
            )
            
            logger.debug(f"Uploaded: {image_key}")
            return True, image_key, result.etag
            
        except S3Error as e:
            logger.error(f"S3 error uploading {file_path}: {e}")
            return False, None, None
        except Exception as e:
            logger.error(f"Error uploading {file_path}: {e}")
            return False, None, None
    
    def store_metadata(self, image_data: List[Dict]) -> int:
        """
        Store image metadata in PostgreSQL
        
        Args:
            image_data: List of metadata dictionaries
            
        Returns:
            int: Number of records inserted
        """
        if self.dry_run or not image_data:
            return len(image_data)
        
        try:
            cursor = self.db_conn.cursor()
            
            insert_query = """
                INSERT INTO image_metadata (
                    image_key, bucket_name, file_name, file_size, mime_type, etag,
                    category, subcategory, label, crop_type, source_dataset,
                    original_path, width, height, color_space,
                    is_uploaded, upload_date, tags
                ) VALUES (
                    %(image_key)s, %(bucket_name)s, %(file_name)s, %(file_size)s,
                    %(mime_type)s, %(etag)s, %(category)s, %(subcategory)s,
                    %(label)s, %(crop_type)s, %(source_dataset)s, %(original_path)s,
                    %(width)s, %(height)s, %(color_space)s, %(is_uploaded)s,
                    %(upload_date)s, %(tags)s
                )
                ON CONFLICT (image_key) DO UPDATE SET
                    upload_date = EXCLUDED.upload_date,
                    is_uploaded = EXCLUDED.is_uploaded,
                    etag = EXCLUDED.etag,
                    updated_at = CURRENT_TIMESTAMP
            """
            
            execute_batch(cursor, insert_query, image_data, page_size=100)
            self.db_conn.commit()
            
            inserted = cursor.rowcount
            cursor.close()
            
            logger.debug(f"Inserted {inserted} metadata records")
            return inserted
            
        except Exception as e:
            logger.error(f"Error storing metadata: {e}")
            self.db_conn.rollback()
            return 0
    
    def process_directory(
        self,
        directory: Path,
        base_path: Path,
        batch_size: int = 100
    ) -> Dict:
        """
        Process all images in a directory
        
        Args:
            directory: Directory to process
            base_path: Base directory for relative path calculation
            batch_size: Number of images to process before storing metadata
            
        Returns:
            dict: Processing statistics
        """
        logger.info(f"Processing directory: {directory}")
        
        # Find all image files
        image_extensions = ('.jpg', '.jpeg', '.png', '.tiff', '.bmp')
        image_files = [
            f for f in directory.rglob('*')
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        
        logger.info(f"Found {len(image_files)} image files")
        self.stats['total_files'] += len(image_files)
        
        # Process images in batches
        batch_data = []
        
        with tqdm(total=len(image_files), desc=directory.name) as pbar:
            for file_path in image_files:
                try:
                    # Parse metadata from folder structure
                    parsed = self._parse_folder_structure(file_path, base_path)
                    
                    if not parsed['category']:
                        logger.warning(f"Could not determine category for: {file_path}")
                        self.stats['skipped'] += 1
                        pbar.update(1)
                        continue
                    
                    # Get image information
                    img_info = self._get_image_info(file_path)
                    
                    # Upload image
                    success, image_key, etag = self.upload_image(
                        file_path,
                        parsed['category'],
                        parsed['subcategory'],
                        parsed
                    )
                    
                    if success:
                        # Prepare metadata for database
                        metadata = {
                            'image_key': image_key,
                            'bucket_name': self.config.get_bucket_name(parsed['category']),
                            'file_name': file_path.name,
                            'file_size': img_info['size_bytes'],
                            'mime_type': f"image/{file_path.suffix[1:]}",
                            'etag': etag,
                            'category': parsed['category'],
                            'subcategory': parsed['subcategory'],
                            'label': parsed['label'],
                            'crop_type': 'tomato',
                            'source_dataset': parsed['source_dataset'],
                            'original_path': str(file_path),
                            'width': img_info['width'],
                            'height': img_info['height'],
                            'color_space': img_info['mode'],
                            'is_uploaded': True,
                            'upload_date': datetime.now(),
                            'tags': [parsed['category'], parsed['subcategory'], 'tomato']
                        }
                        
                        batch_data.append(metadata)
                        self.stats['uploaded'] += 1
                        self.stats['total_size_bytes'] += img_info['size_bytes']
                        
                        # Store batch when it reaches batch_size
                        if len(batch_data) >= batch_size:
                            self.store_metadata(batch_data)
                            batch_data = []
                    else:
                        self.stats['failed'] += 1
                    
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    self.stats['failed'] += 1
                
                pbar.update(1)
        
        # Store remaining batch
        if batch_data:
            self.store_metadata(batch_data)
        
        return self.stats
    
    def close(self):
        """Close database connection"""
        if self.db_conn:
            self.db_conn.close()
            logger.info("Database connection closed")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Upload agricultural images to MinIO and store metadata in PostgreSQL"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run without actually uploading or storing data'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Number of images to process before storing metadata (default: 100)'
    )
    parser.add_argument(
        '--directories',
        nargs='+',
        help='Specific directories to process (default: all three datasets)'
    )
    
    args = parser.parse_args()
    
    # Create logs directory if it doesn't exist
    Path('logs').mkdir(exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("AgriTwin-GH Image Upload Script")
    logger.info("=" * 80)
    logger.info(f"Dry run: {args.dry_run}")
    logger.info(f"Batch size: {args.batch_size}")
    
    # Initialize configuration
    config = MinIOConfig()
    
    try:
        config.validate()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        logger.error("Please set MinIO credentials in .env file")
        return 1
    
    # Define directories to process
    base_path = Path("data/external")
    
    if args.directories:
        directories = [Path(d) for d in args.directories]
    else:
        directories = [
            base_path / "Tomato Diseases",
            base_path / "Tomato Growth Stages",
            base_path / "Tomato Healthy Leaves"
        ]
    
    # Verify directories exist
    for directory in directories:
        if not directory.exists():
            logger.error(f"Directory not found: {directory}")
            return 1
    
    # Initialize uploader
    uploader = ImageUploader(config, dry_run=args.dry_run)
    
    try:
        # Process each directory
        start_time = datetime.now()
        
        for directory in directories:
            uploader.process_directory(directory, base_path, args.batch_size)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Print statistics
        logger.info("=" * 80)
        logger.info("Upload Complete!")
        logger.info("=" * 80)
        logger.info(f"Total files found: {uploader.stats['total_files']}")
        logger.info(f"Successfully uploaded: {uploader.stats['uploaded']}")
        logger.info(f"Failed: {uploader.stats['failed']}")
        logger.info(f"Skipped: {uploader.stats['skipped']}")
        logger.info(f"Total size: {uploader.stats['total_size_bytes'] / (1024**2):.2f} MB")
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info(f"Average: {uploader.stats['uploaded'] / duration:.2f} images/second")
        logger.info("=" * 80)
        
        # Save summary report
        if not args.dry_run:
            report = {
                'timestamp': datetime.now().isoformat(),
                'statistics': uploader.stats,
                'duration_seconds': duration,
                'directories_processed': [str(d) for d in directories]
            }
            
            report_path = Path('logs') / f"upload_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Report saved to: {report_path}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Upload interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1
    finally:
        uploader.close()


if __name__ == "__main__":
    sys.exit(main())
