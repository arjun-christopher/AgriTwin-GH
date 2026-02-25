"""
Query and Retrieve Images from MinIO Storage

This script provides utilities to:
1. Query image metadata from PostgreSQL
2. Download images from MinIO
3. Generate reports and visualizations

Usage:
    # List all images by category
    python query_images.py --list --category disease
    
    # Download specific images
    python query_images.py --download --category disease --subcategory tomato_early_blight --limit 10
    
    # Generate statistics report
    python query_images.py --stats
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import json

import psycopg2
from psycopg2.extras import RealDictCursor
from minio import Minio
from minio.error import S3Error
from dotenv import load_dotenv
from tabulate import tabulate
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.minio_config import MinIOConfig

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ImageQuery:
    """Query and retrieve images from MinIO and PostgreSQL"""
    
    def __init__(self):
        """Initialize connections"""
        self.config = MinIOConfig()
        self.minio_client = None
        self.db_conn = None
        
        self._init_connections()
    
    def _init_connections(self):
        """Initialize MinIO and PostgreSQL connections"""
        # MinIO
        try:
            self.minio_client = Minio(
                self.config.endpoint,
                access_key=self.config.access_key,
                secret_key=self.config.secret_key,
                secure=self.config.secure
            )
            logger.info("Connected to MinIO")
        except Exception as e:
            logger.error(f"Failed to connect to MinIO: {e}")
            raise
        
        # PostgreSQL
        try:
            self.db_conn = psycopg2.connect(
                host=os.getenv("DB_HOST", "localhost"),
                port=os.getenv("DB_PORT", "5432"),
                database=os.getenv("DB_NAME", "agritwin_db"),
                user=os.getenv("DB_USER", "postgres"),
                password=os.getenv("DB_PASSWORD", "")
            )
            logger.info("Connected to PostgreSQL")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise
    
    def get_statistics(self) -> pd.DataFrame:
        """Get image statistics by category"""
        query = """
            SELECT * FROM image_summary
            ORDER BY category, subcategory
        """
        
        df = pd.read_sql_query(query, self.db_conn)
        return df
    
    def list_images(
        self,
        category: Optional[str] = None,
        subcategory: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """
        List images with optional filters
        
        Args:
            category: Filter by category
            subcategory: Filter by subcategory
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            List of image metadata dictionaries
        """
        query = """
            SELECT 
                id,
                image_key,
                bucket_name,
                file_name,
                category,
                subcategory,
                label,
                file_size,
                width,
                height,
                upload_date,
                tags
            FROM image_metadata
            WHERE 1=1
        """
        
        params = []
        
        if category:
            query += " AND category = %s"
            params.append(category)
        
        if subcategory:
            query += " AND subcategory = %s"
            params.append(subcategory)
        
        query += " ORDER BY upload_date DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])
        
        cursor = self.db_conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(query, params)
        results = cursor.fetchall()
        cursor.close()
        
        return [dict(row) for row in results]
    
    def search_images(
        self,
        search_term: str,
        search_in: str = 'all',
        limit: int = 100
    ) -> List[Dict]:
        """
        Search images by keyword
        
        Args:
            search_term: Term to search for
            search_in: Where to search ('label', 'filename', 'tags', 'all')
            limit: Maximum results
            
        Returns:
            List of matching images
        """
        if search_in == 'label':
            condition = "label ILIKE %s"
        elif search_in == 'filename':
            condition = "file_name ILIKE %s"
        elif search_in == 'tags':
            condition = "%s = ANY(tags)"
        else:  # all
            condition = """
                (label ILIKE %s OR 
                 file_name ILIKE %s OR 
                 %s = ANY(tags))
            """
        
        query = f"""
            SELECT 
                id, image_key, bucket_name, file_name,
                category, subcategory, label,
                file_size, upload_date
            FROM image_metadata
            WHERE {condition}
            ORDER BY upload_date DESC
            LIMIT %s
        """
        
        cursor = self.db_conn.cursor(cursor_factory=RealDictCursor)
        
        if search_in == 'all':
            search_pattern = f"%{search_term}%"
            cursor.execute(query, (search_pattern, search_pattern, search_term, limit))
        else:
            search_pattern = f"%{search_term}%" if search_in != 'tags' else search_term
            cursor.execute(query, (search_pattern, limit))
        
        results = cursor.fetchall()
        cursor.close()
        
        return [dict(row) for row in results]
    
    def download_images(
        self,
        image_ids: List[int] = None,
        category: Optional[str] = None,
        subcategory: Optional[str] = None,
        limit: int = 10,
        output_dir: str = "downloads"
    ) -> int:
        """
        Download images from MinIO
        
        Args:
            image_ids: Specific image IDs to download
            category: Filter by category
            subcategory: Filter by subcategory
            limit: Maximum number to download
            output_dir: Directory to save images
            
        Returns:
            Number of successfully downloaded images
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get images to download
        if image_ids:
            query = """
                SELECT id, image_key, bucket_name, file_name, category, subcategory
                FROM image_metadata
                WHERE id = ANY(%s)
            """
            cursor = self.db_conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(query, (image_ids,))
            images = cursor.fetchall()
        else:
            images = self.list_images(category, subcategory, limit)
        
        # Download each image
        downloaded = 0
        
        for img in images:
            try:
                # Create subdirectory structure
                img_dir = output_path / img['category'] / img['subcategory']
                img_dir.mkdir(parents=True, exist_ok=True)
                
                # Download file
                output_file = img_dir / img['file_name']
                
                self.minio_client.fget_object(
                    img['bucket_name'],
                    img['image_key'],
                    str(output_file)
                )
                
                logger.info(f"Downloaded: {img['file_name']}")
                downloaded += 1
                
            except S3Error as e:
                logger.error(f"S3 error downloading {img['file_name']}: {e}")
            except Exception as e:
                logger.error(f"Error downloading {img['file_name']}: {e}")
        
        logger.info(f"Downloaded {downloaded}/{len(images)} images to {output_dir}")
        return downloaded
    
    def get_image_url(
        self,
        image_id: int,
        expiry_hours: int = 24
    ) -> Optional[str]:
        """
        Get a presigned URL for an image
        
        Args:
            image_id: Image ID
            expiry_hours: URL expiry time in hours
            
        Returns:
            Presigned URL or None
        """
        query = """
            SELECT image_key, bucket_name, file_name
            FROM image_metadata
            WHERE id = %s
        """
        
        cursor = self.db_conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(query, (image_id,))
        result = cursor.fetchone()
        cursor.close()
        
        if not result:
            logger.error(f"Image ID {image_id} not found")
            return None
        
        try:
            from datetime import timedelta
            url = self.minio_client.presigned_get_object(
                result['bucket_name'],
                result['image_key'],
                expires=timedelta(hours=expiry_hours)
            )
            return url
        except Exception as e:
            logger.error(f"Error generating URL: {e}")
            return None
    
    def close(self):
        """Close connections"""
        if self.db_conn:
            self.db_conn.close()


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Query and retrieve images from MinIO storage"
    )
    
    # Action arguments
    parser.add_argument('--stats', action='store_true', help='Show statistics')
    parser.add_argument('--list', action='store_true', help='List images')
    parser.add_argument('--search', type=str, help='Search for images by keyword')
    parser.add_argument('--download', action='store_true', help='Download images')
    parser.add_argument('--get-url', type=int, help='Get presigned URL for image ID')
    
    # Filter arguments
    parser.add_argument('--category', type=str, help='Filter by category')
    parser.add_argument('--subcategory', type=str, help='Filter by subcategory')
    parser.add_argument('--limit', type=int, default=100, help='Limit results')
    parser.add_argument('--offset', type=int, default=0, help='Offset for pagination')
    
    # Download arguments
    parser.add_argument('--output-dir', type=str, default='downloads', help='Output directory')
    parser.add_argument('--ids', type=int, nargs='+', help='Specific image IDs')
    
    # Search arguments
    parser.add_argument('--search-in', type=str, default='all',
                       choices=['all', 'label', 'filename', 'tags'],
                       help='Where to search')
    
    # Output format
    parser.add_argument('--format', type=str, default='table',
                       choices=['table', 'json', 'csv'],
                       help='Output format')
    
    args = parser.parse_args()
    
    # Initialize query handler
    query = ImageQuery()
    
    try:
        # Execute requested action
        if args.stats:
            logger.info("Fetching statistics...")
            df = query.get_statistics()
            
            if args.format == 'json':
                print(df.to_json(orient='records', indent=2))
            elif args.format == 'csv':
                print(df.to_csv(index=False))
            else:
                print("\n" + "="*80)
                print("IMAGE STATISTICS")
                print("="*80)
                print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))
                print("="*80)
        
        elif args.list:
            logger.info("Listing images...")
            images = query.list_images(
                category=args.category,
                subcategory=args.subcategory,
                limit=args.limit,
                offset=args.offset
            )
            
            if args.format == 'json':
                print(json.dumps(images, indent=2, default=str))
            elif args.format == 'csv':
                df = pd.DataFrame(images)
                print(df.to_csv(index=False))
            else:
                if images:
                    # Convert to DataFrame for pretty printing
                    df = pd.DataFrame(images)
                    # Select key columns
                    display_cols = ['id', 'file_name', 'category', 'subcategory', 
                                   'label', 'file_size', 'upload_date']
                    display_df = df[display_cols]
                    
                    print("\n" + "="*80)
                    print(f"IMAGES ({len(images)} results)")
                    print("="*80)
                    print(tabulate(display_df, headers='keys', tablefmt='grid', showindex=False))
                    print("="*80)
                else:
                    print("No images found")
        
        elif args.search:
            logger.info(f"Searching for: {args.search}")
            images = query.search_images(
                search_term=args.search,
                search_in=args.search_in,
                limit=args.limit
            )
            
            print(f"\nFound {len(images)} matching images")
            
            if images and args.format == 'table':
                df = pd.DataFrame(images)
                print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))
            elif images:
                print(json.dumps(images, indent=2, default=str))
        
        elif args.download:
            logger.info("Downloading images...")
            count = query.download_images(
                image_ids=args.ids,
                category=args.category,
                subcategory=args.subcategory,
                limit=args.limit,
                output_dir=args.output_dir
            )
            print(f"\n✓ Downloaded {count} images to {args.output_dir}")
        
        elif args.get_url:
            logger.info(f"Getting URL for image ID: {args.get_url}")
            url = query.get_image_url(args.get_url)
            if url:
                print(f"\nPresigned URL (valid for 24 hours):")
                print(url)
            else:
                print("Failed to generate URL")
        
        else:
            parser.print_help()
            return 1
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1
    finally:
        query.close()


if __name__ == "__main__":
    sys.exit(main())
