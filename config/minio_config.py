"""
MinIO Configuration for AgriTwin-GH
Handles connection to MinIO object storage for agricultural image data
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class MinIOConfig:
    """MinIO connection configuration"""
    
    # Connection settings - these will be loaded from environment in __post_init__
    endpoint: str = None
    access_key: str = None
    secret_key: str = None
    secure: bool = None
    
    # Bucket settings
    default_bucket: str = "agritwin-images"
    
    # Bucket names by category
    buckets: dict = None
    
    # Upload settings
    max_file_size_mb: int = 50
    allowed_extensions: tuple = (".jpg", ".jpeg", ".png", ".tiff", ".bmp")
    
    # Connection settings
    http_client_timeout: int = 300  # seconds
    max_retries: int = 3
    
    def __post_init__(self):
        """Initialize configuration from environment variables"""
        # Load connection settings from environment if not provided
        if self.endpoint is None:
            self.endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
        if self.access_key is None:
            self.access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        if self.secret_key is None:
            self.secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin")
        if self.secure is None:
            self.secure = os.getenv("MINIO_SECURE", "false").lower() == "true"
        
        # Initialize bucket dictionary if not provided
        if self.buckets is None:
            self.buckets = {
                "disease": "agritwin-diseases",
                "growth_stage": "agritwin-growth-stages",
                "healthy": "agritwin-healthy",
                "default": self.default_bucket
            }
    
    def get_bucket_name(self, category: str) -> str:
        """Get bucket name for a given category"""
        return self.buckets.get(category, self.default_bucket)
    
    def validate(self) -> bool:
        """Validate configuration"""
        if not self.endpoint:
            raise ValueError("MinIO endpoint is required")
        if not self.access_key or not self.secret_key:
            raise ValueError("MinIO access key and secret key are required")
        return True


# Path structure for organizing images in buckets
class MinIOPathBuilder:
    """Helper to build consistent MinIO object key paths"""
    
    @staticmethod
    def build_image_key(
        category: str,
        subcategory: str,
        filename: str,
        year: Optional[int] = None,
        month: Optional[int] = None
    ) -> str:
        """
        Build a consistent object key path for images
        
        Args:
            category: Main category (disease, growth_stage, healthy)
            subcategory: Specific classification
            filename: Original filename
            year: Optional year for time-based organization
            month: Optional month for time-based organization
            
        Returns:
            str: Object key path like 'disease/early_blight/2024/01/image.jpg'
        """
        parts = [category, subcategory]
        
        # Add time-based organization if provided
        if year:
            parts.append(str(year))
            if month:
                parts.append(f"{month:02d}")
        
        parts.append(filename)
        
        return "/".join(parts)
    
    @staticmethod
    def parse_image_key(image_key: str) -> dict:
        """
        Parse an image key back into components
        
        Args:
            image_key: Object key like 'disease/early_blight/2024/01/image.jpg'
            
        Returns:
            dict: Parsed components
        """
        parts = image_key.split("/")
        
        result = {
            "category": parts[0] if len(parts) > 0 else None,
            "subcategory": parts[1] if len(parts) > 1 else None,
            "filename": parts[-1] if len(parts) > 0 else None,
            "year": None,
            "month": None
        }
        
        # Try to extract year and month if present
        if len(parts) >= 4 and parts[2].isdigit():
            result["year"] = int(parts[2])
            if len(parts) >= 5 and parts[3].isdigit():
                result["month"] = int(parts[3])
        
        return result


# Bucket policies (for reference, to be applied via MinIO admin)
BUCKET_POLICIES = {
    "read_only": {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"AWS": ["*"]},
                "Action": ["s3:GetObject"],
                "Resource": ["arn:aws:s3:::agritwin-images/*"]
            }
        ]
    },
    "read_write": {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"AWS": ["*"]},
                "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
                "Resource": ["arn:aws:s3:::agritwin-images/*"]
            }
        ]
    }
}


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    # Test configuration
    config = MinIOConfig()
    print("MinIO Configuration:")
    print(f"  Endpoint: {config.endpoint}")
    print(f"  Secure: {config.secure}")
    print(f"  Default Bucket: {config.default_bucket}")
    print(f"  Buckets: {config.buckets}")
    
    # Test path builder
    path_builder = MinIOPathBuilder()
    test_key = path_builder.build_image_key("disease", "early_blight", "test.jpg", 2024, 1)
    print(f"\nTest Image Key: {test_key}")
    print(f"Parsed: {path_builder.parse_image_key(test_key)}")
