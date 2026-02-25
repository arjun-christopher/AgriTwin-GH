"""
Verify Image Storage System Setup

This script checks that all components are properly configured:
- MinIO connection
- PostgreSQL connection
- Database schema
- Image directories

Run this before uploading images to ensure everything is set up correctly.

Usage:
    python scripts/verify_setup.py
"""

import os
import sys
from pathlib import Path
from typing import Tuple, List

from dotenv import load_dotenv

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables
load_dotenv()

# ANSI color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


def print_header(text: str):
    """Print section header"""
    print(f"\n{BLUE}{'='*80}")
    print(f"{text}")
    print(f"{'='*80}{RESET}\n")


def print_check(name: str, status: bool, message: str = ""):
    """Print check result"""
    symbol = f"{GREEN}✓{RESET}" if status else f"{RED}✗{RESET}"
    print(f"{symbol} {name}", end="")
    if message:
        color = GREEN if status else RED
        print(f" - {color}{message}{RESET}")
    else:
        print()


def check_environment_variables() -> Tuple[bool, List[str]]:
    """Check if required environment variables are set"""
    print_header("1. ENVIRONMENT VARIABLES")
    
    required_vars = {
        'MinIO': [
            'MINIO_ENDPOINT',
            'MINIO_ACCESS_KEY',
            'MINIO_SECRET_KEY'
        ],
        'Database (PostgreSQL/TimescaleDB)': [
            'DB_HOST',
            'DB_PORT',
            'DB_NAME',
            'DB_USER',
            'DB_PASSWORD'
        ]
    }
    
    all_ok = True
    missing = []
    
    for category, vars in required_vars.items():
        print(f"\n{category}:")
        for var in vars:
            value = os.getenv(var)
            if value:
                # Mask sensitive values
                display_value = value if var not in ['MINIO_SECRET_KEY', 'DB_PASSWORD'] else '***'
                print_check(f"  {var}", True, f"{display_value}")
            else:
                print_check(f"  {var}", False, "NOT SET")
                all_ok = False
                missing.append(var)
    
    if not all_ok:
        print(f"\n{YELLOW}⚠ Missing variables. Set them in .env file{RESET}")
    
    return all_ok, missing


def check_minio_connection() -> bool:
    """Check MinIO connection and buckets"""
    print_header("2. MINIO CONNECTION")
    
    try:
        from minio import Minio
        from config.minio_config import MinIOConfig
        
        config = MinIOConfig()
        
        print(f"Connecting to: {config.endpoint}")
        client = Minio(
            config.endpoint,
            access_key=config.access_key,
            secret_key=config.secret_key,
            secure=config.secure
        )
        
        print_check("Connection", True, "Successfully connected")
        
        # Check buckets
        print("\nBucket Status:")
        all_buckets_ok = True
        for bucket_name in config.buckets.values():
            exists = client.bucket_exists(bucket_name)
            print_check(f"  {bucket_name}", exists,
                       "exists" if exists else "will be created during upload")
            if not exists:
                all_buckets_ok = False
        
        if not all_buckets_ok:
            print(f"\n{YELLOW}ℹ Buckets will be created automatically during first upload{RESET}")
        
        return True
        
    except ImportError as e:
        print_check("MinIO SDK", False, f"Import error: {e}")
        print(f"\n{YELLOW}Run: pip install minio{RESET}")
        return False
    except Exception as e:
        print_check("Connection", False, str(e))
        print(f"\n{YELLOW}Make sure MinIO server is running{RESET}")
        print(f"{YELLOW}Docker: docker start minio{RESET}")
        print(f"{YELLOW}Or: Start MinIO server manually{RESET}")
        return False


def check_postgresql_connection() -> bool:
    """Check PostgreSQL connection"""
    print_header("3. POSTGRESQL CONNECTION")
    
    try:
        import psycopg2
        
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD")
        )
        
        print_check("Connection", True, "Successfully connected")
        
        # Check schema
        cursor = conn.cursor()
        
        print("\nDatabase Schema:")
        
        # Check tables
        tables = ['image_metadata', 'image_annotations', 'image_access_log']
        tables_ok = True
        
        for table in tables:
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = %s
                )
            """, (table,))
            
            exists = cursor.fetchone()[0]
            is_required = table == 'image_metadata'
            
            if is_required:
                print_check(f"  {table} (required)", exists)
                if not exists:
                    tables_ok = False
            else:
                print_check(f"  {table} (optional)", exists)
        
        # Check view
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.views 
                WHERE table_name = 'image_summary'
            )
        """)
        view_exists = cursor.fetchone()[0]
        print_check("  image_summary (view)", view_exists)
        
        cursor.close()
        conn.close()
        
        if not tables_ok:
            print(f"\n{YELLOW}⚠ Required tables missing!{RESET}")
            print(f"{YELLOW}Run: psql -U {os.getenv('POSTGRES_USER')} -d {os.getenv('POSTGRES_DB')} -f database/schema/image_metadata.sql{RESET}")
            return False
        
        return True
        
    except ImportError as e:
        print_check("psycopg2", False, f"Import error: {e}")
        print(f"\n{YELLOW}Run: pip install psycopg2-binary{RESET}")
        return False
    except Exception as e:
        print_check("Connection", False, str(e))
        print(f"\n{YELLOW}Make sure PostgreSQL is running (Docker or local){RESET}")
        print(f"{YELLOW}If using Docker: docker ps | Select-String postgres{RESET}")
        print(f"{YELLOW}Check credentials in .env file{RESET}")
        print(f"{YELLOW}Test: docker exec -it <container> psql -U postgres -l{RESET}")
        return False


def check_image_directories() -> Tuple[bool, dict]:
    """Check if image directories exist"""
    print_header("4. IMAGE DIRECTORIES")
    
    base_path = Path("data/external")
    
    directories = {
        "Tomato Diseases": base_path / "Tomato Diseases",
        "Tomato Growth Stages": base_path / "Tomato Growth Stages",
        "Tomato Healthy Leaves": base_path / "Tomato Healthy Leaves"
    }
    
    all_ok = True
    stats = {}
    
    for name, path in directories.items():
        exists = path.exists()
        print_check(f"{name}", exists, str(path) if exists else "NOT FOUND")
        
        if exists:
            # Count images
            image_files = list(path.rglob('*.jpg')) + list(path.rglob('*.jpeg')) + \
                         list(path.rglob('*.png')) + list(path.rglob('*.JPG'))
            count = len(image_files)
            
            # Calculate total size
            total_size = sum(f.stat().st_size for f in image_files) / (1024 * 1024)
            
            print(f"  └─ {count:,} images, {total_size:.1f} MB")
            stats[name] = {'count': count, 'size_mb': total_size}
        else:
            all_ok = False
            stats[name] = {'count': 0, 'size_mb': 0}
    
    if not all_ok:
        print(f"\n{YELLOW}⚠ Some directories not found{RESET}")
        print(f"{YELLOW}Make sure image datasets are in data/external/{RESET}")
    else:
        total_images = sum(s['count'] for s in stats.values())
        total_size = sum(s['size_mb'] for s in stats.values())
        print(f"\n{GREEN}Total: {total_images:,} images, {total_size:.1f} MB{RESET}")
    
    return all_ok, stats


def check_python_packages() -> bool:
    """Check if required Python packages are installed"""
    print_header("5. PYTHON PACKAGES")
    
    required_packages = [
        ('minio', 'MinIO SDK'),
        ('psycopg2', 'PostgreSQL adapter'),
        ('PIL', 'Pillow image library'),
        ('dotenv', 'python-dotenv'),
        ('tqdm', 'Progress bars'),
        ('pandas', 'Data processing'),
        ('tabulate', 'Table formatting')
    ]
    
    all_ok = True
    
    for package, description in required_packages:
        try:
            __import__(package)
            print_check(f"{description} ({package})", True)
        except ImportError:
            print_check(f"{description} ({package})", False, "NOT INSTALLED")
            all_ok = False
    
    if not all_ok:
        print(f"\n{YELLOW}⚠ Missing packages{RESET}")
        print(f"{YELLOW}Run: pip install -r requirements.txt{RESET}")
    
    return all_ok


def check_logs_directory() -> bool:
    """Check if logs directory exists"""
    print_header("6. LOGS DIRECTORY")
    
    logs_dir = Path("logs")
    
    if not logs_dir.exists():
        logs_dir.mkdir()
        print_check("logs/", True, "Created")
    else:
        print_check("logs/", True, "Exists")
    
    # Check if writable
    test_file = logs_dir / ".test"
    try:
        test_file.touch()
        test_file.unlink()
        print_check("Write permission", True)
        return True
    except Exception as e:
        print_check("Write permission", False, str(e))
        return False


def main():
    """Run all verification checks"""
    print(f"\n{BLUE}{'='*80}")
    print("AGRITWIN-GH IMAGE STORAGE SYSTEM - SETUP VERIFICATION")
    print(f"{'='*80}{RESET}")
    
    # Run checks
    checks = {
        "Environment Variables": check_environment_variables()[0],
        "MinIO Connection": check_minio_connection(),
        "PostgreSQL Connection": check_postgresql_connection(),
        "Image Directories": check_image_directories()[0],
        "Python Packages": check_python_packages(),
        "Logs Directory": check_logs_directory()
    }
    
    # Summary
    print_header("SUMMARY")
    
    passed = sum(1 for v in checks.values() if v)
    total = len(checks)
    
    for name, status in checks.items():
        print_check(name, status)
    
    print(f"\n{BLUE}{'='*80}{RESET}")
    
    if passed == total:
        print(f"{GREEN}✓ All checks passed! ({passed}/{total}){RESET}")
        print(f"\n{GREEN}You're ready to upload images!{RESET}")
        print(f"\nRun: {BLUE}python scripts/upload_images_to_minio.py{RESET}")
    else:
        print(f"{RED}✗ {total - passed} check(s) failed ({passed}/{total} passed){RESET}")
        print(f"\n{YELLOW}Fix the issues above before uploading images{RESET}")
        print(f"\nSee: {BLUE}QUICKSTART_IMAGE_STORAGE.md{RESET} for setup instructions")
    
    print(f"{BLUE}{'='*80}{RESET}\n")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
