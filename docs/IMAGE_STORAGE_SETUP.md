# Image Storage Setup Guide - MinIO + PostgreSQL

This guide explains how to set up and use the image storage system for AgriTwin-GH using MinIO object storage and PostgreSQL metadata database.

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Database Setup](#database-setup)
6. [Running MinIO](#running-minio)
7. [Uploading Images](#uploading-images)
8. [Querying Images](#querying-images)
9. [Troubleshooting](#troubleshooting)

---

## Overview

The system consists of:
- **MinIO**: S3-compatible object storage for storing image files
- **PostgreSQL**: Relational database for storing image metadata
- **Upload Script**: Python script to process and upload images

### Architecture

```
Image Files (Disk)
        ↓
Upload Script (Python)
        ↓
    ┌───────────────────┐
    │   MinIO           │ ← Image blobs (JPG, PNG, etc.)
    │   (Object Store)  │
    └───────────────────┘
        ↓ (Metadata)
    ┌───────────────────┐
    │   PostgreSQL      │ ← Image metadata, categories, paths
    │   (Database)      │
    └───────────────────┘
```

---

## Prerequisites

- Python 3.10+
- **PostgreSQL 14+ in Docker** (✅ Already set up in your environment)
- MinIO Server (or Docker)
- Git

---

## Installation

### 1. Install Python Dependencies

```powershell
# Activate your virtual environment
.\.venv\Scripts\activate

# Install required packages
uv add -r requirements.txt
```

### 2. Install MinIO Server

**Option A: Using Docker (Recommended)**

```powershell
docker run -d `
  -p 9000:9000 `
  -p 9001:9001 `
  --name minio `
  -v E:\minio-data:/data `
  -e "MINIO_ROOT_USER=minioadmin" `
  -e "MINIO_ROOT_PASSWORD=minioadmin123" `
  quay.io/minio/minio server /data --console-address ":9001"
```

**Option B: Using Binary**

1. Download MinIO from https://min.io/download
2. Run:
   ```powershell
   ./minio.exe server E:\minio-data --console-address ":9001"
   ```

### 3. Access MinIO Console

Open your browser to: http://localhost:9001

- **Username**: `minioadmin`
- **Password**: `minioadmin123` (or whatever you set)

---

## Configuration

### 1. Create Environment File

Copy the example environment file:

```powershell
Copy-Item .env.example .env
```

### 2. Edit `.env` File

Update with your actual credentials:

```env
# Database (TimescaleDB Container - used for both timeseries and image metadata)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=agritwin_db
DB_USER=postgres
DB_PASSWORD=agritwin-gh

# MinIO Configuration
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_SECURE=false
```

---

## Database Setup

> **Note**: Image metadata tables will be added to your existing `agritwin_db` database.
> 
> **🐳 Docker Users**: See [POSTGRES_DOCKER_SETUP.md](POSTGRES_DOCKER_SETUP.md) for complete Docker-specific commands.

### 1. Identify Your PostgreSQL Container

```powershell
# List running containers
docker ps

# Or list all containers
docker ps -a
```

### 2. Run Schema in Existing Database

The image metadata tables will be created in your existing `agritwin_db` database:

**Via Docker (PowerShell):**
```powershell
# Run schema using Get-Content (PowerShell doesn't support < redirection)
Get-Content database/schema/image_metadata.sql | docker exec -i agritwin-timescaledb psql -U postgres -d agritwin_db
```

**If you have psql installed locally:**
```powershell
psql -h localhost -U postgres -d agritwin_db -f database/schema/image_metadata.sql
```

### 3. Verify Tables Created

```powershell
# Via Docker
docker exec -i agritwin-timescaledb psql -U postgres -d agritwin_db -c "\dt"

# You should see your existing tables PLUS:
# image_metadata
# image_annotations
# image_access_log
```

---

## Running MinIO

### Start MinIO Server

**If using Docker:**
```powershell
docker start minio
```

**If using binary:**
```powershell
./minio.exe server E:\minio-data --console-address ":9001"
```

### Verify MinIO is Running

```powershell
# Test connection (from Python)
python -c "from minio import Minio; client = Minio('localhost:9000', 'minioadmin', 'minioadmin123', secure=False); print('Connected:', client.bucket_exists('test') or True)"
```

---

## Uploading Images

### Basic Upload

Upload all three image datasets:

```powershell
python scripts/upload_images_to_minio.py
```

### Dry Run (Test without uploading)

```powershell
python scripts/upload_images_to_minio.py --dry-run
```

### Custom Batch Size

```powershell
# Process 50 images at a time
python scripts/upload_images_to_minio.py --batch-size 50
```

### Upload Specific Directory

```powershell
# Upload only disease images
python scripts/upload_images_to_minio.py --directories "data/external/Tomato Diseases"

# Upload multiple specific directories
python scripts/upload_images_to_minio.py --directories "data/external/Tomato Diseases" "data/external/Tomato Growth Stages"
```

### Monitor Progress

The script shows:
- Progress bar for each directory
- Upload statistics
- Error messages
- Final summary report

Example output:
```
Processing directory: Tomato Diseases
Found 6000 image files
Tomato Diseases: 100%|████████████| 6000/6000 [05:30<00:00, 18.12 images/s]
================================================================================
Upload Complete!
================================================================================
Total files found: 14087
Successfully uploaded: 14087
Failed: 0
Skipped: 0
Total size: 2500.45 MB
Duration: 1234.56 seconds
Average: 11.41 images/second
================================================================================
```

---

## Querying Images

### Using SQL

Connect to PostgreSQL and run queries:

```sql
-- Get image counts by category
SELECT * FROM image_summary;

-- Find all early blight disease images
SELECT image_key, file_name, upload_date
FROM image_metadata
WHERE category = 'disease' 
  AND subcategory = 'tomato_early_blight'
LIMIT 10;

-- Get recently uploaded images
SELECT 
    category,
    subcategory,
    label,
    COUNT(*) as count,
    SUM(file_size) / 1024.0 / 1024.0 as size_mb
FROM image_metadata
WHERE upload_date > NOW() - INTERVAL '1 day'
GROUP BY category, subcategory, label;

-- Search by tags
SELECT image_key, label, tags
FROM image_metadata
WHERE 'early_blight' = ANY(tags);
```

### Using Python

Create a script to query and download images:

```python
import psycopg2
from minio import Minio
from dotenv import load_dotenv
import os

load_dotenv()

# Connect to PostgreSQL
conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    database=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD")
)

# Connect to MinIO
minio_client = Minio(
    os.getenv("MINIO_ENDPOINT"),
    access_key=os.getenv("MINIO_ACCESS_KEY"),
    secret_key=os.getenv("MINIO_SECRET_KEY"),
    secure=False
)

# Query images
cursor = conn.cursor()
cursor.execute("""
    SELECT image_key, bucket_name, file_name, label
    FROM image_metadata
    WHERE category = 'disease'
    LIMIT 5
""")

# Download images
for image_key, bucket_name, file_name, label in cursor.fetchall():
    print(f"Downloading: {label} - {file_name}")
    minio_client.fget_object(
        bucket_name,
        image_key,
        f"downloads/{file_name}"
    )

cursor.close()
conn.close()
```

---

## Folder Structure

After upload, images are organized in MinIO as:

```
Buckets:
├── agritwin-diseases/
│   ├── disease/
│   │   ├── tomato_early_blight/
│   │   │   ├── 2024/
│   │   │   │   ├── 12/
│   │   │   │   │   ├── 003f4fcb-18d6-4853-89a4-51002e3164a9___RS_Erly.B 1683.JPG
│   │   │   │   │   ├── 003f4fcb-18d6-4853-89a4-51002e3164a9___RS_Erly.B 1684.JPG
│   │   ├── tomato_late_blight/
│   │   ├── tomato_leaf_mold/
│   │   └── ...
│
├── agritwin-growth-stages/
│   ├── growth_stage/
│   │   ├── stage1_seedling/
│   │   ├── stage2_early_vegetative/
│   │   └── ...
│
└── agritwin-healthy/
    ├── healthy/
    │   ├── healthy_leaf/
    │   │   ├── 2024/
    │   │   │   └── 12/
    │   │   │       ├── image001.jpg
    │   │   │       └── ...
```

---

## Troubleshooting

### MinIO Connection Error

**Problem**: `Connection refused` or `Network error`

**Solution**:
1. Check MinIO is running: `docker ps` or check the MinIO process
2. Verify port 9000 is accessible
3. Check firewall settings

### PostgreSQL Connection Error

**Problem**: `psycopg2.OperationalError`

**Solution**:
1. Verify Docker container is running: `docker ps | Select-String timescale`
2. Check credentials in `.env` match your container password
3. Test connection: `docker exec -i agritwin-timescaledb psql -U postgres -d agritwin_db -c "SELECT 1;"`

### Schema Not Found

**Problem**: `relation "image_metadata" does not exist`

**Solution**:
```powershell
# Run the schema file using Get-Content (PowerShell syntax)
Get-Content database/schema/image_metadata.sql | docker exec -i agritwin-timescaledb psql -U postgres -d agritwin_db
```

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'minio'`

**Solution**:
```powershell
# Ensure virtual environment is activated
.\.venv\Scripts\activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Upload Fails for Large Files

**Problem**: Timeout or memory errors

**Solution**:
1. Reduce batch size: `--batch-size 10`
2. Increase timeout in `config/minio_config.py`
3. Check available disk space

### Permission Denied

**Problem**: Cannot create buckets or upload files

**Solution**:
1. Check MinIO credentials are correct
2. Verify user has write permissions
3. Check MinIO bucket policies

---

## Advanced Usage

### Custom Bucket Configuration

Edit `config/minio_config.py`:

```python
self.buckets = {
    "disease": "my-custom-disease-bucket",
    "growth_stage": "my-custom-stage-bucket",
    "healthy": "my-custom-healthy-bucket",
    "default": "my-default-bucket"
}
```

### Adding Custom Metadata

Modify the upload script to extract additional metadata from image EXIF or filename patterns.

### Batch Processing

For very large datasets:

```powershell
# Process one category at a time
python scripts/upload_images_to_minio.py --directories "data/external/Tomato Diseases"
python scripts/upload_images_to_minio.py --directories "data/external/Tomato Growth Stages"
python scripts/upload_images_to_minio.py --directories "data/external/Tomato Healthy Leaves"
```

### Resume Failed Upload

The script uses `ON CONFLICT ... DO UPDATE` so you can safely re-run it to resume failed uploads.

---

## Next Steps

1. **Image Retrieval API**: Create REST API to query and retrieve images
2. **ML Integration**: Add model inference results to `analysis_results` column
3. **Image Processing Pipeline**: Add preprocessing, augmentation, and quality checks
4. **Monitoring Dashboard**: Visualize upload statistics and storage usage
5. **Backup Strategy**: Set up automated backups for MinIO and PostgreSQL

---

## Resources

- [MinIO Python SDK Docs](https://min.io/docs/minio/linux/developers/python/minio-py.html)
- [PostgreSQL Docs](https://www.postgresql.org/docs/)
- [psycopg2 Documentation](https://www.psycopg.org/docs/)

---

## Support

For issues or questions:
1. Check the logs in `logs/image_upload.log`
2. Review the upload report JSON files in `logs/`
3. Consult PostgreSQL logs for database errors
4. Check MinIO console for storage issues

---

*Last Updated: 2024*
