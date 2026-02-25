"""
Verify Image Upload Status

This script checks:
1. Total images uploaded to database
2. Images by category and subcategory
3. Upload completion status
4. Size statistics

Usage:
    python scripts/verify_image_upload.py
"""

import os
import sys
from pathlib import Path
from datetime import datetime

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from tabulate import tabulate

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
load_dotenv()


def get_db_connection():
    """Create database connection"""
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        database=os.getenv("DB_NAME", "agritwin_db"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "")
    )


def check_total_images(conn):
    """Check total number of images in database"""
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) as total FROM image_metadata;")
        result = cur.fetchone()
        return result[0]


def check_by_category(conn):
    """Get image counts by category"""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT 
                category,
                COUNT(*) as total_images,
                COUNT(CASE WHEN is_uploaded THEN 1 END) as uploaded_count,
                ROUND(SUM(file_size) / 1024.0 / 1024.0, 2) as total_size_mb,
                MIN(upload_date) as first_upload,
                MAX(upload_date) as last_upload
            FROM image_metadata
            GROUP BY category
            ORDER BY category;
        """)
        return cur.fetchall()


def check_by_subcategory(conn):
    """Get image counts by subcategory"""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT * FROM image_summary
            ORDER BY category, subcategory;
        """)
        return cur.fetchall()


def check_upload_status(conn):
    """Check upload completion status"""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT 
                processing_status,
                COUNT(*) as count,
                ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) as percentage
            FROM image_metadata
            GROUP BY processing_status
            ORDER BY count DESC;
        """)
        return cur.fetchall()


def check_expected_folders(conn):
    """Check if all expected folders were uploaded"""
    expected_folders = {
        'disease': [
            'tomato_early_blight',
            'tomato_late_blight', 
            'tomato_leaf_mold',
            'tomato_powdery_mildew',
            'tomato_septoria_leaf_spot',
            'tomato_spider_mites'
        ],
        'growth_stage': [
            'stage1_seedling',
            'stage2_early_vegetative',
            'stage3_flowering_initiation',
            'stage4_flowering',
            'stage5_unripe',
            'stage6_ripe'
        ],
        'healthy': ['healthy_leaf']
    }
    
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT DISTINCT category, subcategory
            FROM image_metadata
            ORDER BY category, subcategory;
        """)
        uploaded = cur.fetchall()
    
    # Organize uploaded by category
    uploaded_by_cat = {}
    for row in uploaded:
        cat = row['category']
        subcat = row['subcategory']
        if cat not in uploaded_by_cat:
            uploaded_by_cat[cat] = []
        uploaded_by_cat[cat].append(subcat)
    
    results = []
    for category, expected_subcats in expected_folders.items():
        uploaded_subcats = uploaded_by_cat.get(category, [])
        
        for subcat in expected_subcats:
            status = '✓' if subcat in uploaded_subcats else '✗'
            results.append({
                'category': category,
                'subcategory': subcat or '(no subcategory)',
                'status': status
            })
    
    return results


def main():
    """Main verification function"""
    print("=" * 80)
    print("IMAGE UPLOAD VERIFICATION")
    print("=" * 80)
    print()
    
    try:
        conn = get_db_connection()
        print("✓ Connected to database")
        print()
        
        # 1. Total Images
        total = check_total_images(conn)
        print(f"TOTAL IMAGES IN DATABASE: {total:,}")
        print()
        
        # 2. By Category
        print("=" * 80)
        print("IMAGES BY CATEGORY")
        print("=" * 80)
        category_data = check_by_category(conn)
        if category_data:
            headers = ['Category', 'Total', 'Uploaded', 'Size (MB)', 'First Upload', 'Last Upload']
            rows = [
                [
                    row['category'],
                    row['total_images'],
                    row['uploaded_count'],
                    row['total_size_mb'],
                    row['first_upload'].strftime('%Y-%m-%d %H:%M') if row['first_upload'] else 'N/A',
                    row['last_upload'].strftime('%Y-%m-%d %H:%M') if row['last_upload'] else 'N/A'
                ]
                for row in category_data
            ]
            print(tabulate(rows, headers=headers, tablefmt='grid'))
        else:
            print("⚠ No data found")
        print()
        
        # 3. By Subcategory
        print("=" * 80)
        print("IMAGES BY SUBCATEGORY")
        print("=" * 80)
        subcat_data = check_by_subcategory(conn)
        if subcat_data:
            headers = ['Category', 'Subcategory', 'Total', 'Uploaded', 'Processed', 'Size (MB)']
            rows = [
                [
                    row['category'],
                    row['subcategory'] or '(none)',
                    row['total_images'],
                    row['uploaded_count'],
                    row['processed_count'],
                    row['total_size_mb']
                ]
                for row in subcat_data
            ]
            print(tabulate(rows, headers=headers, tablefmt='grid'))
        else:
            print("⚠ No data found")
        print()
        
        # 4. Upload Status
        print("=" * 80)
        print("UPLOAD STATUS")
        print("=" * 80)
        status_data = check_upload_status(conn)
        if status_data:
            headers = ['Status', 'Count', 'Percentage']
            rows = [
                [row['processing_status'], row['count'], f"{row['percentage']}%"]
                for row in status_data
            ]
            print(tabulate(rows, headers=headers, tablefmt='grid'))
        else:
            print("⚠ No data found")
        print()
        
        # 5. Expected Folders Check
        print("=" * 80)
        print("EXPECTED FOLDERS CHECK")
        print("=" * 80)
        folder_results = check_expected_folders(conn)
        headers = ['Category', 'Subcategory', 'Status']
        rows = [
            [r['category'], r['subcategory'], r['status']]
            for r in folder_results
        ]
        print(tabulate(rows, headers=headers, tablefmt='grid'))
        print()
        
        # Summary
        all_uploaded = all(r['status'] == '✓' for r in folder_results)
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        if all_uploaded and total > 0:
            print("✓ All expected folders were uploaded successfully!")
            print(f"✓ Total {total:,} images found in database")
        else:
            print("⚠ Some folders may be missing or no images uploaded")
            missing = [r for r in folder_results if r['status'] == '✗']
            if missing:
                print(f"✗ Missing folders: {len(missing)}")
                for m in missing:
                    print(f"  - {m['category']}/{m['subcategory']}")
        print("=" * 80)
        
        conn.close()
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
