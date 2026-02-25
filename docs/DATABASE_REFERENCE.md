# AgriTwin-GH Database Reference

**Complete database schema and usage guide for AgriTwin-GH project**

---

## Table of Contents

- [Overview](#overview)
- [Database Architecture](#database-architecture)
- [Timeseries Tables](#timeseries-tables)
  - [weather_data](#weather_data)
  - [greenhouse_data](#greenhouse_data)
- [Image Metadata Tables](#image-metadata-tables)
  - [image_metadata](#image_metadata)
  - [image_annotations](#image_annotations)
  - [image_access_log](#image_access_log)
- [Views](#views)
- [Indexes and Performance](#indexes-and-performance)
- [Query Patterns](#query-patterns)
- [Data Relationships](#data-relationships)
- [JSON Fields](#json-fields)
- [Best Practices](#best-practices)

---

## Overview

AgriTwin-GH uses a single **PostgreSQL 15 database** (`agritwin_db`) with **TimescaleDB extension** to store two distinct types of agricultural data:

1. **Timeseries Data**: Weather and greenhouse sensor readings
2. **Image Metadata**: Agricultural image metadata with references to MinIO object storage

**Database**: `agritwin_db`  
**Engine**: PostgreSQL 15 with TimescaleDB extension  
**Container**: `agritwin-timescaledb`  
**Port**: 5432

---

## Database Architecture

```
┌─────────────────────────────────────────────────────────┐
│              PostgreSQL 15 (agritwin_db)                │
│                  + TimescaleDB Extension                 │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌────────────────────┐   ┌──────────────────────────┐ │
│  │  TIMESERIES DATA   │   │  IMAGE METADATA          │ │
│  │                    │   │                          │ │
│  │  • weather_data    │   │  • image_metadata        │ │
│  │  • greenhouse_data │   │  • image_annotations     │ │
│  │                    │   │  • image_access_log      │ │
│  │  (Hypertables)     │   │                          │ │
│  └────────────────────┘   └──────────────────────────┘ │
│                                                          │
└─────────────────────────────────────────────────────────┘
                            ↓
                    ┌───────────────┐
                    │  MinIO Storage│
                    │  (Images)     │
                    └───────────────┘
```

**Design Rationale**:
- **Single Database**: Simplifies connection management and allows cross-domain queries
- **TimescaleDB Hypertables**: Optimized for time-series queries with automatic partitioning
- **Separate Storage**: Images stored in MinIO (S3-compatible), metadata in PostgreSQL
- **Flexible JSON**: JSONB fields for ML results and annotations

---

## Timeseries Tables

### weather_data

Stores outdoor weather conditions from external weather API.

**Schema**:
```sql
CREATE TABLE weather_data (
    id INTEGER PRIMARY KEY,
    datetime TIMESTAMP NOT NULL,
    datetime_epoch INTEGER,
    temp FLOAT,                    -- Temperature (°C)
    humidity FLOAT,                -- Humidity (%)
    windspeed FLOAT,               -- Wind speed (km/h)
    solarradiation FLOAT,          -- Solar radiation (W/m²)
    sunrise VARCHAR(20),
    sunrise_epoch FLOAT,
    sunset VARCHAR(20),
    sunset_epoch FLOAT,
    conditions VARCHAR(100),
    description VARCHAR(500)
);

-- Converted to TimescaleDB hypertable
SELECT create_hypertable('weather_data', 'datetime', 
    chunk_time_interval => INTERVAL '1 month');
```

**Key Features**:
- **Timegrained**: Hourly weather observations
- **Indexed** on `datetime` for fast time-range queries
- **Hypertable**: Automatic partitioning by month
- **Source**: Visual Crossing Weather API

**Sample Data**:
```sql
SELECT * FROM weather_data LIMIT 1;
```
| id | datetime | temp | humidity | windspeed | solarradiation | conditions |
|----|----------|------|----------|-----------|----------------|------------|
| 1 | 2024-01-01 00:00:00 | 22.5 | 65.3 | 12.8 | 0.0 | Clear |

**Common Queries**:
```sql
-- Daily temperature extremes
SELECT 
    DATE(datetime) as date,
    MIN(temp) as min_temp,
    MAX(temp) as max_temp,
    AVG(temp) as avg_temp
FROM weather_data
WHERE datetime >= '2025-01-01'
GROUP BY DATE(datetime)
ORDER BY date;

-- High solar radiation days
SELECT DATE(datetime), MAX(solarradiation) as max_radiation
FROM weather_data
WHERE solarradiation > 500
GROUP BY DATE(datetime)
ORDER BY max_radiation DESC;
```

---

### greenhouse_data

Stores indoor greenhouse environmental conditions (synthetic or sensor data).

**Schema**:
```sql
CREATE TABLE greenhouse_data (
    id INTEGER PRIMARY KEY,
    datetime TIMESTAMP NOT NULL,
    indoor_temp FLOAT,             -- Indoor temperature (°C)
    indoor_humidity FLOAT,         -- Indoor humidity (%)
    indoor_air_velocity FLOAT,     -- Air velocity (m/s)
    indoor_co2 FLOAT,              -- CO2 concentration (ppm)
    solarradiation FLOAT,          -- Solar radiation (W/m²)
    day_night_flag INTEGER,        -- Day (1) or Night (0)
    vpd FLOAT,                     -- Vapor Pressure Deficit (kPa)
    dew_point FLOAT,               -- Dew point (°C)
    leaf_wetness_proxy FLOAT       -- Leaf wetness indicator
);

-- Converted to TimescaleDB hypertable
SELECT create_hypertable('greenhouse_data', 'datetime',
    chunk_time_interval => INTERVAL '1 week');
```

**Key Features**:
- **High Frequency**: 5-minute or hourly intervals
- **Hypertable**: Weekly chunks for optimal query performance
- **Derived Metrics**: VPD, dew point, leaf wetness calculated from base sensors
- **Context-aware**: `day_night_flag` for day/night cycle analysis

**Important Metrics**:
- **VPD (Vapor Pressure Deficit)**: Critical for plant transpiration (optimal: 0.8-1.2 kPa)
- **CO2**: Photosynthesis efficiency (optimal: 800-1200 ppm)
- **Leaf Wetness**: Disease risk indicator (high wetness = high disease risk)

**Sample Data**:
```sql
SELECT * FROM greenhouse_data ORDER BY datetime DESC LIMIT 1;
```
| datetime | indoor_temp | indoor_humidity | indoor_co2 | vpd | day_night_flag |
|----------|-------------|-----------------|------------|-----|----------------|
| 2025-01-15 14:30:00 | 24.5 | 68.2 | 950.0 | 0.95 | 1 |

**Common Queries**:
```sql
-- Compare indoor vs outdoor conditions
SELECT 
    g.datetime,
    g.indoor_temp,
    w.temp as outdoor_temp,
    (g.indoor_temp - w.temp) as temp_difference
FROM greenhouse_data g
JOIN weather_data w ON DATE_TRUNC('hour', g.datetime) = DATE_TRUNC('hour', w.datetime)
WHERE g.datetime >= NOW() - INTERVAL '7 days'
ORDER BY g.datetime;

-- High disease risk periods (high humidity + leaf wetness)
SELECT 
    DATE_TRUNC('hour', datetime) as hour,
    AVG(indoor_humidity) as avg_humidity,
    AVG(leaf_wetness_proxy) as avg_wetness
FROM greenhouse_data
WHERE indoor_humidity > 80 AND leaf_wetness_proxy > 0.7
GROUP BY hour
ORDER BY hour DESC;

-- CO2 supplementation effectiveness
SELECT 
    day_night_flag,
    AVG(indoor_co2) as avg_co2,
    MIN(indoor_co2) as min_co2,
    MAX(indoor_co2) as max_co2
FROM greenhouse_data
WHERE datetime >= CURRENT_DATE
GROUP BY day_night_flag;
```

---

## Image Metadata Tables

### image_metadata

Main table storing metadata for agricultural images stored in MinIO.

**Schema**:
```sql
CREATE TABLE image_metadata (
    id SERIAL PRIMARY KEY,
    
    -- MinIO Storage Info
    image_key VARCHAR(255) NOT NULL UNIQUE,
    bucket_name VARCHAR(100) NOT NULL DEFAULT 'agritwin-images',
    file_name VARCHAR(255) NOT NULL,
    file_size BIGINT,
    mime_type VARCHAR(50) DEFAULT 'image/jpeg',
    etag VARCHAR(255),
    
    -- Image Classification
    category VARCHAR(50) NOT NULL,        -- 'disease', 'growth_stage', 'healthy'
    subcategory VARCHAR(100),             -- 'early_blight', 'stage1_seedling', etc.
    label VARCHAR(100),
    
    -- Agricultural Context
    crop_type VARCHAR(50) DEFAULT 'tomato',
    source_dataset VARCHAR(100),
    original_path TEXT,
    
    -- Image Properties
    width INTEGER,
    height INTEGER,
    color_space VARCHAR(20),
    
    -- Processing Status
    is_uploaded BOOLEAN DEFAULT FALSE,
    upload_date TIMESTAMP,
    is_processed BOOLEAN DEFAULT FALSE,
    processed_date TIMESTAMP,
    processing_status VARCHAR(50) DEFAULT 'pending',
    
    -- Analysis Results (JSON)
    analysis_results JSONB,
    
    -- Metadata
    tags TEXT[],
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100),
    
    CONSTRAINT valid_category CHECK (category IN ('disease', 'growth_stage', 'healthy')),
    CONSTRAINT valid_status CHECK (processing_status IN ('pending', 'processing', 'completed', 'failed', 'skipped'))
);
```

**Key Features**:
- **MinIO Reference**: `image_key` points to object in MinIO bucket
- **Multi-category**: Supports disease, growth stage, and healthy leaf images
- **Flexible JSON**: `analysis_results` stores ML model outputs
- **Array Tags**: PostgreSQL array for efficient tag filtering
- **Auto-timestamp**: `updated_at` automatically updated on changes

**Dataset Distribution** (69,607 total images):
- **Diseases**: 23,003 images (348 MB) - 6 disease types
- **Growth Stages**: 43,422 images (11.4 GB) - 6 growth stages
- **Healthy Leaves**: 3,182 images (67 MB)

**Sample Data**:
```sql
SELECT id, file_name, category, subcategory, file_size, width, height 
FROM image_metadata LIMIT 3;
```
| id | file_name | category | subcategory | file_size | width | height |
|----|-----------|----------|-------------|-----------|-------|--------|
| 1 | IMG_001.jpg | disease | tomato_early_blight | 245120 | 1024 | 768 |
| 2 | STAGE2_045.jpg | growth_stage | stage2_early_vegetative | 512340 | 2048 | 1536 |
| 3 | HEALTHY_089.jpg | healthy | NULL | 189760 | 800 | 600 |

**Common Queries**:
```sql
-- Images by category
SELECT category, COUNT(*) as count, SUM(file_size) as total_bytes
FROM image_metadata
GROUP BY category;

-- Recently uploaded disease images
SELECT file_name, subcategory, upload_date
FROM image_metadata
WHERE category = 'disease'
ORDER BY upload_date DESC
LIMIT 10;

-- Search by tags
SELECT file_name, tags
FROM image_metadata
WHERE tags @> ARRAY['high_quality', 'annotated']
LIMIT 20;

-- Unprocessed images
SELECT COUNT(*) as pending_count
FROM image_metadata
WHERE processing_status = 'pending';

-- ML analysis results (JSON query)
SELECT file_name, analysis_results->>'disease' as predicted_disease,
       (analysis_results->>'confidence')::float as confidence
FROM image_metadata
WHERE analysis_results IS NOT NULL
  AND (analysis_results->>'confidence')::float > 0.90
ORDER BY confidence DESC;
```

---

### image_annotations

Stores ML training labels and bounding boxes for images.

**Schema**:
```sql
CREATE TABLE image_annotations (
    id SERIAL PRIMARY KEY,
    image_id INTEGER NOT NULL REFERENCES image_metadata(id) ON DELETE CASCADE,
    annotation_type VARCHAR(50) NOT NULL,     -- 'bounding_box', 'segmentation', 'classification', 'keypoint'
    annotation_data JSONB NOT NULL,
    confidence FLOAT,
    annotator VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT valid_annotation_type CHECK (annotation_type IN 
        ('bounding_box', 'segmentation', 'classification', 'keypoint'))
);
```

**Annotation Data Examples**:

**Bounding Box** (disease detection):
```json
{
  "boxes": [
    {"x": 120, "y": 80, "width": 200, "height": 150, "label": "early_blight", "confidence": 0.92}
  ],
  "image_width": 1024,
  "image_height": 768
}
```

**Classification** (growth stage):
```json
{
  "predicted_class": "stage4_flowering",
  "probabilities": {
    "stage3_flowering_initiation": 0.05,
    "stage4_flowering": 0.89,
    "stage5_unripe": 0.06
  }
}
```

**Segmentation** (leaf mask):
```json
{
  "mask_url": "s3://agritwin-masks/mask_12345.png",
  "num_pixels": 45678,
  "mask_type": "binary"
}
```

**Common Queries**:
```sql
-- Images with bounding box annotations
SELECT im.file_name, ia.annotation_data
FROM image_metadata im
JOIN image_annotations ia ON im.id = ia.image_id
WHERE ia.annotation_type = 'bounding_box'
LIMIT 5;

-- High-confidence annotations
SELECT im.file_name, ia.annotation_type, ia.confidence
FROM image_metadata im
JOIN image_annotations ia ON im.id = ia.image_id
WHERE ia.confidence > 0.95
ORDER BY ia.confidence DESC;

-- Annotations per image
SELECT im.file_name, COUNT(ia.id) as annotation_count
FROM image_metadata im
LEFT JOIN image_annotations ia ON im.id = ia.image_id
GROUP BY im.file_name
HAVING COUNT(ia.id) > 0
ORDER BY annotation_count DESC;
```

---

### image_access_log

Tracks image access for usage analytics and audit trails.

**Schema**:
```sql
CREATE TABLE image_access_log (
    id SERIAL PRIMARY KEY,
    image_id INTEGER NOT NULL REFERENCES image_metadata(id) ON DELETE CASCADE,
    accessed_by VARCHAR(100),
    access_type VARCHAR(50),       -- 'view', 'download', 'process', 'train'
    accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);
```

**Use Cases**:
- Track which images are frequently accessed
- Audit trail for model training datasets
- Analytics on image usage patterns
- Identify popular images for caching

**Sample Log Entry**:
```sql
INSERT INTO image_access_log (image_id, accessed_by, access_type, metadata)
VALUES (
    12345,
    'ml_pipeline_v2',
    'train',
    '{"epoch": 45, "batch_id": 102, "model": "resnet50"}'::jsonb
);
```

**Common Queries**:
```sql
-- Most accessed images
SELECT im.file_name, COUNT(*) as access_count
FROM image_access_log ial
JOIN image_metadata im ON ial.image_id = im.id
GROUP BY im.file_name
ORDER BY access_count DESC
LIMIT 20;

-- Access patterns by type
SELECT access_type, COUNT(*) as count
FROM image_access_log
WHERE accessed_at >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY access_type;

-- Images used for training
SELECT DISTINCT im.file_name, im.category, im.subcategory
FROM image_access_log ial
JOIN image_metadata im ON ial.image_id = im.id
WHERE ial.access_type = 'train'
  AND ial.accessed_at >= '2025-01-01';
```

---

## Views

### image_summary

Pre-computed aggregations for quick statistics.

**Definition**:
```sql
CREATE VIEW image_summary AS
SELECT 
    category,
    subcategory,
    COUNT(*) as total_images,
    SUM(file_size) as total_size_bytes,
    ROUND(SUM(file_size) / 1024.0 / 1024.0, 2) as total_size_mb,
    COUNT(CASE WHEN is_uploaded THEN 1 END) as uploaded_count,
    COUNT(CASE WHEN is_processed THEN 1 END) as processed_count,
    MIN(upload_date) as first_upload,
    MAX(upload_date) as last_upload
FROM image_metadata
GROUP BY category, subcategory;
```

**Usage**:
```sql
-- Get statistics for all categories
SELECT * FROM image_summary ORDER BY category, subcategory;

-- Disease images summary
SELECT * FROM image_summary WHERE category = 'disease';
```

**Sample Output**:
| category | subcategory | total_images | total_size_mb | uploaded_count | processed_count |
|----------|-------------|--------------|---------------|----------------|-----------------|
| disease | tomato_early_blight | 3,828 | 58.12 | 3,828 | 3,205 |
| disease | tomato_late_blight | 4,105 | 62.43 | 4,105 | 3,891 |
| growth_stage | stage1_seedling | 7,234 | 1905.23 | 7,234 | 6,892 |

---

## Indexes and Performance

### Timeseries Indexes

```sql
-- weather_data
CREATE INDEX idx_weather_datetime ON weather_data(datetime);

-- greenhouse_data
CREATE INDEX idx_greenhouse_datetime ON greenhouse_data(datetime);
```

**TimescaleDB Advantages**:
- Automatic chunk-wise indexing
- Time-based partitioning (monthly for weather, weekly for greenhouse)
- Optimized for time-range queries
- Continuous aggregates for pre-computed rollups

### Image Metadata Indexes

```sql
-- Primary lookup
CREATE INDEX idx_image_bucket_key ON image_metadata(bucket_name, image_key);

-- Category filtering
CREATE INDEX idx_image_category ON image_metadata(category);
CREATE INDEX idx_image_subcategory ON image_metadata(subcategory);
CREATE INDEX idx_image_crop_type ON image_metadata(crop_type);

-- Time-based queries
CREATE INDEX idx_image_upload_date ON image_metadata(upload_date DESC);

-- Status filtering
CREATE INDEX idx_image_processing_status ON image_metadata(processing_status);

-- Array and JSON (GIN indexes)
CREATE INDEX idx_image_tags ON image_metadata USING GIN(tags);
CREATE INDEX idx_image_analysis ON image_metadata USING GIN(analysis_results);
```

**Performance Tips**:
- **GIN indexes**: Excellent for JSON and array searches, but larger and slower to update
- **BTREE indexes**: Default for scalar fields, fast for equality and range queries
- **Partial indexes**: Consider for frequently filtered subsets (e.g., `WHERE is_processed = false`)

---

## Query Patterns

### Cross-Domain Queries

**Link growth stages to environmental conditions**:
```sql
SELECT 
    im.subcategory as growth_stage,
    COUNT(*) as image_count,
    AVG(g.indoor_temp) as avg_temp,
    AVG(g.vpd) as avg_vpd
FROM image_metadata im
CROSS JOIN greenhouse_data g
WHERE im.category = 'growth_stage'
  AND g.datetime BETWEEN im.upload_date - INTERVAL '7 days' AND im.upload_date
GROUP BY im.subcategory;
```

**Disease occurrence vs weather conditions**:
```sql
SELECT 
    im.subcategory as disease,
    DATE(im.upload_date) as date,
    AVG(w.humidity) as avg_humidity,
    AVG(w.temp) as avg_temp,
    COUNT(*) as image_count
FROM image_metadata im
JOIN weather_data w ON DATE(w.datetime) = DATE(im.upload_date)
WHERE im.category = 'disease'
GROUP BY disease, date
HAVING AVG(w.humidity) > 70
ORDER BY date DESC;
```

### Time-Series Analytics

**Hourly greenhouse trends**:
```sql
SELECT 
    time_bucket('1 hour', datetime) as hour,
    AVG(indoor_temp) as avg_temp,
    AVG(indoor_humidity) as avg_humidity,
    AVG(indoor_co2) as avg_co2
FROM greenhouse_data
WHERE datetime >= NOW() - INTERVAL '24 hours'
GROUP BY hour
ORDER BY hour;
```

**Daily temperature variance**:
```sql
SELECT 
    DATE(datetime) as date,
    MAX(temp) - MIN(temp) as temp_range,
    STDDEV(temp) as temp_stddev
FROM weather_data
WHERE datetime >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(datetime)
ORDER BY temp_range DESC;
```

### Image Analytics

**Processing pipeline status**:
```sql
SELECT 
    processing_status,
    category,
    COUNT(*) as count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY category), 2) as percentage
FROM image_metadata
GROUP BY processing_status, category
ORDER BY category, count DESC;
```

**Storage usage by category**:
```sql
SELECT 
    category,
    COUNT(*) as file_count,
    pg_size_pretty(SUM(file_size)::bigint) as total_size,
    pg_size_pretty(AVG(file_size)::bigint) as avg_file_size
FROM image_metadata
GROUP BY category;
```

---

## Data Relationships

### Entity Relationship Diagram

```
┌─────────────────┐
│ weather_data    │
│ (Hypertable)    │
│                 │
│ • datetime (PK) │
│ • temp          │
│ • humidity      │
└─────────────────┘

┌──────────────────┐
│ greenhouse_data  │
│ (Hypertable)     │
│                  │
│ • datetime (PK)  │
│ • indoor_temp    │
│ • indoor_co2     │
│ • vpd            │
└──────────────────┘

┌─────────────────────┐          ┌──────────────────────┐
│ image_metadata      │          │ image_annotations    │
│                     │ 1      ∞ │                      │
│ • id (PK)          │◄─────────│ • id (PK)            │
│ • image_key (UK)    │          │ • image_id (FK)      │
│ • category          │          │ • annotation_data    │
│ • subcategory       │          └──────────────────────┘
│ • analysis_results  │
│                     │          ┌──────────────────────┐
│                     │ 1      ∞ │ image_access_log     │
│                     │◄─────────│                      │
└─────────────────────┘          │ • id (PK)            │
                                 │ • image_id (FK)      │
                                 │ • access_type        │
                                 └──────────────────────┘
```

### Referential Integrity

- **image_annotations.image_id** → **image_metadata.id** (CASCADE DELETE)
- **image_access_log.image_id** → **image_metadata.id** (CASCADE DELETE)
- Deleting an image removes all associated annotations and access logs

---

## JSON Fields

### analysis_results (image_metadata)

Stores ML model predictions and metrics.

**Example Structure**:
```json
{
  "model": "efficientnet_b3",
  "version": "v2.1.0",
  "timestamp": "2025-02-25T10:30:00Z",
  "disease": "tomato_early_blight",
  "confidence": 0.94,
  "probabilities": {
    "tomato_early_blight": 0.94,
    "tomato_late_blight": 0.03,
    "healthy": 0.02
  },
  "bounding_boxes": [
    {
      "x": 120, "y": 85, "width": 180, "height": 160,
      "label": "lesion", "score": 0.89
    }
  ],
  "features": {
    "color_histogram": [0.12, 0.23, 0.45, ...],
    "texture_score": 0.67
  }
}
```

**Query Examples**:
```sql
-- Extract specific JSON fields
SELECT 
    file_name,
    analysis_results->>'disease' as disease,
    (analysis_results->>'confidence')::float as confidence
FROM image_metadata
WHERE analysis_results IS NOT NULL;

-- Filter by JSON values
SELECT file_name
FROM image_metadata
WHERE (analysis_results->>'confidence')::float > 0.9
  AND analysis_results->>'disease' = 'tomato_early_blight';

-- Count predictions by disease
SELECT 
    analysis_results->>'disease' as disease,
    COUNT(*) as count
FROM image_metadata
WHERE analysis_results IS NOT NULL
GROUP BY disease
ORDER BY count DESC;
```

### annotation_data (image_annotations)

Flexible structure based on `annotation_type`.

**Querying Nested JSON**:
```sql
-- Extract bounding box coordinates
SELECT 
    im.file_name,
    (ia.annotation_data->'boxes'->0->>'x')::int as box_x,
    (ia.annotation_data->'boxes'->0->>'y')::int as box_y
FROM image_annotations ia
JOIN image_metadata im ON ia.image_id = im.id
WHERE ia.annotation_type = 'bounding_box';
```

---

## Best Practices

### Timeseries Data

1. **Use time_bucket()** for aggregations:
   ```sql
   SELECT time_bucket('1 hour', datetime), AVG(temp)
   FROM weather_data
   GROUP BY 1;
   ```

2. **Leverage continuous aggregates** for frequently accessed rollups:
   ```sql
   CREATE MATERIALIZED VIEW daily_weather
   WITH (timescaledb.continuous) AS
   SELECT time_bucket('1 day', datetime) as day,
          AVG(temp) as avg_temp,
          MAX(temp) as max_temp
   FROM weather_data
   GROUP BY day;
   ```

3. **Partition by time** for large datasets (automatically handled by TimescaleDB)

4. **Use EXPLAIN ANALYZE** to verify index usage on time-range queries

### Image Metadata

1. **Batch inserts** for bulk uploads:
   ```python
   cursor.executemany(
       "INSERT INTO image_metadata (...) VALUES (...)",
       batch_data
   )
   ```

2. **Use JSONB** (not JSON) for indexable fields

3. **Tag strategy**: Use arrays for fixed tags, JSONB for dynamic metadata

4. **Cascade deletes**: Rely on FK constraints to clean up annotations/logs

5. **Update timestamps**: Trigger automatically updates `updated_at`

### General

1. **Connection pooling**: Use pgBouncer or SQLAlchemy pools for high concurrency

2. **Regular VACUUM**: Run `VACUUM ANALYZE` weekly on active tables

3. **Monitor index bloat**: Check with `pg_stat_user_indexes`

4. **Backup strategy**: 
   - Daily snapshots of PostgreSQL
   - Separate MinIO bucket backups
   - Point-in-time recovery enabled

5. **Query optimization**:
   - Always filter on indexed columns first
   - Use `LIMIT` for exploratory queries
   - Avoid `SELECT *` in production code

---

## Summary Statistics

### Current Database Size

```sql
-- Total database size
SELECT pg_size_pretty(pg_database_size('agritwin_db'));

-- Size per table
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

### Data Volume

| Table | Records | Size | Notes |
|-------|---------|------|-------|
| weather_data | ~17,000 | ~2 MB | Hourly data for 2024-2025 |
| greenhouse_data | ~200,000 | ~25 MB | 5-minute intervals |
| image_metadata | 69,607 | ~15 MB | 3 categories of tomato images |
| image_annotations | varies | <1 MB | ML training labels |
| image_access_log | varies | <5 MB | Usage tracking |

**Total Image Storage (MinIO)**: ~11.8 GB

---

## Additional Resources

- **Query Examples**: `examples/query_timeseries_examples.py`, `examples/query_images_examples.py`
- **Schema Files**: `database/schema/image_metadata.sql`
- **Models**: `src/agritwin_gh/models/timeseries.py`
- **TimescaleDB Docs**: https://docs.timescale.com/
- **PostgreSQL JSONB**: https://www.postgresql.org/docs/current/datatype-json.html

---

**Document Version**: 1.0  
**Last Updated**: 2026-02-25  
**Database Version**: PostgreSQL 15 + TimescaleDB
