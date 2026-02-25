-- Image Metadata Schema for AgriTwin-GH
-- Stores metadata for images stored in MinIO object storage

-- Main image metadata table
CREATE TABLE IF NOT EXISTS image_metadata (
    id SERIAL PRIMARY KEY,
    
    -- MinIO Storage Info
    image_key VARCHAR(255) NOT NULL UNIQUE,  -- MinIO object key path
    bucket_name VARCHAR(100) NOT NULL DEFAULT 'agritwin-images',
    file_name VARCHAR(255) NOT NULL,
    file_size BIGINT,  -- Size in bytes
    mime_type VARCHAR(50) DEFAULT 'image/jpeg',
    etag VARCHAR(255),  -- MinIO ETag for file integrity
    
    -- Image Classification
    category VARCHAR(50) NOT NULL,  -- 'disease', 'growth_stage', 'healthy'
    subcategory VARCHAR(100),  -- 'early_blight', 'stage1_seedling', etc.
    label VARCHAR(100),  -- Human-readable label
    
    -- Agricultural Context
    crop_type VARCHAR(50) DEFAULT 'tomato',
    source_dataset VARCHAR(100),  -- Original dataset name
    original_path TEXT,  -- Original file path before upload
    
    -- Image Properties
    width INTEGER,
    height INTEGER,
    color_space VARCHAR(20),  -- RGB, RGBA, etc.
    
    -- Processing Status
    is_uploaded BOOLEAN DEFAULT FALSE,
    upload_date TIMESTAMP,
    is_processed BOOLEAN DEFAULT FALSE,
    processed_date TIMESTAMP,
    processing_status VARCHAR(50) DEFAULT 'pending',  -- pending, processing, completed, failed
    
    -- Analysis Results (JSON for flexibility)
    analysis_results JSONB,  -- Store ML model predictions, confidence scores, etc.
    
    -- Metadata
    tags TEXT[],  -- Array of tags for easy filtering
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100),
    
    -- Constraints
    CONSTRAINT valid_category CHECK (category IN ('disease', 'growth_stage', 'healthy')),
    CONSTRAINT valid_status CHECK (processing_status IN ('pending', 'processing', 'completed', 'failed', 'skipped'))
);

-- Indexes for fast querying
CREATE INDEX IF NOT EXISTS idx_image_bucket_key ON image_metadata(bucket_name, image_key);
CREATE INDEX IF NOT EXISTS idx_image_category ON image_metadata(category);
CREATE INDEX IF NOT EXISTS idx_image_subcategory ON image_metadata(subcategory);
CREATE INDEX IF NOT EXISTS idx_image_upload_date ON image_metadata(upload_date DESC);
CREATE INDEX IF NOT EXISTS idx_image_processing_status ON image_metadata(processing_status);
CREATE INDEX IF NOT EXISTS idx_image_crop_type ON image_metadata(crop_type);
CREATE INDEX IF NOT EXISTS idx_image_tags ON image_metadata USING GIN(tags);
CREATE INDEX IF NOT EXISTS idx_image_analysis ON image_metadata USING GIN(analysis_results);

-- Trigger to auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_image_metadata_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_update_image_metadata_timestamp
    BEFORE UPDATE ON image_metadata
    FOR EACH ROW
    EXECUTE FUNCTION update_image_metadata_timestamp();

-- Optional: Image annotations table (for ML training labels)
CREATE TABLE IF NOT EXISTS image_annotations (
    id SERIAL PRIMARY KEY,
    image_id INTEGER NOT NULL REFERENCES image_metadata(id) ON DELETE CASCADE,
    annotation_type VARCHAR(50) NOT NULL,  -- 'bounding_box', 'segmentation', 'classification'
    annotation_data JSONB NOT NULL,  -- Store coordinates, masks, labels, etc.
    confidence FLOAT,  -- Confidence score (0-1)
    annotator VARCHAR(100),  -- Who/what created this annotation
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT valid_annotation_type CHECK (annotation_type IN ('bounding_box', 'segmentation', 'classification', 'keypoint'))
);

CREATE INDEX IF NOT EXISTS idx_annotation_image ON image_annotations(image_id);
CREATE INDEX IF NOT EXISTS idx_annotation_type ON image_annotations(annotation_type);
CREATE INDEX IF NOT EXISTS idx_annotation_data ON image_annotations USING GIN(annotation_data);

-- Optional: Image usage log (track when images are accessed)
CREATE TABLE IF NOT EXISTS image_access_log (
    id SERIAL PRIMARY KEY,
    image_id INTEGER NOT NULL REFERENCES image_metadata(id) ON DELETE CASCADE,
    accessed_by VARCHAR(100),
    access_type VARCHAR(50),  -- 'view', 'download', 'process', 'train'
    accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB  -- Additional context
);

CREATE INDEX IF NOT EXISTS idx_access_image ON image_access_log(image_id);
CREATE INDEX IF NOT EXISTS idx_access_date ON image_access_log(accessed_at DESC);

-- View for easy querying
CREATE OR REPLACE VIEW image_summary AS
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
GROUP BY category, subcategory
ORDER BY category, subcategory;

-- Comments for documentation
COMMENT ON TABLE image_metadata IS 'Central metadata table for images stored in MinIO object storage';
COMMENT ON COLUMN image_metadata.image_key IS 'Unique path/key for the object in MinIO bucket';
COMMENT ON COLUMN image_metadata.category IS 'High-level category: disease, growth_stage, or healthy';
COMMENT ON COLUMN image_metadata.analysis_results IS 'JSON structure for storing ML model predictions and analysis';
