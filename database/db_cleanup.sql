-- ============================================================
--  db_cleanup.sql  —  cleanup queries for realtime_greenhouse_stream
--
--  Run these in psql or any PostgreSQL client to remove test data
--  written by run_realtime_loop.py.
-- ============================================================


-- ── 1. Inspect what runs exist before deleting ───────────────────────────────
SELECT
    run_id,
    MIN(datetime)  AS started,
    MAX(datetime)  AS ended,
    COUNT(*)       AS total_rows,
    SUM(CASE WHEN source = 'bootstrap' THEN 1 ELSE 0 END) AS bootstrap_rows,
    SUM(CASE WHEN source = 'dt_sim'    THEN 1 ELSE 0 END) AS dt_sim_rows
FROM realtime_greenhouse_stream
GROUP BY run_id
ORDER BY started DESC;


-- ── 2. Delete rows from a SPECIFIC run (replace the run_id value) ────────────
DELETE FROM realtime_greenhouse_stream
WHERE run_id = 'rt_20260402_185434';


-- ── 3. Delete ALL realtime stream data (full table reset) ────────────────────
DELETE FROM realtime_greenhouse_stream;


-- ── 4. Truncate for faster full reset (no row-level transaction log) ─────────
--      WARNING: cannot be rolled back; also resets the auto-increment ID.
TRUNCATE TABLE realtime_greenhouse_stream RESTART IDENTITY;


-- ── 5. Delete only old runs (keep the last 7 days of data) ───────────────────
DELETE FROM realtime_greenhouse_stream
WHERE created_at < NOW() - INTERVAL '7 days';


-- ── 6. Delete only bootstrap seed rows (keep DT-simulated outputs) ───────────
DELETE FROM realtime_greenhouse_stream
WHERE source = 'bootstrap';


-- ── 7. Delete runs older than a specific date ────────────────────────────────
DELETE FROM realtime_greenhouse_stream
WHERE created_at < '2026-04-01 00:00:00';


-- ── 8. Keep only the last N runs (delete everything older) ───────────────────
--      This example keeps the most recent 3 distinct run_ids.
DELETE FROM realtime_greenhouse_stream
WHERE run_id NOT IN (
    SELECT DISTINCT run_id
    FROM   realtime_greenhouse_stream
    ORDER  BY MIN(datetime) DESC
    LIMIT  3
);


-- ── 9. Verify the table is empty after cleanup ───────────────────────────────
SELECT COUNT(*) AS remaining_rows FROM realtime_greenhouse_stream;
