-- Per-engine lifecycle summary: total cycles, first/last sensor readings and
-- a simple degradation delta. Useful for quickly spotting which engines have
-- the most aggressive sensor drift over their lifetime.

WITH train_cycles AS (
    SELECT
        *,
        MAX(cycle) OVER (PARTITION BY engine_id) AS max_cycle
    FROM engine_cycle
    WHERE dataset = 'train'
)
SELECT
    engine_id,
    COUNT(*)                                AS total_cycles,
    MIN(cycle)                              AS first_cycle,
    MAX(cycle)                              AS last_cycle,
    AVG(s1)                                 AS mean_s1,
    AVG(s3)                                 AS mean_s3,
    MAX(s1) - MIN(s1)                       AS s1_range,
    MAX(s3) - MIN(s3)                       AS s3_range,
    -- Net drift between the first and final 10 cycles of operation
    AVG(CASE WHEN cycle <= 10 THEN s1 END)  AS s1_early_mean,
    AVG(CASE WHEN cycle > max_cycle - 10 THEN s1 END) AS s1_late_mean
FROM train_cycles
GROUP BY engine_id
ORDER BY engine_id;
