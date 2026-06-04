-- Identify test engines that fall into the "faulty" class (true TTF <= 30
-- cycles) and return their final-cycle sensor snapshot. This is the slice
-- a maintenance team would action first.

WITH last_test_cycle AS (
    SELECT
        engine_id,
        cycle,
        s1, s2, s3, s4,
        ROW_NUMBER() OVER (PARTITION BY engine_id ORDER BY cycle DESC) AS rn
    FROM engine_cycle
    WHERE dataset = 'test'
)
SELECT
    t.engine_id,
    t.true_ttf,
    lc.cycle               AS last_observed_cycle,
    lc.s1, lc.s2, lc.s3, lc.s4,
    CASE WHEN t.true_ttf <= 30 THEN 'FAULTY' ELSE 'HEALTHY' END AS health_status
FROM engine_test_truth t
JOIN last_test_cycle lc ON t.engine_id = lc.engine_id AND lc.rn = 1
WHERE t.true_ttf <= 30
ORDER BY t.true_ttf ASC;
