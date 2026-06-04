-- Flag sensor readings that sit more than 3 standard deviations from the
-- training-set mean for each sensor. This is a simple anomaly screen that a
-- data analyst could put on a dashboard before a model is even trained.

WITH sensor_stats AS (
    SELECT
        AVG(s1) AS mu_s1, STDDEV(s1) AS sd_s1,
        AVG(s2) AS mu_s2, STDDEV(s2) AS sd_s2,
        AVG(s3) AS mu_s3, STDDEV(s3) AS sd_s3,
        AVG(s4) AS mu_s4, STDDEV(s4) AS sd_s4
    FROM engine_cycle
    WHERE dataset = 'train'
)
SELECT
    ec.engine_id,
    ec.cycle,
    ec.s1, ec.s2, ec.s3, ec.s4,
    (ec.s1 - ss.mu_s1) / ss.sd_s1 AS z_s1,
    (ec.s2 - ss.mu_s2) / ss.sd_s2 AS z_s2,
    (ec.s3 - ss.mu_s3) / ss.sd_s3 AS z_s3,
    (ec.s4 - ss.mu_s4) / ss.sd_s4 AS z_s4
FROM engine_cycle ec
CROSS JOIN sensor_stats ss
WHERE ec.dataset = 'train'
  AND (
        ABS((ec.s1 - ss.mu_s1) / ss.sd_s1) > 3
     OR ABS((ec.s2 - ss.mu_s2) / ss.sd_s2) > 3
     OR ABS((ec.s3 - ss.mu_s3) / ss.sd_s3) > 3
     OR ABS((ec.s4 - ss.mu_s4) / ss.sd_s4) > 3
  )
ORDER BY ec.engine_id, ec.cycle;
