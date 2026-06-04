-- Schema for the turbofan engine sensor dataset.
-- Targeted at PostgreSQL; portable to most SQL dialects.

CREATE TABLE IF NOT EXISTS engine_cycle (
    engine_id  INTEGER NOT NULL,
    cycle      INTEGER NOT NULL,
    s1         DOUBLE PRECISION,
    s2         DOUBLE PRECISION,
    s3         DOUBLE PRECISION,
    s4         DOUBLE PRECISION,
    ttf        INTEGER,                -- only populated for the training set
    dataset    VARCHAR(10) NOT NULL,   -- 'train' or 'test'
    PRIMARY KEY (engine_id, cycle, dataset)
);

CREATE INDEX idx_engine_cycle_dataset ON engine_cycle(dataset);
CREATE INDEX idx_engine_cycle_engine  ON engine_cycle(engine_id);

CREATE TABLE IF NOT EXISTS engine_test_truth (
    engine_id  INTEGER PRIMARY KEY,
    true_ttf   INTEGER NOT NULL
);
