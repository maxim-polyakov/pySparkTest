-- PostgreSQL: таблица alpaca (как в HF tatsu-lab/alpaca, без id)
CREATE TABLE IF NOT EXISTS alpaca (
    instruction TEXT NOT NULL,
    input       TEXT NOT NULL DEFAULT '',
    output      TEXT NOT NULL,
    text        TEXT NOT NULL
);
