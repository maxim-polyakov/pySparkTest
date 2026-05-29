-- MovieLens: movies.csv + ratings.csv
-- PostgreSQL (как в jdbc_postgres_example: host.docker.internal:5432/postgres)

-- Порядок: сначала movies (FK из ratings)

DROP TABLE IF EXISTS ratings;
DROP TABLE IF EXISTS movies;

CREATE TABLE movies (
    "movieId" INTEGER NOT NULL PRIMARY KEY,
    title   TEXT    NOT NULL,
    genres  TEXT
);

COMMENT ON TABLE movies IS 'Каталог фильмов (movies.csv)';
COMMENT ON COLUMN movies."movieId" IS 'ID фильма';
COMMENT ON COLUMN movies.title IS 'Название';
COMMENT ON COLUMN movies.genres IS 'Жанры через |, например Comedy|Drama';

CREATE TABLE ratings (
    "userId"    INTEGER       NOT NULL,
    "movieId"   INTEGER       NOT NULL,
    rating      NUMERIC(2, 1) NOT NULL CHECK (rating >= 0.5 AND rating <= 5.0),
    "timestamp" BIGINT        NOT NULL,
    PRIMARY KEY ("userId", "movieId"),
    CONSTRAINT fk_ratings_movie
        FOREIGN KEY ("movieId") REFERENCES movies ("movieId")
);

COMMENT ON TABLE ratings IS 'Оценки пользователей (ratings.csv)';
COMMENT ON COLUMN ratings."userId" IS 'ID пользователя';
COMMENT ON COLUMN ratings."movieId" IS 'ID фильма';
COMMENT ON COLUMN ratings.rating IS 'Оценка 0.5–5.0';
COMMENT ON COLUMN ratings."timestamp" IS 'Unix time (секунды)';

CREATE INDEX idx_ratings_user ON ratings ("userId");
CREATE INDEX idx_ratings_movie ON ratings ("movieId");
CREATE INDEX idx_ratings_time ON ratings ("timestamp");

-- ---------------------------------------------------------------------------
-- Загрузка CSV (psql на хосте, пути замените на свои)
-- ---------------------------------------------------------------------------
-- \copy movies ("movieId", title, genres) FROM 'movies.csv' WITH (FORMAT csv, HEADER true);
-- \copy ratings ("userId", "movieId", rating, "timestamp") FROM 'ratings.csv' WITH (FORMAT csv, HEADER true);

-- Проверка
-- SELECT COUNT(*) FROM movies;
-- SELECT COUNT(*) FROM ratings;
-- SELECT * FROM movies LIMIT 3;
-- SELECT * FROM ratings LIMIT 3;

-- ---------------------------------------------------------------------------
-- Hive (опционально, схема pysparktest)
-- ---------------------------------------------------------------------------
-- CREATE DATABASE IF NOT EXISTS pysparktest;
-- USE pysparktest;
--
-- DROP TABLE IF EXISTS ratings;
-- DROP TABLE IF EXISTS movies;
--
-- CREATE TABLE movies (
--     movieId INT,
--     title   STRING,
--     genres  STRING
-- )
-- ROW FORMAT DELIMITED FIELDS TERMINATED BY ','
-- STORED AS TEXTFILE
-- TBLPROPERTIES ('skip.header.line.count'='1');
--
-- CREATE TABLE ratings (
--     userId    INT,
--     movieId   INT,
--     rating    DOUBLE,
--     timestamp BIGINT
-- )
-- ROW FORMAT DELIMITED FIELDS TERMINATED BY ','
-- STORED AS TEXTFILE
-- TBLPROPERTIES ('skip.header.line.count'='1');
