-- Create schemas
CREATE SCHEMA IF NOT EXISTS dev;
CREATE SCHEMA IF NOT EXISTS warehouse;

-- Source table: dev.transformed_race_data
CREATE TABLE IF NOT EXISTS dev.transformed_race_data (
    driverId INT,
    avg_race_pos FLOAT,
    avg_sprint_pos FLOAT,
    avg_lap_time FLOAT,
    points FLOAT,
    avg_qual_pos FLOAT,
    forename TEXT,
    surname TEXT
);

-- Target table: warehouse.predicted_race_data
CREATE TABLE IF NOT EXISTS warehouse.predicted_race_data (
    driverId INT,
    avg_race_pos FLOAT,
    avg_sprint_pos FLOAT,
    avg_lap_time FLOAT,
    points FLOAT,
    avg_qual_pos FLOAT,
    forename TEXT,
    surname TEXT,
    driver TEXT,
    driver_encoded INT,
    win BOOLEAN,
    win_probability FLOAT
);
