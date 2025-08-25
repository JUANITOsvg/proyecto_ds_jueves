-- models/test_table_1.sql
SELECT
    *
FROM {{ source('public', 'test_table_1') }}
