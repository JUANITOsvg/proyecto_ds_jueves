{{
    config(
        materialized='table',
        unique_key='id'
    )
}}

SELECT 
    *
FROM
    {{ source('warehouse', 'predicted_race_data')}}