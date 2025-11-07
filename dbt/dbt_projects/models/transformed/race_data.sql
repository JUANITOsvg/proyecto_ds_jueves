{{
    config(
        materialized='table',
        unique_key='id'
    )
}}

SELECT 
    *
FROM
    {{ source('dev', 'transformed_race_data')}}