{{
    config(
        materialized='table',
        unique_key='id'
    )
}}

SELECT *
FROM {{
    ref('predicted_race_data')
}}