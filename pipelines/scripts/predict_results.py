# TODO: ETL SCRIPT WITH THE MODEL'S API

# 1- BASE CSV/TABLE TRANSFORMED AND STORED INTO transformed_race_data (see dbt/dbt_projects/models/transformed/race_data.sql)
# 2- API REQUEST TO THE PREDICTION API WITH THE DATA WE HAVE ON OUR INPUT DF
# 3- UPLOAD OF THE RESULTING DATA INTO predicted_race_data (dbt/dbt_projects/models/result/race_data.sql)