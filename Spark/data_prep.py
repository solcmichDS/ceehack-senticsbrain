# Databricks notebook source
import pyspark.sql.functions as F

# COMMAND ----------

dbutils.fs.mounts()

# COMMAND ----------

# MAGIC %md ### alivecors data

# COMMAND ----------

# MAGIC %fs ls dbfs:/ehh_hackathon_alivecors/AC116529

# COMMAND ----------

# MAGIC %md #### labsALL data

# COMMAND ----------

! apt-get install --upgrade p7zip-full

# COMMAND ----------

# MAGIC %sh
# MAGIC cd /dbfs && curl -sS https://owncloud.ikem.cz/index.php/s/Gd3Km5g0eNY4Kwr/download > labsall.7z && 7z e labsall.7z

# COMMAND ----------

(
    spark.read.option("header",True)
    .option("delimiter", ";").csv('dbfs:/LabsALL\ IOL.csv')
     .write.format('delta').mode("overwrite").saveAsTable('ceehacks_labs_all')
)

# COMMAND ----------

# MAGIC %sql OPTIMIZE ceehacks_labs_all

# COMMAND ----------

labsall = spark.read.table('ceehacks_labs_all')

# COMMAND ----------

display(labsall)

# COMMAND ----------

display(labsall.where(F.col('Patient') == '10164861')) # single patient?

# COMMAND ----------

# MAGIC %md #### hackaton dataset

# COMMAND ----------

! ls /dbfs/ehh_hack/ -l 

# COMMAND ----------

! find /dbfs/ehh_hack/ -name "2021-03-19T02_32_35.674Z.txt"

# COMMAND ----------

! ! find /dbfs/ehh_hack/ -type f -name "2022-11-09*"

# COMMAND ----------

! grep -r --include "dg.txt" 'I501' /dbfs/ehh_hack/     #-type f #-exec grep -H "invalidTemplateName" {} \;    grep  /dbfs/ehh_hack/

# COMMAND ----------

! find /dbfs/ehh_hack/ -type f | grep -v 'dg.txt' | awk -F/ '{print $4}' | sort | uniq | wc -l # count all patients that have at least more than just dg.txt file

# COMMAND ----------

! find /dbfs/ehh_hack/ -type f -printf "%f\n" | sort | uniq -c # count of all the files in the folder

# COMMAND ----------

# MAGIC %fs ls dbfs:/ehh_hack/10164861

# COMMAND ----------

patient_data = (
    spark.read.format("json").load('dbfs:/ehh_hack/*') # read all as json (except ECG files)
      .withColumn('path', F.input_file_name())
      .withColumn('id', F.element_at(F.split(F.col('path'), '/'), -2)) # extract patient ID
      .withColumn('dataset', F.element_at(F.split(F.element_at(F.split(F.col('path'), '/'), -1), '\.'),1)) # extract dataset from the file type
      .cache()
)
df_cleaned = (
  patient_data
    .withColumn('date', # either date or datetime will be present - combine those columns to get either > cast to date
        F.when(F.col('datetime').isNotNull(), F.to_date(F.col('datetime')))
        .when(F.col('date').isNotNull(), F.to_date(F.col('date'), 'yyyy-MM-dd'))
    ) 
    .where(F.col('_corrupt_record').isNull()) # ignore corrupted data (dg.txt)
    .drop('_corrupt_record')
)
# check that all the corrupted rows were dg > OK
# display( 
#     df_cleaned.where((F.col('_corrupt_record').isNotNull()) & (F.col('dataset') != 'dg'))
# )

# COMMAND ----------

dg_cleaned = (
    spark.read.format("text").load('dbfs:/ehh_hack/*/dg.txt') # read all as json (except ECG files)
    .withColumn('path', F.input_file_name())
    .cache()
)

# COMMAND ----------

heart_diseases = (
 dg_cleaned
    .withColumn('id', F.element_at(F.split(F.col('path'), '/'), -2)) # extract patient ID
    .withColumn('dataset', F.element_at(F.split(F.element_at(F.split(F.col('path'), '/'), -1), '\.'),1)) # extract dataset from the file type
    .withColumn('heart_diagnoses', F.array_distinct(F.expr(r"regexp_extract_all(value, '(I[3-4][0-9\.]*|I5[0-2\.]*)', 1)")))
    .withColumn('heart_disease', F.when(F.size('heart_diagnoses') > 0, 1).otherwise(0))
#     .where(F.col('heart_disease') == 1)
    .select('id', 'heart_diagnoses', 'heart_disease')
)
# display(heart_diseases.select(F.explode(F.col('heart_diagnoses'))).groupBy('col').count()) #unique heart diseases

# COMMAND ----------

df_rest = (
    df_cleaned.where(F.col('dataset') != 'bp') # all except bp data
    .groupBy('id', 'date')
    .pivot("dataset")
    .agg(F.element_at(F.collect_list('v'),-1).alias('v')) # take always only last element if there are multiple events in a day
)

bp_cleaned = (
    df_cleaned.where(F.col('dataset') == 'bp') # bp data only
    .groupBy('id', 'date')
    .pivot('dataset')
    .agg(F.element_at(F.collect_list('sys'),-1).alias('sys'), F.element_at(F.collect_list('dia'),-1).alias('dia')) # take always only last element if there are multiple events in a day
)

df_complete = df_rest.join(bp_cleaned, ['id', 'date'], how='outer') # join the dataset
# display(df_complete)
# display(df_rest.join())
# display(df_complete.groupBy('id', 'datetime').count().where(F.col('count') > 1)) # check that we do not have any id+datetime duplicates
# display(df_complete.where( (F.size('energy') > 1) | (F.size('exercise') > 1) | (F.size('hrresting') > 1) | (F.size('hrwalking') > 1) | (F.size('waist') > 1) | (F.size('weight') > 1) )) # check that we do not have any arrays with multiple values for single id+datetime

# COMMAND ----------

# display(df_complete)

# COMMAND ----------

ecg = (
    spark.read.format("json").load('dbfs:/ehh_hack/*/ecg/*') # list all ecg records
    .withColumn('path', F.input_file_name())
    .withColumn('id', F.element_at(F.split(F.col('path'), '/'), -3)) # extract patient ID from folder
    .withColumn('date', F.to_date(F.element_at(F.split(F.element_at(F.split(F.col('path'), '/'), -1), 'T'), 1))) # extract dataset from the file type
    .groupBy('id', 'date')
    .agg(F.element_at(F.collect_list('samples'),-1).alias('ecg'), # take last measurement of the day
         F.element_at(F.collect_list('samplingFrequency'),-1).alias('samplingFrequency'),
         F.element_at(F.collect_list('flags'),-1).alias('flags')
    ) 
)

# COMMAND ----------

# write out dataset
display(
    df_complete.join(ecg, ['id', 'date'], how='outer')
    .join(heart_diseases, ['id'], 'inner')
#     .count()
#     .count()
    .write.format('delta').mode("overwrite").option("overwriteSchema", "True").saveAsTable('ceehacks_patientdata')
)

# COMMAND ----------

# MAGIC %sql OPTIMIZE ceehacks_patientdata

# COMMAND ----------

# MAGIC %sql SELECT * FROM ceehacks_patientdata

# COMMAND ----------

# MAGIC %sql SELECT count(distinct(id)) as patients_count FROM ceehacks_patientdata -- should correspond to the bash count above

# COMMAND ----------

# MAGIC %sql SELECT id,date,count(*) as c FROM ceehacks_patientdata GROUP BY id, date HAVING c > 1 -- check for duplicate id+date, should be empty

# COMMAND ----------

# MAGIC %sql ANALYZE TABLE ceehacks_patientdata COMPUTE STATISTICS NOSCAN

# COMMAND ----------

# MAGIC %md ### ECG sampling
# MAGIC most of the ECG samples are 15k in length.   
# MAGIC we select 100 samples of length 1k per from the middle of the dataset

# COMMAND ----------

import random

# COMMAND ----------

ecg = spark.read.table('ceehacks_patientdata').where(F.col('ecg').isNotNull()).select('id', 'ecg')

# COMMAND ----------

# SAMPLE_SIZE = 1000 # how long a single sample should be
# N_SAMPLES = 100 # how many samples per one patient we want


# for SAMPLE_SIZE in [4000]:
#     for N_SAMPLES in [500]:

N_MEASUREMENTS_THRESHOLD = 10 # filter out patients with at least this many measurements

ecg = spark.read.table('ceehacks_patientdata').where(F.col('ecg').isNotNull()).select('id', 'ecg')

ecg_filtered = ecg.groupBy('id').count().where(F.col('count') > N_MEASUREMENTS_THRESHOLD).join(ecg, ['id'], how='left')

# mean_length = round(ecg.select(F.size('ecg').alias('ecgsize')).agg(F.avg('ecgsize').alias('ecgmean')).collect()[0].ecgmean) # mean length of all ecg measurements (most of them 15k)
max_length = 15000 # max length of all ecg measurements (most of them  15k)
interval_start = round(max_length * 0.2) # we want to select samples from the middle of the interval
interval_end = round(max_length * 0.7) - SAMPLE_SIZE

random.seed(42)
starting_indices = [random.randint(interval_start, interval_end) for x in range(N_SAMPLES)]

(
    ecg_filtered
      .select('id', (F.array_repeat(F.col('ecg'), N_SAMPLES).alias('ecg_multiplied')))
      .withColumn('sample_size', F.lit(SAMPLE_SIZE))
      .withColumn('starting_indices', F.array([F.lit(s) for s in starting_indices])) # multiply array in a single 
      .withColumn("n", F.arrays_zip("ecg_multiplied", "starting_indices")) # zip arrays 
      .withColumn("n", F.explode("n")) # and explode them together
      .select('id', F.slice('n.ecg_multiplied', F.col('n.starting_indices'), F.col('sample_size')).alias('slice')) # slice(col,start,length)
      .write.format('delta').mode("overwrite").saveAsTable(f'ceehacks_ecg_samples_slice_{SAMPLE_SIZE}_n_samples_{N_SAMPLES}')
)

# COMMAND ----------

# MAGIC %sql SELECT * FROM ceehacks_ecg_samples_

# COMMAND ----------

# MAGIC %sql SELECT count(distinct(`id`)) as patients_count
# MAGIC FROM ceehacks_ecg_samples
