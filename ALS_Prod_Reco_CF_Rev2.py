from pyspark.ml.feature import Normalizer, StandardScaler, MinMaxScaler
from pyspark.sql.functions import udf, lit, col, slice
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, FloatType, DoubleType
from pyspark.sql.functions import split, explode
import subprocess
import pyspark.sql.functions


modelPath = ('{0}/shopstyle/model_product_reco/personalization_cf/').format(hdfs_base_dir)

# Query data
dt = spark.sql("select user_id, product_name, brand_name, product_id, brand_id, sum(case when event_type = 'Click' then 1 when event_type = 'Favorite' then 1 when event_type = 'AddSaleAlert' then 1 else 0 end) as click_counts from ods_ss.shopstyle_shopper_events where user_id!=0 and app_name = 'shopstyle' and locale = 'en_US' and event_type in ('Click', 'Favorite', 'AddSaleAlert') and not (event_type like 'Email%') and click_through_event_id > 0 group by 1,2,3,4,5")
#dt.show(10, truncate = False)

#adhoc query
#dt = spark.sql("select user_id, product_name, brand_name, product_id, brand_id, sum(case when event_type = 'Click' then 1 when event_type = 'Favorite' then 1 when event_type = 'AddSaleAlert' then 1 else 0 end) as click_counts from ods_ss.shopstyle_shopper_events where user_id!=0 and app_name = 'shopstyle' and locale = 'en_US' and event_type in ('Click', 'Favorite', 'AddSaleAlert') and not (event_type like 'Email%') and click_through_event_id > 0 group by 1,2,3,4,5 limit 10000")

# Filter null/missing product id's
dt = dt.filter(dt.product_id.isNotNull())
dt = dt.filter(dt.product_id > 1)
dt = dt.filter(dt.user_id > 1)

# Spark models numeric data type only
dt_new = dt.withColumn("user", dt["user_id"].cast(IntegerType())).withColumn("product", dt["product_id"].cast(IntegerType()))

# Group out unique user : product combos and count up clicks oer combo
dt_new.createOrReplaceTempView('dt_new')

sqlquery = "select user, product, product_name, sum(click_counts) as click from dt_new group by user, product, product_name"
dt2 = spark.sql(sqlquery)

# Scale standardize clicks and center data at zero
dt2.createOrReplaceTempView("t")

dt3 = spark.sql("select t.*, (t.click - sub.mnm)/(sub.mxm - sub.mnm) as scaledclicks from t cross join (select min(click) as mnm, max(click) as mxm from t) sub")

########################################################################################################################

# Call best ALS model...

model = ALS.load(path_to_model)

########################################################################################################################

# Build the recommendation model using ALS on the training data
als = ALS(rank = 10, maxIter=5, regParam=0.01, userCol="user", itemCol="product", ratingCol="scaledclicks", implicitPrefs=True,
          coldStartStrategy="drop")
model = als.fit(dt3)



# Predict
predictions = model.transform(dt3)
userRecs = model.recommendForAllUsers(20)

# Break the recommendations from array to columnar format
exploded_1 = userRecs.select("user",explode(col("recommendations")).alias("top_20_products"))
#exploded_2 = exploded_1.rdd.map(lambda x: (x[0], x[1][0], x[1][1]))

# Separate product id and ratings into separate columns from the array type in the top_20_recommendations array column
exploded_2 = exploded_1.withColumn("product_id", col("top_20_products.product")).withColumn("prob", col("top_20_products.rating"))

#exploded_2.top(100).foreach(println)

# # Rename output dataframe
# header = ["user","product_id","prob"]
# result = exploded_2.select("user", "product_id", "prob").toDF(header)

# Drop the recommendations array column
result = exploded_2.select("user", "product_id", "prob")

# Create a product_name and id table
product_index = dt2.select("product_name", 'product').distinct()
#brnd_indx.show()

# Merge results with product index table to join with product names
result_final = result.alias('a').join(product_index.alias('b'), col('a.product_id') == col('b.product')) \
    .select('a.user', 'a.product_id', 'a.prob', 'b.product_name')

# Sort the recommendations table by users and their recommendations in descending order
result_final.createOrReplaceTempView("temp")

final_rec = spark.sql("select temp.* from temp order by temp.user asc, temp.prob asc")

final_rec.show()

# Save output on hadoop cluster
hdfs_base_dir = '/user/hdpprod/data_science'
load_directory = ('{0}/shopstyle/production/personalization_cf').format(hdfs_base_dir)
staging_directory = ('{0}/shopstyle/staging/personalization_cf').format(hdfs_base_dir)
backup_directory = ('{0}/shopstyle/backup/personalization_cf').format(hdfs_base_dir)

subprocess.check_output(['hdfs', 'dfs', '-mkdir', '-p', backup_directory])
subprocess.check_output(['hdfs', 'dfs', '-mkdir', '-p', load_directory])

try:
    subprocess.check_output(['hdfs', 'dfs', '-rm', '-r', staging_directory])
except subprocess.CalledProcessError, e:
    print e.output

try:
    subprocess.check_output(['hdfs', 'dfs', '-mkdir', '-p', staging_directory])
except subprocess.CalledProcessError, e:
    print e.output


final_rec.write.parquet(staging_directory, 'overwrite', partitionBy = None)

subprocess.check_output(['hdfs', 'dfs', '-mv', load_directory, backup_directory])
subprocess.check_output(['hdfs', 'dfs', '-mv', staging_directory, load_directory])

spark.stop()

# result_new.repartition(4).write \
#     .mode('overwrite').option("header", "true").csv("/Users/rdixit/hiAff_all")
