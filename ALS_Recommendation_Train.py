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
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import pyspark.sql.functions


hdfs_base_dir = '/user/hdpprod/data_science'
modelPath = ('{0}/shopstyle/model_product_reco/personalization_cf/').format(hdfs_base_dir)

# Query data
dt = spark.sql("select user_id, product_name, brand_name, product_id, brand_id, sum(case when event_type = 'Click' then 1 when event_type = 'Favorite' then 1 when event_type = 'AddSaleAlert' then 1 else 0 end) as click_counts from ods_ss.shopstyle_shopper_events where user_id!=0 and app_name = 'shopstyle' and locale = 'en_US' and event_type in ('Click', 'Favorite', 'AddSaleAlert') and not (event_type like 'Email%') and click_through_event_id > 0 group by 1,2,3,4,5")


# Filter null/missing product id's
dt = dt.filter(dt.product_id.isNotNull())
dt = dt.filter(dt.product_id > 1)
dt = dt.filter(dt.brand_id > 1)
dt = dt.filter(dt.user_id > 1)

# Spark models numeric data type only
dt_new = dt.withColumn("user", dt["user_id"].cast(IntegerType())).withColumn("product", dt["product_id"].cast(IntegerType())).withColumn("brand", dt["brand_id"].cast(IntegerType()))

# Group out unique user : product combos and count up clicks per combo
dt_new.createOrReplaceTempView('dt_new')

sqlquery = "select user, product, product_name, sum(click_counts) as click from dt_new group by user, product, product_name"
dt_prod = spark.sql(sqlquery)

# Group unique user : brand combos and count up clicks per combo

sqlquery = "select user, brand, brand_name, sum(click_counts) as click from dt_new group by user, brand, brand_name"
dt_brand = spark.sql(sqlquery)

# Scale standardize clicks and center data at zero
dt_prod.createOrReplaceTempView("t")

dt_prod = spark.sql("select t.*, (t.click - sub.mnm)/(sub.mxm - sub.mnm) as scaledclicks from t cross join (select min(click) as mnm, max(click) as mxm from t) sub")

dt_brand.createOrReplaceTempView("s")

dt_brand = spark.sql("select s.*, (s.click - sub.mnm)/(sub.mxm - sub.mnm) as scaledclicks from s cross join (select min(click) as mnm, max(click) as mxm from s) sub")

# kfold cross validation for the ALS model...
als = ALS(userCol="user", itemCol="product", ratingCol="scaledclicks", implicitPrefs=False, coldStartStrategy="drop")

######
# """Spark allows users to set the coldStartStrategy parameter to “drop” in order to drop any rows in the DataFrame of\
# predictions that contain NaN values. The evaluation metric will then be computed over the non-NaN data and will be valid. """

#"""If the rating matrix is derived from another source of information (i.e. it is inferred from other signals), you\
#  can set implicitPrefs to True to get better results"""
######

def als_tune (df, numFolds = 5):
    (trainingset, validationset) = df.randomSplit([70.0, 30.0])
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="scaledclicks", predictionCol="prediction")
    paramGrid = ParamGridBuilder().addGrid(als.rank, [5, 10]).addGrid(als.maxIter, [5]).addGrid(als.regParam, [0.01, 0.1, 0.5]).build()
    crossval = CrossValidator(estimator=als, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds = numFolds)
    cvModel = crossval.fit(trainingset)
    predictions = cvModel.transform(validationset)
    print("The root mean squared error for our model is: " + str(evaluator.evaluate(predictions.na.drop())))
    print("The best model out of all cross-validated grid search options is: " + str(cvModel.bestModel))
    print("The best model's hyper-parameters are: " + str(cvModel.bestModel.extractParamMap()))

als_tune(dt_prod)

getMaxIter = (cvModel.bestModel._java_obj.parent().getMaxIter())

getRank = cvModel.bestModel.rank

bestModel.save(path_to_model)

spark.stop()

# result_new.repartition(4).write \
#     .mode('overwrite').option("header", "true").csv("/Users/rdixit/hiAff_all")
