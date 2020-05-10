import pyspark
from scipy import sparse
from pyspark import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql.functions import collect_list
from pyspark.sql.functions import udf, expr, concat, col
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.ml.recommendation import ALS
from lightfm import LightFM
from lightfm.evaluation import precision_at_k
import time
import numpy as np

sc = SparkContext()

spark = SparkSession(sc)

train_set = spark.read.parquet('hdfs:/user/as12152/subsample_1_train.parquet')
train = train_set.select("user_id", "book_id", "rating")
train = train.selectExpr("user_id as user", "book_id as item", "rating")

rows = np.concatenate(
        train.select("user").rdd.glom().map(
          lambda x: np.array([elem[0] for elem in x]))
        .collect())

cols = np.concatenate(
        train.select("item").rdd.glom().map(
          lambda x: np.array([elem[0] for elem in x]))
        .collect())

data = np.concatenate(
        train.select("rating").rdd.glom().map(
          lambda x: np.array([elem[0] for elem in x]))
        .collect())

n = max(max(rows), max(cols)) + 1

sparse_matrix = sparse.coo_matrix((data, (rows, cols)), 
                    shape=(n, n))

print('sparse matrix created')

model = LightFM(learning_rate=0.5, loss='bpr')

start = time.time()

model.fit_partial(sparse_matrix, epochs=1)

end = time.time()

test_set = spark.read.parquet('hdfs:/user/as12152/subsample_1_test.parquet')
test = test_set.select("user_id", "book_id", "rating")
test = test.selectExpr("user_id as user", "book_id as item", "rating")
rows_test = np.concatenate(
        test.select("user").rdd.glom().map(
          lambda x: np.array([elem[0] for elem in x]))
        .collect())

cols_test = np.concatenate(
        test.select("item").rdd.glom().map(
          lambda x: np.array([elem[0] for elem in x]))
        .collect())

data_test = np.concatenate(
        test.select("rating").rdd.glom().map(
          lambda x: np.array([elem[0] for elem in x]))
        .collect())

n = max(max(rows_test), max(cols_test)) + 1

sparse_matrix_test = sparse.coo_matrix((data_test, (rows_test, cols_test)), 
                    shape=(n, n))


# for 1% sub sample
# learning rate 1 = Model Fitting time is 2.86
# learning rate 0.5 = Model Fitting time is 2.74

print('Model Fitting time is %.2f' % (end-start))

train_precision = precision_at_k(model, sparse_matrix, k=500, num_threads = 4).mean()

print('Precision: train %.2f, test %.2f.' % (train_precision, test_precision))