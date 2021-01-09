from pyspark.sql import SparkSession, DataFrame
from pyspark import SparkConf, SparkContext
from pyspark.sql.column import Column, _to_java_column, _to_seq
from pyspark.sql.functions import col, window, struct
from pyspark.ml.feature import VectorAssembler, MinMaxScalerModel
from pyspark.sql.functions import to_json, from_json
from pyspark.sql.pandas.functions import pandas_udf
from pyspark.ml.functions import vector_to_array
import pandas as pd
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, TimestampType, \
    BooleanType, ArrayType, LongType
import numpy as np
from typing import Iterator
from seldon_core.seldon_client import SeldonClient


def main():
    ss = None
    try:
        sconf = SparkConf() \
            .set("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .set("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")
        ss = SparkSession.builder \
            .appName("Streaming Inference - Sacmi") \
            .config(conf=sconf) \
            .master("local[*]")\
            .getOrCreate()

        json_schema = StructType([
            StructField('set_t', IntegerType(), False),
            StructField('v_t', IntegerType(), False)
        ])

        stream_df = ss.readStream\
            .format("kafka")\
            .option("kafka.bootstrap.servers", "localhost:9092")\
            .option("kafka.group.id", "stream_demo_3")\
            .option("subscribe", "temp_modbus_demo")\
            .option("failOnDataLoss", "false")\
            .load()\
            .select(
                col('timestamp').cast(TimestampType()).alias('timestamp'),
                from_json(col('value').cast(StringType()), json_schema).alias('modbus_data')
            )

        flattened_df = stream_df.select(
            col('timestamp'),
            col('modbus_data.set_t').alias('set_t'),
            col('modbus_data.v_t').alias('v_t')
        )

        assembler = VectorAssembler(inputCols=['set_t', 'v_t'], outputCol='features')
        vec_df = assembler.transform(flattened_df)

        scaler_model = MinMaxScalerModel.load('/home/tarlo/min_max_scaler_model_modbus')
        scaled_df = scaler_model.transform(vec_df).select(
            col('timestamp'),
            col('timestamp').cast(LongType()).alias('timestamp_long'),
            col('scaled_features')
            #vector_to_array(col('scaled_features')).alias('scaled_features')
        )
        #Se vuoi tornare come prima rimuovi il secondo VectorAssembler!
        assembler2 = VectorAssembler(inputCols=['timestamp_long', 'scaled_features'], outputCol='features_with_timestamp')
        assembled_df = assembler2.transform(scaled_df).select(
            col('timestamp'),
            vector_to_array(col('features_with_timestamp')).alias('features_with_timestamp')
        )

        """"#Alternativa: scompatto e aggiungo campi:
        #   threshold -> threshold_set_t, threshold_v_t -> DoubleType()
        #   mae -> mae_set_t, mae_v_t -> DoubleType()
        pandas_schema = StructType([
            StructField('num_of_elements', IntegerType(), False)
        ])

        def predict(df: pd.DataFrame) -> pd.DataFrame:
            #Controlla la configurazione una volta che hai fatto il deployment
            #Fai il load del MAE threshold!
            threshold = pd.read_parquet(engine='pyarrow', path='file:///home/tarlo/sacmi_mae_threshold')

            length = len(df)
            print("Lunghezza df: " + str(length))
            print(df)
            return pd.DataFrame({'num_of_elements': [length]})

        @pandas_udf('int')
        def demo_pred(series: pd.Series) -> int:
            print("Series: \n")
            print(series)
            return len(series)"""

        def featurize(col):
            sc = SparkContext._active_spark_context
            _featurize = sc._jvm.udaf_demo.GroupArray.apply
            return Column(_featurize(_to_seq(sc, [col], _to_java_column)))
        """prediction_df = scaled_df.withWatermark('timestamp', '5 seconds')\
            .groupby(window(col('timestamp'), '20 seconds', '1 seconds'))\
            .applyInPandas(predict, pandas_schema)"""

        prediction_df = assembled_df.withWatermark('timestamp', '5 seconds')\
            .groupby(window(col('timestamp'), '20 seconds', '1 seconds'))\
            .agg(featurize('features_with_timestamp').alias('features'))

        prediction_df.printSchema()

        #Scrivo i risultati delle prediction su Kafka
        sq = prediction_df.withColumn("value", struct(col('features'), col('window.start').alias('window_start'),
                                                      col('window.end').alias('window_end')))\
            .select(
                to_json(col('value')).alias('value')
            )\
            .writeStream.format('kafka')\
            .option('kafka.bootstrap.servers', 'localhost:9092')\
            .option('topic', 'stream_demo_pandas')\
            .option('checkpointLocation', '/tmp/spark-checkpoint')\
            .start()
        sq.awaitTermination()

    except(SystemExit, KeyboardInterrupt):
        if ss is not None:
            ss.stop()


if __name__ == "__main__":
    main()