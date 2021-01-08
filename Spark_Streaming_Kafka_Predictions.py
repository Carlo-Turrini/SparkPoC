from pyspark.sql import SparkSession, DataFrame
from pyspark import SparkConf
from pyspark.sql.functions import col, window, struct
from pyspark.ml.feature import VectorAssembler, MinMaxScalerModel
from pyspark.sql.functions import to_json, from_json
from pyspark.sql.pandas.functions import pandas_udf
from pyspark.ml.functions import vector_to_array
import pandas as pd
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, TimestampType, \
    BooleanType, ArrayType
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
            .option("kafka.group.id", "spark_modbus")\
            .option("subscribe", "temperature_modbus")\
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
            vector_to_array(col('scaled_features')).alias('scaled_features')
        )

        pandas_schema = StructType([
            StructField('isAnomaly', BooleanType(), True),
            StructField('threshold', ArrayType(DoubleType()), False),
            StructField('mae', ArrayType(DoubleType()), True),
            StructField('reconstructed', ArrayType(ArrayType(DoubleType())), True),
            StructField('original', ArrayType(ArrayType(DoubleType())), False)
        ])

        @pandas_udf(pandas_schema)
        def predict(df_iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
            #Controlla la configurazione una volta che hai fatto il deployment
            #Fai il load del MAE threshold!
            threshold = pd.read_parquet(engine='pyarrow', path='file:///home/tarlo/sacmi_mae_threshold')
            sc = SeldonClient(deployment_name='lstm_anomaly', namespace='istio-seldon',
                              gateway_endpoint='localhost:8003', gateway='istio')
            for df in df_iterator:
                if len(df) < 20:
                    yield pd.DataFrame(data=[None, threshold.to_numpy(), None, None,
                                             df['scaled_features'].values.tolist()],
                                       columns=['isAnomaly', 'threshold', 'mae', 'reconstructed', 'original'])
                else:
                    print(df)
                    features = df['scaled_features'].values
                    features_arr = np.array(features)
                    print(features_arr.shape) #Dovrebbe essere (1, 20, 2)

                    print("Da completare")
                    #Fai la chimata al client Seldon
                    #Recupera la prediction
                    #Calcola l'mae tra la prediction e original -> np.mean(np.abs(preds - origs), axis=1)
                    #Confronta con il threshold per determinare se è un'anomalia
                    #Crea il df di ritorno
                    #mae = np.mean(np.abs(prediction - features_arr), axis=1)
                    #Seldon request: tensor, ndarray, tftensor
                    #Seldon response: protobuf, dict
                    #transport:grpc
                    yield pd.DataFrame()

        prediction_df = scaled_df.withWatermark('timestamp', '5 seconds')\
            .groupby(window(col('timestamp'), '20 seconds', '1 seconds'))\
            .apply(predict)

        #Scrivo i risultati delle prediction su Kafka
        sq = prediction_df.withColumn("value", struct(col('window.start').alias('window_start'),
                                                      col('window.end').alias('window_end'),
                                                      col('isAnomaly'),
                                                      col('mae'),
                                                      col('threshold'),
                                                      col('original'),
                                                      col('reconstructed')))\
            .select(
                to_json(col('value')).alias('value'),
                col('window.start').alias('key')
            )\
            .writeStream.format('kafka')\
            .option('kafka.boostrap.servers', 'localhost:9092')\
            .option('topic', 'predictions')\
            .option('checkpointLocation', '/tmp/spark-checkpoint')\
            .start()
        sq.awaitTermination()

    except(SystemExit, KeyboardInterrupt):
        if ss is not None:
            ss.stop()


if __name__ == "__main__":
    main()