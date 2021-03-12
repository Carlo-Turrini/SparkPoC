from pyspark.sql import SparkSession, DataFrame
from pyspark import SparkConf, SparkContext
from pyspark.sql.column import Column, _to_java_column, _to_seq
from pyspark.sql.functions import col, window, struct, unix_timestamp
from pyspark.ml.feature import VectorAssembler, MinMaxScalerModel
from pyspark.sql.functions import to_json, from_json
from pyspark.sql.pandas.functions import pandas_udf
from pyspark.ml.functions import vector_to_array
import pandas as pd
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, TimestampType, \
    BooleanType, ArrayType, LongType
import numpy as np
from typing import Iterator
import grpc
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
import tensorflow as tf
"""
spark-submit \
--jars ./scala/sbtscalademo_2.12-0.2.jar \
--packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.0.1 \
Spark_TFServing_Streaming_Kafka_Predictions.py
"""
def main():
    ss = None
    try:
        #Minikube esplode se settiamo il parallelismo di Spark > 1 essendo che esegue in locale su un nodo solo
        sconf = SparkConf() \
            .set("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .set("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")
        ss = SparkSession.builder \
            .appName("Streaming Inference - TF Serving") \
            .config(conf=sconf) \
            .master("local[1]")\
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
            col('timestamp').cast(LongType()).alias('timestamp_long'),
            col('scaled_features')
        )

        assembler2 = VectorAssembler(inputCols=['timestamp_long', 'scaled_features'],
                                     outputCol='features_with_timestamp')
        assembled_df = assembler2.transform(scaled_df).select(
            col('timestamp'),
            vector_to_array(col('features_with_timestamp')).alias('features_with_timestamp')
        )

        #Builds ordered sequences for predictions
        def featurize(col):
            sc = SparkContext._active_spark_context
            _featurize = sc._jvm.udaf_demo.GroupArray.apply
            return Column(_featurize(_to_seq(sc, [col], _to_java_column)))

        features_df = assembled_df.withWatermark('timestamp', '5 seconds') \
            .groupby(window(col('timestamp'), '20 seconds', '1 seconds')) \
            .agg(featurize('features_with_timestamp').alias('features'))

        features_df.printSchema()

        pandas_schema = StructType([
            StructField('isAnomaly', BooleanType(), True),
            StructField('threshold_set_t', DoubleType(), True),
            StructField('threshold_v_t', DoubleType(), True),
            StructField('mae_set_t', DoubleType(), True),
            StructField('mae_v_t', DoubleType(), True),
            StructField('prediction', ArrayType(ArrayType(DoubleType())), True)
        ])

        @pandas_udf(pandas_schema)
        def predict(series_iterator: Iterator[pd.Series]) -> Iterator[pd.DataFrame]:
            #Fare import necessari?
            #   import pandas as pd
            #   import numpy as np
            #   from seldon_core.seldon_client import SeldonClient
            #Fai il load del MAE threshold!
            threshold = pd.read_parquet(engine='pyarrow', path='file:///home/tarlo/sacmi_mae_threshold')
            threshold = threshold.to_numpy()[0][0]
            #Avvia GRPC per TFServing
            channel = grpc.insecure_channel('localhost:8500')
            stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
            grpc_request = predict_pb2.PredictRequest()
            grpc_request.model_spec.name = 'lstm_anomaly'
            grpc_request.model_spec.signature_name = 'serving_default'
            for series in series_iterator:
                results = []
                for elem in series.values:
                    elem = elem.tolist()
                    if len(elem) < 20:
                        data_point = [None, threshold[0], threshold[1], None, None, None]
                        results.append(data_point)
                    else:
                        features = np.array(elem).astype(np.float32)
                        features = features.reshape(1, 20, 2)
                        grpc_request.inputs['input'].CopyFrom(tf.make_tensor_proto(features, shape=features.shape))
                        result = stub.Predict(grpc_request, 10)
                        prediction = np.array(list(result.outputs['output'].float_val)).reshape(-1, 20, 2)
                        mae = np.mean(np.abs(prediction - features), axis=1)[0]
                        anomaly = mae[0] > threshold[0] or mae[1] > threshold[1]
                        data_point = [anomaly, threshold[0], threshold[1], mae[0], mae[1], prediction.tolist()[0]]
                        results.append(data_point)
                yield pd.DataFrame(data=results, columns=['isAnomaly', 'threshold_set_t', 'threshold_v_t',
                                                          'mae_set_t', 'mae_v_t', 'prediction'])

        prediction_df = features_df.select(
            col('window'),
            col('features'),
            predict('features').alias('predictions')
        )

        #Se voglio filtrare le righe che presentano valori null, ovvero per le quali non è stata fatta alcuna
        #   prediction per mancanza di dati sufficienti nella window:

        prediction_df = prediction_df.select(
            col('window.start').alias('window_start'),
            col('window.end').alias('window_end'),
            col('features'),
            col('predictions.isAnomaly').alias('isAnomaly'),
            col('predictions.mae_set_t').alias('mae_set_t'),
            col('predictions.mae_v_t').alias('mae_v_t'),
            col('predictions.threshold_set_t').alias('threshold_set_t'),
            col('predictions.threshold_v_t').alias('threshold_v_t'),
            col('predictions.prediction').alias('prediction')
        ).filter(col('isAnomaly').isNotNull())


        #Scrivo i risultati delle prediction su Kafka
        sq = prediction_df.withColumn("value", struct(col('window_start'),
                                                      col('window_end'),
                                                      unix_timestamp(col('window_end')).alias('event_timestamp'),
                                                      col('isAnomaly'),
                                                      col('mae_set_t'),
                                                      col('mae_v_t'),
                                                      col('threshold_set_t'),
                                                      col('threshold_v_t'),
                                                      col('features'),
                                                      col('prediction')))\
            .select(
                to_json(col('value')).alias('value'),
                col('window_end').cast(StringType()).alias('key'),
            )\
            .writeStream.format('kafka')\
            .option('kafka.bootstrap.servers', 'localhost:9092')\
            .option('topic', 'predictions')\
            .option('checkpointLocation', '/tmp/spark-streaming-tf-serving-checkpoint')\
            .start()
        sq.awaitTermination()

    except(SystemExit, KeyboardInterrupt):
        if ss is not None:
            ss.stop()


if __name__ == "__main__":
    main()