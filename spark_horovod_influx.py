from influxdb_client import InfluxDBClient
from influxdb_client.client.flux_table import FluxTable
from pyspark.sql import SparkSession, DataFrame
from pyspark import SparkConf
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, ArrayType, DoubleType, IntegerType, TimestampType
from pyspark.ml.feature import VectorAssembler, MinMaxScaler, MinMaxScalerModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql.pandas.functions import pandas_udf
import horovod.spark.keras as hvd
from horovod.spark.keras import KerasModel
from horovod.spark.common.store import LocalStore
import os
from tensorflow import keras
import pandas as pd
import numpy as np


def influx_data():
    # You can generate a Token from the "Tokens Tab" in the UI
    token = "Wcn-rvYOH1ax_LrpNm3sMmFl-470BCC0Bws_jyR1IfzGPCKT4Bzh2YbkRyNGhmSBeKgjMFVIaE7C_Oir7Lv3Vg=="
    org = "BD4M"
    bucket = "telegraf_kafka"
    # url = "http://172.26.169.47:8086"
    url = "http://localhost:8086"

    client = InfluxDBClient(url=url, token=token, org=org)

    query = f'from(bucket: \"{bucket}\")' \
            f'|> range(start: -1d) ' \
            f'|> filter(fn: (r) => r["_measurement"] == "kafka_consumer")' \
            f'|> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")' \
            f'|> keep(columns: ["_time", "set_t", "v_t"])' \
            f'|> rename(columns: {{_time: "timestamp"}})' \
            f'|> sort(columns: ["timestamp], desc: false)'

    tables: list[FluxTable] = client.query_api().query(query)
    output_list = []
    for table in tables:
        for record in table.records:
            tuple_record = (record.values.get("timestamp"),
                            record.values.get("set_t"),
                            record.values.get("v_t"))
            output_list.append(tuple_record)
    return output_list


def main():
    ss = None
    num_proc = 2
    try:
        config = SparkConf() \
            .set("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .set("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")
        ss = SparkSession.builder \
            .master("local[{}]".format(num_proc)) \
            .appName("Sacmi Use Case Demo") \
            .config(conf=config) \
            .getOrCreate()

        schema = StructType([
            StructField('timestamp', TimestampType(), False),
            StructField('set_t', IntegerType(), False),
            StructField('v_t', IntegerType(), False)
        ])

        # Crete df from influx data
        df = ss.createDataFrame(data=influx_data(), schema=schema)

        # Prepare features for training with Horovod
        assembler = VectorAssembler(inputCols=["set_t", "v_t"], outputCol="features")

        features_df = assembler.transform(df)

        # Per lo use case Sacmi forse varrebbe la pena di settare come min 20 e come max 50
        scaler = MinMaxScaler(inputCol="features", outputCol="scaled_features")

        scaler_model: MinMaxScalerModel = scaler.fit(features_df)
        scaler_model.save('/home/tarlo/min_max_scaler_model_modbus')

        scaled_df = scaler_model.transform(features_df)

        # Crea df con le sequenze per il training
        scaled_df = scaled_df.select(
            col('set_t'),
            col('v_t'),
            col('timestamp'),
            vector_to_array(col('scaled_features')).alias('scaled_features')
        )

        # Nota: se la conversione da pandas a arrow dovesse fallire, ricorda che arrow non
        # può convertire ndarray di dimensione > 1, però può convertire liste innestate per cui
        # ricorda nel caso di convertire la struttura in modo tale che vi siano al più ndarray di dim == 1
        def featurize(iterator):
            TIME_STEPS = 20
            for pdf in iterator:
                values = pdf['scaled_features'].values.tolist()
                df = pd.DataFrame(values, columns=['set_t', 'v_t'])
                print(df)
                xs, ys = [], []
                for i in range(len(df) - TIME_STEPS):
                    xs.append(df.iloc[i:(i + TIME_STEPS)].values.tolist())
                    ys.append(df.iloc[i + TIME_STEPS].values)
                yield pd.DataFrame({'features': xs, 'labels': ys})

        pandas_schema = StructType([
            StructField("features", ArrayType(ArrayType(DoubleType())), False),
            StructField("labels", ArrayType(DoubleType()), False)
        ])

        feature_df = scaled_df.coalesce(1).mapInPandas(featurize, schema=pandas_schema)

        # Prepare Horovod Estimator for training
        store = LocalStore('file:///home/tarlo/spark_horovod_influx_sacmi_store')

        # Tutto da rivedere! Devi prima creare delle sequenze per i dati da mandare in input alla rete
        # Per info cerca online LSTM for anomaly detection
        # Se non ti ci raccapezzi magari usa la Horovod Run Api che è di più basso livello e
        # ti permette di avere più controllo rispetto alla Horovod Estimator Api
        # Chiaramente devi anche cambiare la label_cols in KerasEstimator
        def model_builder():
            input_layer = keras.layers.Input(shape=(20, 2), name='input')

            def builder(input):
                # Input shape: (TIME_SAMPLES, FEATURES) -> note: we have 2 features
                time_steps = 20
                features = 2
                lstm1 = keras.layers.LSTM(units=64, name='lstm1')(input)
                drop1 = keras.layers.Dropout(rate=0.2, name='drop1')(lstm1)
                repeat_vec = keras.layers.RepeatVector(n=time_steps, name='repeat_vec')(drop1)
                lstm2 = keras.layers.LSTM(units=64, return_sequences=True, name='lstm2')(repeat_vec)
                drop2 = keras.layers.Dropout(rate=0.2, name='drop2')(lstm2)
                time_distrib = keras.layers.TimeDistributed(keras.layers.Dense(features), name='output')(drop2)
                return time_distrib

            model = keras.models.Model(input_layer, builder(input_layer), name="lstm_sacmi")
            model.summary()
            return model

        model = model_builder()
        optimizer = 'adam'
        loss = 'mae'
        batch_size = 64
        epochs = 10

        # Disable shuffling of data! Enable statefulness of LSTM! Probably need to use Horovod Run API?
        # Maybe setting shuffle_buffer_size=0 will do the trick
        keras_estimator = hvd.KerasEstimator(
            num_proc=num_proc,
            shuffle_buffer_size=1,
            store=store,
            model=model,
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy'],
            batch_size=batch_size,
            epochs=epochs,
            feature_cols=['features'],
            label_cols=['labels'])

        keras_model: KerasModel = keras_estimator.fit(feature_df) \
            .setOutputCols(['predict'])
        # Prima di salvare il modello ricorda di calcolare il max di MAE sul training dataset e di salvarlo
        trained_model = keras_model.getModel()

        model_dir = "/home/tarlo/models/sacmi_anomaly_detection"
        version = 1
        export_path = os.path.join(model_dir, str(version))

        keras.models.save_model(
            trained_model,
            export_path,
            overwrite=True,
            include_optimizer=True,
            save_format='tf',
            signatures=None,
            options=None)

        pred_df = keras_model.transform(features_df)

        @pandas_udf("double")
        def calc_mae(pred_series: pd.Series, orig_series: pd.Series) -> pd.Series:
            mae_arr = []
            for pred, orig in zip(pred_series, orig_series):
                pred = np.array(pred)
                orig = np.array(orig)
                mae_arr.append(np.mean(np.abs(pred - orig), axis=1))
            return pd.Series(mae_arr)

        pred_df = pred_df.withColumn("mae", calc_mae("predict", "features"))

        max_mae_df = pred_df.groupby().max('mae')

        max_mae_df.write.mode("overwrite").csv(path='/home/tarlo/sacmi_mae_threshold', header=True)

        ss.stop()

    except(SystemExit, KeyboardInterrupt):
        if ss is not None:
            ss.stop()


if __name__ == "__main__":
    main()
