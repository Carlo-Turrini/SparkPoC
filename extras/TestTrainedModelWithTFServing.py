import grpc
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
import tensorflow as tf
import numpy as np


# Avvia TF server:
#   tensorflow_model_server --port=8500 --rest_api_port=8501 --model_name=lstm_anomaly
#   --model_base_path="/home/tarlo/models/sacmi_anomaly_detection"
def predict():
    channel = grpc.insecure_channel('localhost:8500')
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    grpc_request = predict_pb2.PredictRequest()
    grpc_request.model_spec.name = 'lstm_anomaly'
    grpc_request.model_spec.signature_name = 'serving_default'

    data_point = [[0.45, 0.66], [0.45, 0.57], [0.45, 0.66], [0.45, 0.57], [0.45, 0.66], [0.45, 0.57],
                  [0.45, 0.66], [0.45, 0.57], [0.45, 0.66], [0.45, 0.57], [0.45, 0.66], [0.45, 0.57],
                  [0.45, 0.66], [0.45, 0.57], [0.45, 0.66], [0.45, 0.57], [0.45, 0.66], [0.45, 0.57],
                  [0.45, 0.66], [0.45, 0.57]]
    data_arr = np.array(data_point).astype(np.float32)
    print("Shape before reshape: " + str(data_arr.shape))
    data_arr = data_arr.reshape(1, 20, 2)
    print("Shape after reshape: " + str(data_arr.shape))
    print(type(data_arr))

    grpc_request.inputs['input'].CopyFrom(tf.make_tensor_proto(data_arr, shape=data_arr.shape))
    result = stub.Predict(grpc_request, 10)
    print(result)  # Come si vede result è un tensore con 10 float_val
    # -> il max rappresenta la classe giusta -> trova il modo di estrarre il valore corretto posizionale
    result_arr = np.array(list(result.outputs['output'].float_val))
    result_arr = result_arr.reshape(1, 20, 2)
    print(result_arr)


if __name__ == "__main__":
    predict()
