from seldon_core.seldon_client import SeldonClient
import numpy as np

# Copy model to PV:
#   kubectl cp /root/my-local-file my-namepace/my-pod:/root/remote-filename
#   kubectl cp /home/tarlo/models/sacmi-anomaly-detection/ istio-seldon/data-access:/data/models/lstm-anomaly
# Deploy model:
#   kubectl apply -f /mnt/c/Users/carlo/PycharmProjects/SparkPoC/lstm_sacmi_anomaly_detection.yaml
# Esponi istio:
#   minikube tunnel
# Port forwarding:
#   kubectl port-forward $(kubectl get pods -l istio=ingressgateway -n istio-system
#   -o jsonpath='{.items[0].metadata.name}') -n istio-system 8003:8080
# Delete deployment:
#   kubectl delete -f /mnt/c/Users/carlo/PycharmProjects/SparkPoC/lstm_sacmi_anomaly_detection.yaml
def main():
    sc = SeldonClient(deployment_name='lstm-sacmi-model', namespace='istio-seldon',
                      gateway_endpoint='localhost:8003', gateway='istio')
    data_point = [[[0.45, 0.66], [0.45, 0.57], [0.45, 0.66], [0.45, 0.57], [0.45, 0.66], [0.45, 0.57],
                  [0.45, 0.66], [0.45, 0.57], [0.45, 0.66], [0.45, 0.57], [0.45, 0.66], [0.45, 0.57],
                  [0.45, 0.66], [0.45, 0.57], [0.45, 0.66], [0.45, 0.57], [0.45, 0.66], [0.45, 0.57],
                  [0.45, 0.66], [0.45, 0.53]]]
    data_arr = np.array(data_point).astype(np.float32)
    print("Shape before reshape: " + str(data_arr.shape))
    data_arr = data_arr.reshape(1, 20, 2)
    print("Shape after reshape: " + str(data_arr.shape))
    r = sc.predict(transport='grpc', shape=data_arr.shape, data=data_arr, client_return_type='dict')
    print(np.array(r.response.get('data').get('tftensor').get('floatVal')).reshape(-1, 20, 2))


if __name__ == "__main__":
    main()