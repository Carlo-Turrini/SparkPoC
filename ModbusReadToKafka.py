from pymodbus.client.sync import ModbusTcpClient
import time
import json
from confluent_kafka import Producer
import datetime as dt


# Script che legge i dati presenti negli holding registers Modbus e li scrive sul topic Kafka temperature_modbus
def main():
    #Sostituire l'indirizzo del client Modbus, usando ModbusPal dovrebbe essere localhost
    client = ModbusTcpClient('172.24.80.1', port=502)
    client.connect()
    UNIT = 0x1
    topic = "temperature_modbus"
    # Se localhost non dovesse funzionare come indirzzo per il broker Kafka, bisogna recuperare l'indirizzo del
    # container del broker.
    prod_conf = {'bootstrap.servers': "localhost:9092"}
    producer: Producer = Producer(prod_conf)
    key = "modb_temp"
    try:
        while True:
            # I dati vengono prodotti e inviati a Kafka con la frequenza di 1Hz
            set_t = client.read_holding_registers(1, 1, unit=UNIT)
            v_t = client.read_holding_registers(3, 1, unit=UNIT)
            value = json.dumps({'set_t': set_t.registers[0], 'v_t': v_t.registers[0],
                                'event_timestamp': dt.datetime.now().strftime('%s')})
            print("Producing record: {}\t{}".format(key, value))
            producer.produce(topic=topic, key=key, value=value, on_delivery=acked)
            producer.poll(1)
            time.sleep(1)
    except(SystemExit, KeyboardInterrupt):
        producer.flush()
        client.close()


def acked(err, msg):
    if err is not None:
        print("Failed to deliver message: %s: %s" % (str(msg), str(err)))
    else:
        print("Message produced: %s" % (str(msg)))


if __name__ == "__main__":
    main()