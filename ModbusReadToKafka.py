from pymodbus.client.sync import ModbusTcpClient
import time
import json
from confluent_kafka import Producer


def main():
    #Ricorda di avviarlo da WSL2
    #Se non funziona controlla che l'indirizzo IPv4 su Windows sia quello riportato nella connessione
    #   a ModbusTcpClient tramite ipconfig
    client = ModbusTcpClient('192.168.1.179', port=502)
    client.connect()
    UNIT = 0x1
    topic = "temperature_modbus"
    prod_conf = {'bootstrap.servers': "localhost:9092"}
    producer: Producer = Producer(prod_conf)
    key = "modb_temp"
    try:
        while True:
            set_t = client.read_holding_registers(1, 1, unit=UNIT)
            v_t = client.read_holding_registers(3, 1, unit=UNIT)
            value = json.dumps({'set_t': set_t.registers[0], 'v_t': v_t.registers[0]})
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