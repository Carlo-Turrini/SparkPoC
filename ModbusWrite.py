from pymodbus.client.sync import ModbusTcpClient
import random
import time


def main():
    # SET_T varia ogni 600 secondi (ovvero 10 minuti)
    # V_T varia ogni secondo
    client = ModbusTcpClient('localhost', port=502)
    client.connect()
    try:
        # UNIT rappresenta l'identificativo dello slave Modbus su cui andare a scrivere i dati
        UNIT = 0x1
        counter = 0
        set_t = 0
        while True:
            if counter == 0:
                set_t = random.randint(20, 50)
                client.write_register(1, set_t, unit=UNIT)
            v_t = random.randint(set_t - 2, set_t + 2)
            client.write_register(3, v_t, unit=UNIT)
            counter = 0 if counter + 1 == 600 else counter + 1
            time.sleep(1)
    except(SystemExit, KeyboardInterrupt):
        client.close()


if __name__ == "__main__":
    main()