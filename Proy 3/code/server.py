from flask import Flask, request
import time
import csv
import os
from threading import Lock

app = Flask(__name__)
lock = Lock()  
logfile = "tiempos_servicio.csv"

# crea archivo de logs si no existe
if not os.path.exists(logfile):
    with open(logfile, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_llegada", "tiempo_servicio", "timestamp_salida"])

@app.route("/")
def atender():
    llegada = time.time()
    with lock:  
        inicio_servicio = time.time()
        # simula tiempo de servicio
        time.sleep(1 + 0.5 * (time.time() % 1))  # simula variabilidad en el tiempo de servicio
        salida = time.time()
        duracion = salida - inicio_servicio

        # guardar registro
        with open(logfile, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([llegada, duracion, salida])

    return f"Atendida en {duracion:.2f}s\n"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=False)
