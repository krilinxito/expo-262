import requests
import random
import time

URL = "http://127.0.0.1:5000/"
n_solicitudes = 50

for i in range(n_solicitudes):
    interarrival = random.expovariate(1/2)  # media 2 seg -> λ ≈ 0.5
    time.sleep(interarrival)
    inicio = time.time()
    r = requests.get(URL)
    print(f"Req {i+1}: {r.text.strip()} (total {time.time() - inicio:.2f}s)")
