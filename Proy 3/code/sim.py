import simpy
import random
import numpy as np

# Parámetros empíricos
LAMBDA =  0.3057  # solicitudes/seg (tasa de llegada)
MU = 0.8185     # solicitudes/seg (tasa de servicio)
TIEMPO_SIMULACION = 200  

# Variables para recolectar datos
tiempos_espera = []
tiempos_sistema = []

def cliente(env, servidor):
    """Proceso que representa a un cliente (una solicitud)."""
    llegada = env.now
    # Espera turno si el servidor está ocupado
    with servidor.request() as req:
        yield req
        inicio_servicio = env.now
        espera = inicio_servicio - llegada
        tiempos_espera.append(espera)

        # Servicio (exponencial)
        tiempo_servicio = random.expovariate(MU)
        yield env.timeout(tiempo_servicio)

        salida = env.now
        tiempos_sistema.append(salida - llegada)

def generador_clientes(env, servidor):
    """Genera clientes según un proceso de Poisson."""
    i = 0
    while True:
        i += 1
        # Tiempo entre llegadas (exponencial)
        interarrival = random.expovariate(LAMBDA)
        yield env.timeout(interarrival)
        env.process(cliente(env, servidor))

# Crear entorno y servidor
env = simpy.Environment()
servidor = simpy.Resource(env, capacity=1)

# Iniciar procesos
env.process(generador_clientes(env, servidor))
env.run(until=TIEMPO_SIMULACION)

# Resultados
print(f"λ = {LAMBDA}, μ = {MU}")
print(f"Promedio tiempo de espera en cola (simulado): {np.mean(tiempos_espera):.4f} s")
print(f"Promedio tiempo en el sistema (simulado): {np.mean(tiempos_sistema):.4f} s")

# Fórmulas teóricas (M/M/1)
rho = LAMBDA / MU
Wq_teorico = rho / (MU - LAMBDA)
W_teorico = 1 / (MU - LAMBDA)
print(f"ρ = {rho:.4f}")
print(f"Wq teórico = {Wq_teorico:.4f} s")
print(f"W  teórico = {W_teorico:.4f} s")
