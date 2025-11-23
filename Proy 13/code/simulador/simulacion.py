import simpy
import random
import numpy as np

def cliente(env, servidor, MU, tiempos_espera, tiempos_sistema):
    llegada = env.now

    with servidor.request() as req:
        yield req

        inicio_servicio = env.now
        tiempos_espera.append(inicio_servicio - llegada)

        servicio = random.expovariate(MU)
        yield env.timeout(servicio)

        tiempos_sistema.append(env.now - llegada)


def generador(env, LAMBDA, servidor, MU, tiempos_espera, tiempos_sistema):
    while True:
        interarrival = random.expovariate(LAMBDA)
        yield env.timeout(interarrival)
        env.process(cliente(env, servidor, MU, tiempos_espera, tiempos_sistema))


def simular_mm_c(LAMBDA, MU, c, TIEMPO):
    env = simpy.Environment()
    servidor = simpy.Resource(env, capacity=c)

    tiempos_espera = []
    tiempos_sistema = []

    env.process(generador(env, LAMBDA, servidor, MU, tiempos_espera, tiempos_sistema))
    env.run(until=TIEMPO)

    Wq_sim = np.mean(tiempos_espera)
    W_sim = np.mean(tiempos_sistema)

    return Wq_sim, W_sim, tiempos_espera, tiempos_sistema
