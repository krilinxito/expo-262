# simulator.py
import simpy
import random
import numpy as np
import pandas as pd
import math   # ← AÑADIDO

def run_simulation(lambda_rate, mu_rate, num_servers, sim_time=600, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
    arrivals = []
    queue_lengths = []
    system_lengths = []
    wait_times = []
    response_times = []

    def arrival(env, servers):
        while True:
            yield env.timeout(random.expovariate(lambda_rate))
            arrivals.append(env.now)
            env.process(customer(env, servers))
    
    def customer(env, servers):
        arrive = env.now
        with servers.request() as req:
            queue_lengths.append(len(servers.queue))
            system_lengths.append(len(servers.queue) + servers.count)
            
            yield req
            wait = env.now - arrive
            wait_times.append(wait)
            
            service_time = random.expovariate(mu_rate)
            yield env.timeout(service_time)
            
            response_times.append(env.now - arrive)
    
    env = simpy.Environment()
    servers = simpy.Resource(env, capacity=num_servers)
    env.process(arrival(env, servers))
    env.run(until=sim_time)

    # === CÁLCULO TEÓRICO CORREGIDO ===
    rho = lambda_rate / (num_servers * mu_rate)
    stable = rho < 1

    if stable and lambda_rate > 0 and mu_rate > 0:
        r = lambda_rate / mu_rate
        sum_term = sum(r**k / math.factorial(k) for k in range(num_servers))
        last_term = r**num_servers / (math.factorial(num_servers) * (1 - rho))
        P0 = 1 / (sum_term + last_term)

        Lq = P0 * (r**num_servers) * rho / (math.factorial(num_servers) * (1 - rho)**2)
        Wq = Lq / lambda_rate
        W = Wq + 1/mu_rate
        L = lambda_rate * W
    else:
        P0 = Lq = Wq = W = L = np.nan

    Lq_sim = np.mean(queue_lengths) if queue_lengths else 0
    Wq_sim = np.mean(wait_times) if wait_times else 0
    L_sim = Lq_sim + rho * num_servers

    return {
        'times': arrivals[:len(queue_lengths)],  # para que coincidan longitudes
        'queue_lengths': queue_lengths,
        'wait_times': wait_times,
        'response_times': response_times,
        'rho': rho,
        'stable': stable,
        'Lq_theory': Lq,
        'Wq_theory': Wq,
        'L_theory': L,
        'W_theory': W,
        'Lq_sim': Lq_sim,
        'Wq_sim': Wq_sim,
        'L_sim': L_sim,
    }