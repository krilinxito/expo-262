from simulador.simulacion import simular_mm_c
from simulador.teorico import Wq_teorico, W_teorico
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Parámetros
LAMBDA = 0.3057
MU = 0.8185
TIEMPO = 200
Cs = [1, 2, 3, 4, 5]

Wq_sims, W_sims = [], []
Wq_teos, W_teos = [], []

# Crear carpetas si no existen
os.makedirs("graficos", exist_ok=True)
os.makedirs("datos_simulacion", exist_ok=True)

print("\n==============================")
print("     RESULTADOS SIMULACIÓN    ")
print("==============================\n")

for c in Cs:
    print(f"\n---- Simulando M/M/{c} ----")

    Wq_sim, W_sim, tiempos_espera, tiempos_sistema = simular_mm_c(
        LAMBDA, MU, c, TIEMPO
    )

    # Guardar promedios
    Wq_sims.append(Wq_sim)
    W_sims.append(W_sim)

    Wq_teo = Wq_teorico(LAMBDA, MU, c)
    W_teo = W_teorico(LAMBDA, MU, c)

    Wq_teos.append(Wq_teo)
    W_teos.append(W_teo)

    # --- Asegurar que las listas tengan la misma longitud ---
    n = min(len(tiempos_espera), len(tiempos_sistema))
    tiempos_espera = tiempos_espera[:n]
    tiempos_sistema = tiempos_sistema[:n]

    # Guardar CSV con pandas
    df = pd.DataFrame({
        "tiempo_espera": tiempos_espera,
        "tiempo_sistema": tiempos_sistema
    })

    df.to_csv(f"datos_simulacion/resultados_c{c}.csv", index=False)

    # ---- IMPRIMIR RESULTADOS EN CONSOLA ----
    print(f"λ = {LAMBDA}, μ = {MU}, c = {c}")
    rho = LAMBDA / (c * MU)
    print(f"Utilización ρ = {rho:.4f}")
    print(f"Clientes procesados: {n}")

    print("\n--- Simulación ---")
    print(f"Wq simulado = {Wq_sim:.4f} s")
    print(f"W  simulado = {W_sim:.4f} s")

    print("\n--- Teórico (Erlang C) ---")
    print(f"Wq teórico = {Wq_teo:.4f} s")
    print(f"W  teórico = {W_teo:.4f} s")

    print("\nDiferencias:")
    print(f"|Wq_sim - Wq_teo| = {abs(Wq_sim - Wq_teo):.4f}")
    print(f"|W_sim - W_teo|  = {abs(W_sim - W_teo):.4f}")
    print("----------------------------")

# ====== GRÁFICO Wq ======
plt.plot(Cs, Wq_sims, marker='o', label='Wq simulado')
plt.plot(Cs, Wq_teos, marker='s', label='Wq teórico')
plt.title("Wq vs número de servidores (c)")
plt.xlabel("Número de servidores (c)")
plt.ylabel("Tiempo promedio en cola (Wq)")
plt.grid()
plt.legend()
plt.savefig("graficos/wq_vs_c.png")
plt.show()

# ====== GRÁFICO W ======
plt.plot(Cs, W_sims, marker='o', label='W simulado')
plt.plot(Cs, W_teos, marker='s', label='W teórico')
plt.title("W vs número de servidores (c)")
plt.xlabel("Número de servidores (c)")
plt.ylabel("Tiempo promedio en el sistema (W)")
plt.grid()
plt.legend()
plt.savefig("graficos/w_vs_c.png")
plt.show()

print("\nSimulación completa.\n")
