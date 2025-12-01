# app.py
import os
import sys
import json
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # backend sin ventana
import matplotlib.pyplot as plt

from flask import Flask, render_template, jsonify, request, Response

# --------------------------
# Configuración básica
# --------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_DIR = os.path.join(BASE_DIR, "file")
PLOTS_DIR = os.path.join(BASE_DIR, "static", "plots")
EXPERIMENTS_FILE = os.path.join(FILE_DIR, "experiments.json")
WIRESHARK_CSV = os.path.join(FILE_DIR, "wireshark.csv")

os.makedirs(FILE_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

app = Flask(__name__)

server_process = None
client_process = None


# --------------------------
# Utilidades de datos
# --------------------------
def cargar_datos_server_client():
    server_csv = os.path.join(FILE_DIR, "tiempos_servicio.csv")
    client_csv = os.path.join(FILE_DIR, "datos_cliente.csv")

    if not (os.path.exists(server_csv) and os.path.exists(client_csv)):
        return None, None, False

    df_server = pd.read_csv(server_csv)
    df_client = pd.read_csv(client_csv)
    return df_server, df_client, True


def calcular_metricas_basicas(df_server, df_client):
    # ordenar llegadas por si acaso
    df_server_sorted = df_server.sort_values("timestamp_llegada").reset_index(drop=True)
    llegadas = df_server_sorted["timestamp_llegada"].values
    interarrivals_server = np.diff(llegadas)

    interarrival_prom_server = interarrivals_server.mean()
    lambda_hat_server = 1.0 / interarrival_prom_server

    interarrival_prom_client = df_client["interarrival"].mean()
    lambda_hat_client = 1.0 / interarrival_prom_client

    servicio_prom = df_server["tiempo_servicio"].mean()
    mu_hat = 1.0 / servicio_prom

    rho_hat = lambda_hat_server / mu_hat

    Wq_emp = df_server["tiempo_espera_cola"].mean()
    W_emp = df_server["tiempo_en_sistema"].mean()

    df_client_ok = df_client[df_client["t_respuesta"].notna()].copy()
    W_client_emp = df_client_ok["t_respuesta"].mean() if not df_client_ok.empty else np.nan

    if rho_hat < 1:
        Wq_teo = rho_hat / (mu_hat - lambda_hat_server)
        W_teo = 1.0 / (mu_hat - lambda_hat_server)
        Lq_teo = lambda_hat_server * Wq_teo
        L_teo = lambda_hat_server * W_teo
    else:
        Wq_teo = W_teo = Lq_teo = L_teo = np.nan

    L_emp = lambda_hat_server * W_emp
    Lq_emp = lambda_hat_server * Wq_emp

    metricas = {
        "lambda_hat_server": float(lambda_hat_server),
        "lambda_hat_client": float(lambda_hat_client),
        "mu_hat": float(mu_hat),
        "rho_hat": float(rho_hat),
        "Wq_emp": float(Wq_emp),
        "W_emp": float(W_emp),
        "W_client_emp": float(W_client_emp) if not np.isnan(W_client_emp) else None,
        "L_emp": float(L_emp),
        "Lq_emp": float(Lq_emp),
        "Wq_teo": float(Wq_teo),
        "W_teo": float(W_teo),
        "Lq_teo": float(Lq_teo),
        "L_teo": float(L_teo),
        "interarrivals_server": interarrivals_server,
    }
    return metricas


# --------------------------
# Simulación M/M/1
# --------------------------
def simular_mm1(lambda_, mu_, n_clientes=1000, seed=42):
    rng = np.random.default_rng(seed)

    t = 0.0
    next_arrival = rng.exponential(1.0 / lambda_)
    next_departure = np.inf
    queue = 0
    server_busy = False

    arrival_times = []
    service_start_times = []
    departure_times = []

    while len(departure_times) < n_clientes:
        if next_arrival <= next_departure:
            t = next_arrival
            arrival_times.append(t)
            queue += 1
            next_arrival = t + rng.exponential(1.0 / lambda_)

            if not server_busy:
                server_busy = True
                service_start_times.append(t)
                st = rng.exponential(1.0 / mu_)
                next_departure = t + st
        else:
            t = next_departure
            queue -= 1
            departure_times.append(t)
            if queue > 0:
                service_start_times.append(t)
                st = rng.exponential(1.0 / mu_)
                next_departure = t + st
            else:
                server_busy = False
                next_departure = np.inf

    arrival_arr = np.array(arrival_times[:n_clientes])
    depart_arr = np.array(departure_times[:n_clientes])
    start_arr = np.array(service_start_times[:n_clientes])

    W = depart_arr - arrival_arr
    Wq = start_arr - arrival_arr

    W_sim = float(W.mean())
    Wq_sim = float(Wq.mean())
    L_sim = float(lambda_ * W_sim)
    Lq_sim = float(lambda_ * Wq_sim)

    return {
        "W_sim": W_sim,
        "Wq_sim": Wq_sim,
        "L_sim": L_sim,
        "Lq_sim": Lq_sim,
    }


# --------------------------
# Análisis Wireshark
# --------------------------
def analizar_wireshark():
    if not os.path.exists(WIRESHARK_CSV):
        return None

    try:
        dfw = pd.read_csv(WIRESHARK_CSV)
    except Exception:
        return None

    columnas = dfw.columns.str.lower()
    col_time = None
    col_info = None

    for c in dfw.columns:
        cl = c.lower()
        if col_time is None and "time" in cl:
            col_time = c
        if "info" in cl:
            col_info = c

    if col_time is None:
        return None

    dfw = dfw.sort_values(col_time).reset_index(drop=True)

    if col_info is not None:
        mask_req = dfw[col_info].astype(str).str.contains("POST", case=False, na=False)
        df_req = dfw[mask_req].copy()
    else:
        df_req = dfw.copy()

    if len(df_req) < 2:
        return None

    times = df_req[col_time].astype(float).values
    interarrivals = np.diff(times)
    interarrivals = interarrivals[interarrivals > 0]

    if len(interarrivals) == 0:
        return None

    lambda_w = 1.0 / interarrivals.mean()

    stats = {
        "lambda_wireshark": float(lambda_w),
        "interarrivals_w": interarrivals,
        "n_packets": int(len(df_req)),
    }
    return stats


# --------------------------
# Historial de experimentos
# --------------------------
def load_experiments():
    if not os.path.exists(EXPERIMENTS_FILE):
        return []
    try:
        with open(EXPERIMENTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []


def save_experiments(exps):
    with open(EXPERIMENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(exps, f, ensure_ascii=False, indent=2)


# --------------------------
# Generación de gráficas PNG
# --------------------------
def generar_plots(df_server, df_client, metricas, wireshark_stats):
    plot_files = []

    # 1) Hist tiempos de servicio
    servicios = df_server["tiempo_servicio"].values
    mu_hat = metricas["mu_hat"]
    x_vals = np.linspace(0, max(servicios.max(), 1.0), 200)
    pdf_exp = mu_hat * np.exp(-mu_hat * x_vals)

    plt.figure()
    plt.hist(servicios, bins=20, density=True, alpha=0.7, label="Empírico")
    plt.plot(x_vals, pdf_exp, "r-", label="Exp(μ̂)")
    plt.title("Tiempos de servicio vs exponencial")
    plt.xlabel("Tiempo de servicio")
    plt.ylabel("Densidad")
    plt.legend()
    fname1 = "hist_tiempo_servicio.png"
    plt.savefig(os.path.join(PLOTS_DIR, fname1), bbox_inches="tight")
    plt.close()
    plot_files.append((fname1, "Tiempos de servicio vs exponencial"))

    # 2) Hist interarrivals cliente
    inter_client = df_client["interarrival"].values
    lambda_hat_client = metricas["lambda_hat_client"]
    x_vals2 = np.linspace(0, max(inter_client.max(), 1.0), 200)
    pdf_exp_client = lambda_hat_client * np.exp(-lambda_hat_client * x_vals2)

    plt.figure()
    plt.hist(inter_client, bins=20, density=True, alpha=0.7, label="Interarrivals cliente")
    plt.plot(x_vals2, pdf_exp_client, "r-", label="Exp(λ̂_cliente)")
    plt.title("Interarrivals del cliente vs exponencial")
    plt.xlabel("Tiempo entre llegadas")
    plt.ylabel("Densidad")
    plt.legend()
    fname2 = "hist_interarrivals_cliente.png"
    plt.savefig(os.path.join(PLOTS_DIR, fname2), bbox_inches="tight")
    plt.close()
    plot_files.append((fname2, "Interarrivals del cliente vs exponencial"))

    # 3) Serie tiempo en sistema
    plt.figure()
    plt.plot(range(len(df_server)), df_server["tiempo_en_sistema"], marker="o", linestyle="-")
    plt.title("Tiempo en el sistema por cliente (server)")
    plt.xlabel("Cliente (orden)")
    plt.ylabel("Tiempo en sistema")
    fname3 = "serie_tiempo_sistema_server.png"
    plt.savefig(os.path.join(PLOTS_DIR, fname3), bbox_inches="tight")
    plt.close()
    plot_files.append((fname3, "Tiempo en el sistema por cliente (server)"))

    # 4) Boxplot tiempos
    plt.figure()
    data_box = [
        df_server["tiempo_espera_cola"],
        df_server["tiempo_en_sistema"],
        df_server["tiempo_servicio"],
    ]
    plt.boxplot(data_box, labels=["Espera en cola", "Tiempo en sistema", "Servicio"])
    plt.title("Distribución de tiempos (server)")
    plt.ylabel("Tiempo")
    fname4 = "boxplot_tiempos_server.png"
    plt.savefig(os.path.join(PLOTS_DIR, fname4), bbox_inches="tight")
    plt.close()
    plot_files.append((fname4, "Distribución de tiempos (server)"))

    # 5) Hist interarrivals Wireshark (si existe)
    if wireshark_stats is not None:
        iw = wireshark_stats["interarrivals_w"]
        lambda_w = wireshark_stats["lambda_wireshark"]
        xw = np.linspace(0, max(iw.max(), 1.0), 200)
        pdf_w = lambda_w * np.exp(-lambda_w * xw)

        plt.figure()
        plt.hist(iw, bins=20, density=True, alpha=0.7, label="Interarrivals Wireshark")
        plt.plot(xw, pdf_w, "r-", label="Exp(λ̂_Wireshark)")
        plt.title("Interarrivals observados por Wireshark")
        plt.xlabel("Δt entre requests")
        plt.ylabel("Densidad")
        plt.legend()
        fname5 = "hist_interarrivals_wireshark.png"
        plt.savefig(os.path.join(PLOTS_DIR, fname5), bbox_inches="tight")
        plt.close()
        plot_files.append((fname5, "Interarrivals observados por Wireshark"))

    return plot_files


# --------------------------
# Rutas principales
# --------------------------
@app.route("/")
def index():
    return """
    <h1>Dashboard de Proyectos</h1>
    <ul>
        <li><a href="/mm1">Proyecto 1: Sistema M/M/1</a></li>
    </ul>
    """


@app.route("/mm1")
def mm1_dashboard():
    df_server, df_client, data_available = cargar_datos_server_client()

    metricas = None
    sim_results = None
    wireshark_stats = None
    plot_files = []

    lambda_teo = request.args.get("lambda_teo", type=float)
    mu_teo = request.args.get("mu_teo", type=float)
    n_sim = request.args.get("n_sim", type=int, default=1000)

    if data_available:
        metricas = calcular_metricas_basicas(df_server, df_client)

        if lambda_teo is None:
            lambda_teo = metricas["lambda_hat_server"]
        if mu_teo is None:
            mu_teo = metricas["mu_hat"]

        sim_results = simular_mm1(lambda_teo, mu_teo, n_clientes=n_sim)
        wireshark_stats = analizar_wireshark()
        plot_files = generar_plots(df_server, df_client, metricas, wireshark_stats)

    experiments = load_experiments()
    experiments = experiments[-10:]

    return render_template(
        "mm1.html",
        data_available=data_available,
        metricas=metricas,
        sim_results=sim_results,
        wireshark_stats=wireshark_stats,
        lambda_teo=lambda_teo,
        mu_teo=mu_teo,
        n_sim=n_sim,
        experiments=experiments,
        plot_files=plot_files,
    )


# --------------------------
# API: iniciar server/cliente
# --------------------------
@app.route("/api/start_server", methods=["POST"])
def api_start_server():
    global server_process
    if server_process is None or server_process.poll() is not None:
        try:
            env = os.environ.copy()
            env.pop("WERKZEUG_SERVER_FD", None)
            env.pop("WERKZEUG_RUN_MAIN", None)

            server_process = subprocess.Popen(
                [sys.executable, "server.py"],
                cwd=BASE_DIR,
                env=env,
            )
            return jsonify(success=True, message="Servidor iniciado (server.py).")
        except Exception as e:
            return jsonify(success=False, message=f"Error al iniciar servidor: {e}")
    else:
        return jsonify(success=True, message="Servidor ya está corriendo.")


@app.route("/api/start_client", methods=["POST"])
def api_start_client():
    global client_process
    if client_process is None or client_process.poll() is not None:
        try:
            client_process = subprocess.Popen(
                [sys.executable, "client.py"],
                cwd=BASE_DIR,
            )
            return jsonify(success=True, message="Cliente iniciado (client.py).")
        except Exception as e:
            return jsonify(success=False, message=f"Error al iniciar cliente: {e}")
    else:
        return jsonify(success=True, message="Cliente ya está corriendo.")


# --------------------------
# API: guardar experimento
# --------------------------
@app.route("/api/save_experiment", methods=["POST"])
def api_save_experiment():
    data = request.get_json(force=True)

    experiments = load_experiments()
    data["timestamp"] = datetime.now().isoformat(timespec="seconds")
    experiments.append(data)
    save_experiments(experiments)

    return jsonify(success=True, message="Experimento guardado.")


# --------------------------
# Exportar historial como CSV
# --------------------------
@app.route("/experiments/export")
def export_experiments():
    experiments = load_experiments()
    if not experiments:
        return Response("No hay experimentos.", mimetype="text/plain")

    df = pd.DataFrame(experiments)
    csv_data = df.to_csv(index=False)

    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=experiments.csv"},
    )


if __name__ == "__main__":
    app.run(debug=True)
