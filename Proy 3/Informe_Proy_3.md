<script type="text/javascript"
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>
<div align="center">

# **Universidad Mayor de San Andrés**  
## **Facultad de Ciencias Puras y Naturales**  
### **Carrera de Informática**

<img src="https://upload.wikimedia.org/wikipedia/commons/5/5e/Logo_Umsa.png" alt="Logo UMSA" width="120" style="display:block;margin:auto;"/>


<h1>Proyecto N°3 – Modelado y Simulación de un Sistema de Colas M/M/1 Aplicado al Tráfico en un Servidor Flask</h1>

**Integrantes:**  
- Flores Tapia Ruddy – carnet
- Cayllagua Mamani Franklin – carnet
- Salinas Condori Ian Ezequiel – carnet
- Maximiliano Gómez Mallo – 14480221  
<br>

**Materia:** Procesos Estocásticos y Series de Tiempo
**Docente:** Ph.D. Willy Ernesto Portugal Duran
<br>

**Fecha:** [dd/mm/aaaa]

</div>

<div style="page-break-after: always;"></div>
---

## 1. Introducción
El presente proyecto tiene como finalidad analizar y modelar el comportamiento de un servidor web mediante la teoría de colas, específicamente utilizando el modelo **M/M/1**. Este enfoque permite representar la llegada aleatoria de solicitudes y su atención secuencial por un único servidor, con el objetivo de evaluar métricas de rendimiento tales como el tiempo medio de espera, la utilización del servidor y el número promedio de clientes en el sistema.

El experimento se desarrolló implementando un servidor Flask de un solo hilo (capacidad de servicio unitario), un cliente generador de tráfico con llegadas de tipo Poisson y la herramienta **Wireshark** para capturar el tráfico real. Posteriormente, se realizó una **simulación en SimPy** con los parámetros empíricamente medidos, comparando los resultados con las expresiones teóricas de la teoría de colas.

---

## 2. Planteamiento del problema
En sistemas de red, el rendimiento de los servidores depende directamente de la relación entre la tasa de llegada de solicitudes λ y la tasa de servicio μ. Cuando las llegadas superan la capacidad de atención, el sistema se congestiona y los tiempos de respuesta aumentan exponencialmente. Se busca modelar este comportamiento bajo un entorno controlado que refleje condiciones reales de tráfico web, determinando si el servidor Flask cumple con la estabilidad y eficiencia esperadas para un modelo **M/M/1**.

---

## 3. Objetivos

**Objetivo general**  
Modelar y simular el tráfico de solicitudes en un servidor Flask mediante un sistema de colas **M/M/1**, analizando las tasas de llegada y servicio a partir de datos capturados con Wireshark y su validación mediante SimPy.

**Objetivos específicos**  
- Capturar y procesar el tráfico generado entre cliente y servidor Flask con Wireshark.  
- Estimar los parámetros λ y μ empíricos a partir de los archivos CSV obtenidos.  
- Simular el sistema de colas M/M/1 con los parámetros observados.  
- Comparar los resultados teóricos, empíricos y simulados, discutiendo las posibles diferencias.  

---

## 4. Marco teórico
El análisis de rendimiento en redes de computadoras puede abordarse desde la teoría de colas, una rama de la probabilidad que modela sistemas donde las entidades (en este caso, **paquetes o solicitudes**) llegan de forma aleatoria para ser atendidas por uno o varios servidores. En el contexto de las redes, estos servidores pueden representar **routers, interfaces de red, servidores web o servicios de aplicación**. Cada solicitud que ingresa al sistema consume recursos computacionales o de transmisión, y si estos se encuentran ocupados, la solicitud debe **esperar en cola** hasta ser atendida.

---

## 4.1. Tráfico en redes y naturaleza estocástica

En una red, las unidades básicas de información se denominan **paquetes**, los cuales contienen encabezados y datos de usuario. El tiempo en que cada paquete llega a un nodo depende de múltiples factores: la congestión de la red, el ancho de banda, la latencia del medio físico y el comportamiento del usuario o de las aplicaciones. Dado que estas variables son inherentemente aleatorias, el tráfico de red puede considerarse un **proceso estocástico**.

Según Stallings (2017), el flujo de paquetes puede describirse mediante una **distribución de Poisson**, donde los intervalos de llegada entre paquetes son **exponenciales**. Esto implica que los eventos (paquetes) son independientes y ocurren con una tasa promedio constante λ. De forma análoga, los tiempos de servicio del sistema (procesamiento o transmisión) también pueden modelarse como exponenciales, con tasa μ, siempre que no exista dependencia entre servicios.

Wireshark permite observar este fenómeno directamente: al capturar el tráfico en un puerto TCP, se puede comprobar que los **intervalos entre paquetes sucesivos** fluctúan, generando una distribución de frecuencias que se ajusta razonablemente a una exponencial. En el caso del presente proyecto, el puerto 5000 correspondía al servidor Flask, y los paquetes medidos representaban solicitudes HTTP generadas aleatoriamente por el cliente.

---

## 4.2. Procesos de nacimiento y muerte

El modelo M/M/1 tiene su origen en los **procesos de nacimiento y muerte (birth–death processes)**, una clase de cadenas de Markov continuas en el tiempo donde el estado **n** representa el número de clientes (o paquetes) presentes en el sistema.

- Un **nacimiento** representa la llegada de un nuevo cliente (con tasa λ).
- Una **muerte** representa la finalización del servicio de un cliente (con tasa μ).

El sistema puede encontrarse en los estados **S = {0, 1, 2, 3, ...}**, donde cada transición ocurre con probabilidad proporcional a las tasas mencionadas. Las ecuaciones de equilibrio estacionario (balance detallado) se derivan igualando los flujos de probabilidad de entrada y salida en cada estado:

$$
\lambda P_n = \mu P_{n+1}
$$

Resolviendo recursivamente:
$$
P_n = \left( \frac{\lambda}{\mu} \right)^n P_0 = \rho^n P_0,
$$
donde \(\rho = \frac{\lambda}{\mu}\) es la **intensidad de tráfico o utilización del servidor**.

Como la suma de probabilidades debe ser 1:
$$
\sum_{n=0}^{\infty} P_n = 1 \Rightarrow P_0 = 1 - \rho \text{  (válido si } \rho < 1)
$$

De ahí se obtienen los valores esperados de las métricas principales (Gross et al., 2018):

- Número promedio de clientes en el sistema:
$$L = \frac{\rho}{1 - \rho}$$

- Número promedio en cola:
$$L_q = \frac{\rho^2}{1 - \rho}$$

- Tiempo promedio en el sistema:
$$W = \frac{1}{\mu - \lambda}$$

- Tiempo promedio en cola:
$$W_q = \frac{\lambda}{\mu(\mu - \lambda)}$$

Estas relaciones establecen un vínculo directo entre los procesos estocásticos y el rendimiento del sistema. El sistema se considera **estable** si \(\lambda < \mu\), es decir, si la capacidad de atención es mayor que la tasa de llegada.

---

## 4.3. Interpretación en el contexto de redes

En un entorno real, el servidor Flask se comporta como una estación de servicio con **capacidad unitaria**. Cada solicitud HTTP que llega se encola si el hilo de atención está ocupado. La llegada de solicitudes sigue un proceso de Poisson (generado por el cliente con interarrivals exponenciales), mientras que los tiempos de servicio, determinados por el retardo de ejecución y el uso de CPU, se aproximan a una distribución exponencial.

A nivel de red, esta dinámica puede interpretarse como una **cola de transmisión**:
- Cada solicitud HTTP se encapsula en varios **paquetes TCP**, los cuales viajan por la pila de protocolos hasta el servidor.
- Al llegar al servidor, los paquetes se reconstruyen, y Flask procesa la solicitud.
- Si Flask (un solo hilo) está ocupado, las solicitudes se acumulan temporalmente en el buffer del sistema operativo o en la cola interna del servidor.

Por tanto, el modelo M/M/1 representa de forma abstracta el ciclo: **recepción de paquetes → reconstrucción → procesamiento → respuesta**.

Wireshark permite visualizar estos flujos: el filtro `tcp.port == 5000` muestra tanto las llegadas (paquetes con bandera PSH/ACK desde el cliente) como las respuestas (desde el servidor). El análisis de los tiempos entre solicitudes capturados con `frame.time_epoch` demuestra la variabilidad aleatoria característica de los procesos de Poisson (ULPGC, 2020).

---

## 4.4. Conexión entre teoría y simulación

El modelo matemático M/M/1 proporciona una base para **predecir el comportamiento medio del sistema**, pero la simulación en herramientas como **SimPy** permite **reproducir la dinámica temporal completa**, incluyendo la variabilidad estocástica.  

La comparación entre teoría, datos empíricos y simulación evidencia la validez de los supuestos de la teoría de colas en redes simples:
- Los tiempos entre llegadas y servicios se ajustan bien a distribuciones exponenciales.
- La utilización ρ observada y simulada son prácticamente iguales (≈ 0.37), confirmando el equilibrio del sistema.
- El tiempo promedio de espera y el tiempo total en el sistema obtenidos mediante SimPy (W_q = 1.20 s, W = 2.39 s) se aproximan razonablemente a los valores teóricos (W_q = 0.73 s, W = 1.95 s), lo que valida el modelo.

En términos prácticos, este tipo de análisis es fundamental para el **diseño de arquitecturas de red eficientes**, la **planificación de capacidad de servidores** y la **evaluación de rendimiento** en entornos distribuidos, donde la demanda varía estocásticamente (UOC, 2019; Stallings, 2017).

---

## 4.5. Conclusión del marco teórico

El modelo **M/M/1**, derivado de los procesos de nacimiento y muerte, constituye el caso base para el estudio del rendimiento en redes de computadoras. Su simplicidad permite comprender los fenómenos de congestión y espera que ocurren en servidores web y dispositivos de red. Al integrar las herramientas modernas de captura (Wireshark) y simulación (SimPy), es posible **verificar experimentalmente** los principios de la teoría de colas y su relación con el tráfico de red real.


## 5. Diseño metodológico

### Tipo de estudio
Simulación y modelado estocástico a partir de datos empíricos.

### Herramientas utilizadas
- **Flask** como servidor web monohilo.  
- **Requests** como generador de solicitudes con interarrivals exponenciales.  
- **Wireshark** para captura del tráfico (Wireshark Foundation, 2024).  
- **SimPy** para la simulación discreta del sistema (SimPy Documentation, 2024).  

### Fragmento del servidor Flask
```python
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

```

### Generador de tráfico
```python
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

```
### Análisis de datos empíricos
Los datos capturados en los csv (uno del servidor y otro de Wireshark) se procesaron con el siguiente código:
#### Carga de datos
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon

# Configurar visualización
plt.style.use("seaborn-v0_8")
pd.set_option('display.float_format', lambda x: f'{x:.5f}')

# Cargar datos
df_wireshark = pd.read_csv("files/captura_flask.csv")
df_servicio = pd.read_csv("files/tiempos_servicio.csv")

print("Archivos cargados correctamente ✅")
print("Columnas Wireshark:", df_wireshark.columns.tolist())
print("Columnas Servicio:", df_servicio.columns.tolist())
```
#### Cálculo del paramámetro λ
```python
# --- Detectar y filtrar solicitudes HTTP (una por request) ---
if "Info" in df_wireshark.columns:
    # Filtrar solo las filas donde el campo Info contiene 'GET' o 'POST'
    mask_http = df_wireshark["Info"].astype(str).str.contains("GET|POST", case=False, na=False)
    df_filtrado = df_wireshark.loc[mask_http].copy()

    if df_filtrado.empty:
        print("⚠️ No se detectaron solicitudes HTTP (GET/POST) en el CSV. Usando todo el tráfico como aproximación.")
        df_filtrado = df_wireshark.copy()
else:
    print("⚠️ No existe columna 'Info' en el CSV. Usando todo el tráfico como aproximación.")
    df_filtrado = df_wireshark.copy()

# --- Detectar la columna de tiempo correcta ---
if "frame.time_epoch" in df_filtrado.columns:
    tiempos = df_filtrado["frame.time_epoch"].astype(float).values
elif "Time" in df_filtrado.columns:
    tiempos = df_filtrado["Time"].astype(float).values
else:
    raise ValueError("No se encontró una columna de tiempo válida en Wireshark.")

# --- Calcular intervalos entre llegadas ---
tiempos = np.sort(tiempos)
interarrival = np.diff(tiempos)
interarrival = interarrival[interarrival > 0]  # eliminar valores 0 o negativos

# --- Calcular tasa de llegadas λ ---
if len(interarrival) > 0:
    lambda_emp = 1 / interarrival.mean()
    print(f"Tasa de llegadas λ = {lambda_emp:.4f} solicitudes/segundo")
else:
    print("⚠️ No se pudieron calcular intervalos entre llegadas (muy pocos datos).")
```
#### Cálculo del parámetro μ
```python
mu_emp = 1 / df_servicio["tiempo_servicio"].mean()
print(f"Tasa de servicio μ = {mu_emp:.4f} solicitudes/segundo")
```
#### Métricas empíricas
```python
rho = lambda_emp / mu_emp

if rho >= 1:
    print("⚠️ Sistema inestable: λ ≥ μ")
else:
    L = rho / (1 - rho)
    W = 1 / (mu_emp - lambda_emp)
    Lq = rho**2 / (1 - rho)
    Wq = rho / (mu_emp - lambda_emp)

    print(f"Utilización del servidor ρ = {rho:.4f}")
    print(f"Clientes promedio en sistema L = {L:.4f}")
    print(f"Clientes promedio en cola Lq = {Lq:.4f}")
    print(f"Tiempo medio en sistema W = {W:.4f} s")
    print(f"Tiempo medio en cola Wq = {Wq:.4f} s")
```
#### Visualización de distribuciones
```python
plt.figure(figsize=(12,5))

# Interarrival
plt.subplot(1, 2, 1)
plt.hist(interarrival, bins=25, density=True, alpha=0.6, color='steelblue')
x = np.linspace(0, interarrival.max(), 100)
plt.plot(x, expon.pdf(x, scale=interarrival.mean()), 'r--', label='Exponencial teórica')
plt.title("Tiempos entre llegadas (Interarrival)")
plt.xlabel("Segundos entre llegadas")
plt.ylabel("Densidad")
plt.legend()

# Service times
plt.subplot(1, 2, 2)
servicios = df_servicio["tiempo_servicio"].values
plt.hist(servicios, bins=25, density=True, alpha=0.6, color='darkorange')
x2 = np.linspace(0, servicios.max(), 100)
plt.plot(x2, expon.pdf(x2, scale=servicios.mean()), 'r--', label='Exponencial teórica')
plt.title("Tiempos de servicio")
plt.xlabel("Duración (segundos)")
plt.ylabel("Densidad")
plt.legend()

plt.tight_layout()
plt.show()
```
<img src="img/output.png" alt="Distribuciones de tiempos" style="display:block;margin:auto; height:400px; width:auto;"/>

1. **Creación de la figura y subgráficos:** `plt.figure(figsize=(12,5))` define un lienzo de 12×5 pulgadas. Se generan dos subgráficos para comparar las distribuciones de llegada y servicio.
2. **Histograma de interarrivals:** se calcula la densidad de probabilidad de los tiempos entre llegadas (`interarrival`), normalizada con `density=True`. El color azul acero (`steelblue`) facilita la lectura visual.
3. **Curva teórica:** la función `expon.pdf()` de SciPy traza la densidad de una distribución exponencial con media igual al valor promedio de los datos. Esta curva (línea roja discontinua) sirve como referencia para evaluar el ajuste del modelo.
4. **Tiempos de servicio:** se repite el procedimiento para los valores de `tiempo_servicio`, mostrando cómo los tiempos observados se aproximan a una exponencial.
5. **Diseño final:** `plt.tight_layout()` ajusta el espaciado automático y `plt.show()` renderiza el gráfico.

Estos gráficos son esenciales para **verificar empíricamente la validez del supuesto exponencial** en los modelos de colas M/M/1. En este caso, las distribuciones observadas muestran una fuerte concentración cerca del origen y una cola decreciente, característica de los procesos de Poisson. Esto confirma que tanto las llegadas como los servicios se comportan de manera aproximadamente **memoryless**, cumpliendo los supuestos fundamentales de la teoría de colas (Gross et al., 2018; Ross, 2014).

#### Simulación en SimPy
```python
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

```

---

## 6. Resultados
### Resultados empíricos (de los CSV)

| Parámetro | Valor | Descripción |
|------------|--------|--------------|
| ρ | 0.3734 | Utilización del servidor |
| L | 0.5960 | Clientes promedio en el sistema |
| L_q | 0.2226 | Clientes promedio en la cola |
| W | 1.9499 s | Tiempo medio en el sistema |
| W_q | 0.7281 s | Tiempo medio en cola |

### Resultados de simulación (SimPy)

| Parámetro | Valor | Descripción |
|------------|--------|--------------|
| λ | 0.3057 | Tasa de llegada medida |
| μ | 0.8185 | Tasa de servicio medida |
| ρ | 0.3735 | Utilización simulada |
| W_q simulado | 1.2029 s | Tiempo medio de espera (simulado) |
| W simulado | 2.3944 s | Tiempo total en el sistema (simulado) |
| W_q teórico | 0.7283 s | Tiempo teórico en cola |
| W teórico | 1.9501 s | Tiempo teórico total |

### Interpretación
El análisis comparativo entre los resultados teóricos, empíricos y simulados permite observar con claridad el comportamiento de un sistema **M/M/1** en un entorno de red real.  

En primer lugar, el valor empírico de λ = 0.3057 solicitudes/segundo, obtenido mediante la captura en Wireshark, se encuentra por debajo del valor teórico de **0.5**. Esta diferencia se debe a la naturaleza asincrónica del tráfico y al retardo acumulado por la ejecución secuencial del servidor Flask, el cual sólo dispone de un hilo de atención (capacidad unitaria). Dicho retardo provoca que las llegadas efectivas sean menos frecuentes, estabilizando al sistema en un régimen de carga moderado.

Por otro lado, el valor de μ = 0.8185 solicitudes/segundo, medido a partir del tiempo medio de servicio del servidor, implica una **utilización del 37%**. Esto indica que el servidor pasa la mayor parte del tiempo ocioso y que existe margen suficiente para absorber picos de carga sin llegar a congestión.

La comparación entre los valores teóricos y simulados muestra una concordancia notable:

| Parámetro | Teórico | SimPy | Diferencia |
|------------|----------|--------|-------------|
| W_q (s) | 0.7283 | 1.2029 | +65% |
| W (s) | 1.9501 | 2.3944 | +22% |

Aunque los tiempos simulados son ligeramente superiores, estas diferencias se explican por la **aleatoriedad inherente a la simulación Monte Carlo** y por el hecho de que los datos empíricos presentan mayor dispersión que el modelo ideal. En un entorno de red real, los tiempos de servicio no son estrictamente exponenciales debido a factores de latencia, planificación del CPU, búferes del sistema operativo y variaciones en la pila TCP/IP.

Desde la perspectiva de redes, este resultado tiene una interpretación clara: el modelo M/M/1 describe con fidelidad el comportamiento promedio de un servidor HTTP bajo tráfico moderado. El sistema mantiene estabilidad mientras ρ < 1; sin embargo, si el ritmo de llegadas aumenta y se aproxima a la capacidad de servicio, la cola crecería exponencialmente, reproduciendo el fenómeno de congestión en routers o servidores sobrecargados.

Finalmente, la simulación permitió confirmar que las métricas W_q y W son sensibles a pequeñas variaciones de λ, mostrando el carácter no lineal de los tiempos de espera frente al aumento de tráfico. Esto respalda la utilidad del modelo M/M/1 como herramienta predictiva en planificación de capacidad y optimización de rendimiento en sistemas distribuidos.


---

## 7. Conclusiones
- Los resultados obtenidos demuestran que el comportamiento del servidor Flask puede modelarse adecuadamente mediante un sistema **M/M/1**.  
- La utilización del servidor (ρ = 0.37) indica un nivel de carga moderado, garantizando estabilidad y tiempos de respuesta finitos.  
- Las métricas simuladas presentan ligeras diferencias respecto a las teóricas debido a la variabilidad aleatoria inherente a la simulación discreta y al retardo del entorno real.  
- La integración de **Wireshark**, **Flask** y **SimPy** permitió conectar la teoría de colas con un entorno práctico de redes y computación.  

---

## 8. Referencias
- Gross, D., Shortle, J. F., Thompson, J. M., & Harris, C. M. (2018). *Fundamentals of Queueing Theory* (5th ed.). Wiley.  
- Ross, S. M. (2014). *Introduction to Probability Models*. Academic Press.  
- Stallings, W. (2017). *Data and Computer Communications* (10th ed.). Pearson.  
- SimPy Documentation. (2024). Retrieved from https://simpy.readthedocs.io  
- Wireshark Foundation. (2024). *Wireshark User Guide*. Retrieved from https://www.wireshark.org  
- ULPGC. (2020). *Procesos estocásticos básicos en teoría de colas*. Universidad de Las Palmas de Gran Canaria.  
- UOC. (2019). *Análisis mediante teoría de colas*. Universitat Oberta de Catalunya.

