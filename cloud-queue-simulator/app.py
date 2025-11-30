# app.py - Cloud Queue Simulator M/M/c - Versión FINAL y SIN ERRORES
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from simulator import run_simulation

# ===================== CONFIGURACIÓN DE PÁGINA =====================
st.set_page_config(
    page_title="Cloud Queue Simulator M/M/c",
    page_icon="cloud",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fondo estilo nube + vidrio esmerilado
page_bg = '''
<style>
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)),
                    url("https://images.unsplash.com/photo-1492011221367-f47e3ccd77a0?ixlib=rb-4.0.3&auto=format&fit=crop&q=80");
        background-size: cover;
        background-attachment: fixed;
    }
    .metric-card {
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        padding: 1.2rem;
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.18);
        text-align: center;
        color: white;
    }
    h1, h2, h3 { color: white !important; }
</style>
'''
st.markdown(page_bg, unsafe_allow_html=True)

# ===================== TÍTULO =====================
st.markdown("""
    <h1 style='text-align: center;'>
        Cloud Queue Simulator M/M/c
    </h1>
    <h3 style='text-align: center; color: #cccccc;'>
        Simulación visual de colas en servicios en la nube
    </h3>
    <hr style='border-color: #444;'>
""", unsafe_allow_html=True)

# ===================== SIDEBAR =====================
with st.sidebar:
    st.header("Configuración de la simulación")
    lambda_rate = st.slider("Tasa de llegada (λ) – peticiones/segundo", 10, 400, 120, 10)
    mu_rate = st.slider("Tasa de servicio por servidor (μ)", 10, 100, 30, 5)
    num_servers = st.slider("Número de servidores (c)", 1, 40, 6, 1)
    sim_time = st.slider("Duración de la simulación (segundos)", 120, 1800, 600, 60)
    costo_por_servidor = st.number_input("Costo mensual por servidor ($)", 10, 2000, 120, 10)

    st.markdown("---")
    if st.button("Ejecutar simulación", type="primary", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# ===================== SIMULACIÓN CACHÉ =====================
@st.cache_data(show_spinner="Simulando colas en la nube...")
def ejecutar_simulacion(l, m, c, t):
    return run_simulation(l, m, c, t)

# Ejecutamos solo una vez
if 'results' not in st.session_state:
    with st.spinner("Ejecutando simulación M/M/c..."):
        st.session_state.results = ejecutar_simulacion(lambda_rate, mu_rate, num_servers, sim_time)

results = st.session_state.results
rho = results['rho']
stable = "Estable" if results['stable'] else "Inestable (cola infinita)"

# ===================== MÉTRICAS PRINCIPALES =====================
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"<div class='metric-card'><h3>Utilización ρ</h3><h2>{rho:.1%}</h2><small>{stable}</small></div>", unsafe_allow_html=True)
with col2:
    color = "green" if results['Wq_sim'] < 0.5 else "orange" if results['Wq_sim'] < 2 else "red"
    st.markdown(f"<div class='metric-card'><h3>Tiempo en cola</h3><h2 style='color:{color}'>{results['Wq_sim']:.3f} s</h2></div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='metric-card'><h3>Peticiones en cola</h3><h2>{results['Lq_sim']:.1f}</h2></div>", unsafe_allow_html=True)
with col4:
    costo_total = num_servers * costo_por_servidor
    st.markdown(f"<div class='metric-card'><h3>Costo mensual</h3><h2>${costo_total:,}</h2></div>", unsafe_allow_html=True)

# ===================== PESTAÑAS =====================
tab1, tab2, tab3 = st.tabs(["Animación en tiempo real", "Análisis de escalabilidad", "Costo vs Rendimiento"])

# ------------------- Pestaña 1 -------------------
with tab1:
    st.markdown("#### Evolución de la longitud de cola durante la simulación")
    if len(results['queue_lengths']) > 0:
        df_queue = pd.DataFrame({
            'Tiempo (s)': np.linspace(0, sim_time, len(results['queue_lengths'])),
            'Longitud de cola': results['queue_lengths']
        })
        fig = px.area(df_queue, x='Tiempo (s)', y='Longitud de cola',
                      title=f"Simulación M/M/{num_servers} – λ={lambda_rate}, μ={mu_rate}",
                      template="plotly_dark")
        fig.update_traces(line_color="#00ff88")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No hay datos de cola (puede que el sistema esté vacío).")

# ------------------- Pestaña 2 -------------------
with tab2:
    st.markdown("#### ¿Cuántos servidores necesito realmente?")
    min_servers = max(1, int(lambda_rate / mu_rate) - 2, 1)
    servers_range = list(range(min_servers, num_servers + 15))

    data = []
    for c in servers_range:
        res = run_simulation(lambda_rate, mu_rate, c, sim_time=400, seed=123)
        data.append({
            'Servidores': c,
            'Tiempo en cola (s)': round(res['Wq_sim'], 4),
            'Cola promedio': round(res['Lq_sim'], 2),
            'Utilización ρ': round(res['rho'], 3),
            'Costo mensual ($)': c * costo_por_servidor
        })
    df = pd.DataFrame(data)

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df['Servidores'], y=df['Tiempo en cola (s)'],
                              mode='lines+markers', name='Tiempo en cola (s)',
                              line=dict(color='#ff4444', width=4)))
    fig1.add_trace(go.Scatter(x=df['Servidores'], y=df['Costo mensual ($)'],
                              mode='lines+markers', name='Costo mensual ($)',
                              yaxis='y2', line=dict(color='#44ff88')))

    fig1.update_layout(
        title="Impacto del número de servidores",
        xaxis_title="Número de servidores (c)",
        yaxis_title="Tiempo en cola (segundos)",
        yaxis2=dict(title="Costo mensual ($)", overlaying="y", side="right"),
        template="plotly_dark",
        height=550
    )
    st.plotly_chart(fig1, use_container_width=True)
    st.dataframe(df.style.background_gradient(cmap='RdYlGn_r', subset=['Tiempo en cola (s)']), use_container_width=True)

# ------------------- Pestaña 3 -------------------
with tab3:
    st.markdown("#### Trade-off: Costo vs Rendimiento")
    st.info("Busca el 'codo' de la curva → punto óptimo calidad-precio")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df['Costo mensual ($)'],
        y=df['Tiempo en cola (s)'],
        mode='lines+markers',
        name='Frontera eficiente',
        line=dict(color='#00ffcc', width=5),
        marker=dict(size=10)
    ))

    # Tu configuración actual
    fig2.add_vline(x=num_servers * costo_por_servidor,
                  line=dict(dash="dash", color="yellow", width=3),
                  annotation_text=f" Tu configuración ({num_servers} servidores)",
                  annotation_position="top")

    fig2.update_layout(
        title="Costo mensual vs Tiempo de respuesta",
        xaxis_title="Costo mensual ($)",
        yaxis_title="Tiempo en cola promedio (s)",
        template="plotly_dark",
        height=600
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Tabla con el punto óptimo sugerido
    df_optimo = df[df['Tiempo en cola (s)'] < 0.5].nsmallest(1, 'Costo mensual ($)')
    if not df_optimo.empty:
        opt = df_optimo.iloc[0]
        st.success(f"Punto óptimo sugerido: **{int(opt['Servidores'])} servidores** → "
                   f"Tiempo en cola ≈ {opt['Tiempo en cola (s)']:.3f}s → Costo = ${opt['Costo mensual ($)']:,.0f}")

# ===================== FOOTER =====================
st.markdown("---")
st.markdown("""
    <p style='text-align: center; color: #888;'>
        Proyecto de Modelos de Colas M/M/c • Servicios en la Nube • 2025<br>
        Simulación discreta con SimPy + visualización con Streamlit & Plotly
    </p>
""", unsafe_allow_html=True)