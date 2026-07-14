import streamlit as st
import time
import pandas as pd
import joblib
import os

# ==========================================================
# CONFIGURACIÓN DE LA PÁGINA
# ==========================================================

st.set_page_config(
    page_title="Dashboard ACS - Riesgo",
    page_icon="📈",
    layout="wide"
)

# ==========================================================
# CARGA DEL MODELO (Se ejecuta una sola vez)
# ==========================================================

@st.cache_resource
def cargar_modelo():
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_modelo = os.path.join(directorio_actual, "modelo_morosidad_acs.joblib")

    if os.path.exists(ruta_modelo):
        return joblib.load(ruta_modelo)

    return None

modelo = cargar_modelo()

# ==========================================================
# ESTILOS CSS
# ==========================================================

st.markdown("""
<style>

.main-header{
font-size:2.6rem;
color:#1a73e8;
font-weight:bold;
margin-bottom:0px;
}

.sub-header{
font-size:1.1rem;
color:#616161;
margin-bottom:25px;
}

.card{
background:#F8F9FA;
padding:15px;
border-radius:10px;
border-left:6px solid #1a73e8;
}

.result-box{
background:#fafafa;
padding:15px;
border-radius:10px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/2942/2942821.png",
        width=90
    )

    st.markdown("## 🏢 Cooperativa ACS")

    st.write("👤 **Usuario:** Área de Tesorería")

    st.write("📅 **Periodo:** Julio 2026")

    st.divider()

    st.success("✅ Motor de Machine Learning conectado")

    st.info(
        """
Modelo Predictivo

• Árbol de Decisión

• Precisión: 82.5%

• Variables utilizadas: 12
"""
    )

# ==========================================================
# ENCABEZADO
# ==========================================================

st.markdown(
    '<p class="main-header">📈 Dashboard de Monitoreo y Prevención de Mora</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="sub-header">Sistema Predictivo para la Evaluación del Riesgo de Morosidad en Socios Comerciantes</p>',
    unsafe_allow_html=True
)

# ==========================================================
# KPIs
# ==========================================================

k1,k2,k3,k4=st.columns(4)

k1.metric(
    "👥 Total Socios",
    "320",
    "+12 este mes"
)

k2.metric(
    "⚠️ Morosidad",
    "40.9 %",
    "-1.5 %",
    delta_color="inverse"
)

k3.metric(
    "💰 Cartera Riesgosa",
    "S/18 340",
    "Requiere seguimiento",
    delta_color="off"
)

k4.metric(
    "🛡️ Evaluaciones",
    "14",
    "Hoy"
)

st.divider()

# ==========================================================
# FORMULARIO
# ==========================================================

st.markdown("## 📝 Evaluación Individual del Socio")

col1,col2,col3=st.columns(3)

# ----------------------------------------------------------

with col1:

    st.markdown("### 👤 Perfil")

    sector=st.selectbox(
        "Sector Comercial",
        ["A","B","C","D"]
    )

    tipo_puesto=st.selectbox(
        "Giro del Negocio",
        [
            "alimentos",
            "ropa",
            "servicios",
            "otros"
        ]
    )

    antiguedad=st.number_input(
        "Antigüedad (meses)",
        min_value=1,
        value=12,
        step=1
    )

    canal_pago=st.selectbox(
        "Canal de Pago",
        [
            "presencial",
            "transferencia",
            "yape_plin"
        ]
    )

# ----------------------------------------------------------

with col2:

    st.markdown("### 📊 Historial")

    pagos_puntuales=st.number_input(
        "Pagos Puntuales",
        min_value=0,
        max_value=6,
        value=4
    )

    aportes_atrasados=st.number_input(
        "Aportes Atrasados",
        min_value=0,
        max_value=6,
        value=0
    )

    dias_retraso=st.number_input(
        "Días de Retraso Promedio",
        min_value=0.0,
        value=0.0,
        step=1.0
    )

    recordatorio=st.checkbox(
        "¿Se envió recordatorio preventivo?"
    )

# ----------------------------------------------------------

with col3:

    st.markdown("### 💰 Situación Financiera")

    cuota_mensual=st.number_input(
        "Cuota Mensual (S/)",
        min_value=10.0,
        value=60.0,
        step=10.0
    )

    saldo_pendiente=st.number_input(
        "Saldo Pendiente (S/)",
        min_value=0.0,
        value=0.0,
        step=10.0
    )

st.write("")

_,c,_=st.columns([1,2,1])

with c:

    analizar=st.button(
        "🚀 Procesar Evaluación con Machine Learning",
        use_container_width=True,
        type="primary"
    )
    # ==========================================================
# LÓGICA DEL MODELO
# ==========================================================

if analizar:

    # ------------------------------------------------------
    # VALIDACIÓN DE DATOS
    # ------------------------------------------------------

    if pagos_puntuales + aportes_atrasados > 6:

        st.error(
            "❌ La suma de pagos puntuales y aportes atrasados no puede superar los 6 meses evaluados."
        )

        st.stop()

    # ------------------------------------------------------

    with st.spinner("Ejecutando algoritmo de Árbol de Decisión..."):

        time.sleep(1)

        # ===============================================
        # Ingeniería de características
        # ===============================================

        indice_hist_pago = round(
            pagos_puntuales /
            (pagos_puntuales + aportes_atrasados + 1),
            3
        )

        ratio_saldo = round(
            saldo_pendiente / cuota_mensual,
            2
        ) if cuota_mensual > 0 else 0

        recibio_rec = 1 if recordatorio else 0

        # ===============================================
        # Mostrar variables calculadas
        # ===============================================

        st.markdown("## 📌 Variables Calculadas Automáticamente")

        c1,c2=st.columns(2)

        c1.metric(
            "Índice Historial de Pago",
            f"{indice_hist_pago:.2f}"
        )

        c2.metric(
            "Ratio Saldo / Cuota",
            f"{ratio_saldo:.2f}"
        )

        # ===============================================
        # Construcción del DataFrame
        # ===============================================

        datos_entrada = pd.DataFrame([{

            "sector":sector,

            "tipo_puesto":tipo_puesto,

            "antiguedad_meses":antiguedad,

            "monto_cuota_mensual":cuota_mensual,

            "pagos_puntuales_6m":pagos_puntuales,

            "aportes_atrasados_6m":aportes_atrasados,

            "dias_retraso_promedio":dias_retraso,

            "saldo_pendiente":saldo_pendiente,

            "recibio_recordatorio":recibio_rec,

            "canal_pago":canal_pago,

            "indice_historial_pago":indice_hist_pago,

            "ratio_saldo_cuota":ratio_saldo

        }])

        # ===============================================
        # Predicción
        # ===============================================

        if modelo is None:

            st.error(
                "No se encontró el archivo modelo_morosidad_acs.joblib"
            )

            st.stop()

        prediccion = modelo.predict(datos_entrada)[0]

        probabilidad = modelo.predict_proba(datos_entrada)[0]

        prob_mora_pct = round(probabilidad[1] * 100,1)

        # ===============================================
        # Clasificación del riesgo
        # ===============================================

        if prob_mora_pct >= 80:

            nivel = "🔴 MUY ALTO"

            color = "red"

        elif prob_mora_pct >= 60:

            nivel = "🟠 ALTO"

            color = "orange"

        elif prob_mora_pct >= 40:

            nivel = "🟡 MODERADO"

            color = "gold"

        else:

            nivel = "🟢 BAJO"

            color = "green"
        # ==========================================================
# RESULTADOS
# ==========================================================

    st.divider()

    st.markdown("## 🎯 Diagnóstico del Modelo Predictivo")

    col_resultado, col_graficos = st.columns([1.2,1])

    # ==========================================================
    # COLUMNA IZQUIERDA
    # ==========================================================

    with col_resultado:

        st.metric(
            "Nivel de Riesgo",
            nivel
        )

        if prediccion == 1:

            st.error("🚨 SOCIO CLASIFICADO CON RIESGO DE MOROSIDAD")

        else:

            st.success("✅ SOCIO CLASIFICADO COMO BAJO RIESGO")

        st.metric(
            "Probabilidad de Mora",
            f"{prob_mora_pct:.1f}%"
        )

        st.progress(int(prob_mora_pct))

        st.markdown("---")

        st.markdown("### 📋 Resumen Ejecutivo")

        st.info(f"""

    **Resultado del análisis**

    • Nivel de riesgo: **{nivel}**

    • Probabilidad estimada: **{prob_mora_pct:.1f}%**

    • Historial de pago: **{indice_hist_pago:.2f}**

    • Ratio deuda/cuota: **{ratio_saldo:.2f}**

    """)

        # ===========================================
        # INTERPRETACIÓN DEL MODELO
        # ===========================================

        st.markdown("### 🔍 Factores que influyeron en la evaluación")

        factores=[]

        if aportes_atrasados>=2:
            factores.append(
                "• Alto número de aportes atrasados."
            )

        if dias_retraso>=10:
            factores.append(
                "• Promedio elevado de días de retraso."
            )

        if ratio_saldo>=1.5:
            factores.append(
                "• El saldo pendiente representa una deuda importante respecto a la cuota mensual."
            )

        if indice_hist_pago<0.50:
            factores.append(
                "• El índice del historial de pago es desfavorable."
            )

        if pagos_puntuales<=2:
            factores.append(
                "• Baja frecuencia de pagos puntuales."
            )

        if len(factores)==0:

            st.success(
                "No se identificaron factores críticos."
            )

        else:

            for f in factores:
                st.write(f)

        st.markdown("---")

        # ===========================================
        # RECOMENDACIONES
        # ===========================================

        st.markdown("### 📞 Recomendaciones")

        if prediccion==1:

            st.warning("""

    1. Contactar preventivamente al socio.

    2. Realizar seguimiento personalizado.

    3. Evaluar refinanciamiento.

    4. Restringir nuevos créditos hasta regularizar la deuda.

    5. Mantener monitoreo continuo.

    """)

        else:

            st.success("""

    1. Mantener el flujo normal de cobranza.

    2. No requiere seguimiento especial.

    3. Elegible para campañas de fidelización.

    4. Continuar monitoreo periódico.

    """)

        # ===========================================
        # EXPANDER
        # ===========================================

        with st.expander("📄 Información financiera detallada"):

            meses_deuda=saldo_pendiente/cuota_mensual if cuota_mensual>0 else 0

            st.write(f"**Saldo pendiente:** S/ {saldo_pendiente:.2f}")

            st.write(f"**Cuota mensual:** S/ {cuota_mensual:.2f}")

            st.write(f"**Equivale aproximadamente a {meses_deuda:.1f} cuotas.**")

            st.write(f"**Pagos puntuales:** {pagos_puntuales}")

            st.write(f"**Aportes atrasados:** {aportes_atrasados}")

            st.write(f"**Promedio de retraso:** {dias_retraso} días")


    # ==========================================================
    # COLUMNA DERECHA
    # ==========================================================

    with col_graficos:

        st.markdown("### 📊 Comparación de Deuda")

        grafico1=pd.DataFrame({

            "Monto":[
                cuota_mensual,
                saldo_pendiente
            ]

        },
        index=[
            "Cuota",
            "Saldo"
        ])

        st.bar_chart(grafico1)

        st.markdown("---")

        st.markdown("### 📈 Historial de Pagos")

        grafico2=pd.DataFrame({

            "Cantidad":[
                pagos_puntuales,
                aportes_atrasados
            ]

        },
        index=[
            "Pagos Puntuales",
            "Atrasados"
        ])

        st.bar_chart(grafico2)

        st.markdown("---")

        st.markdown("### 📌 Variables Calculadas")

        c1,c2=st.columns(2)

        c1.metric(
            "Índice Historial",
            f"{indice_hist_pago:.2f}"
        )

        c2.metric(
            "Ratio Saldo/Cuota",
            f"{ratio_saldo:.2f}"
        )

        st.markdown("---")

        with st.expander("⚙ Información Técnica del Modelo"):

            st.write("**Algoritmo utilizado:** Árbol de Decisión")

            st.write("**Variables originales:** 10")

            st.write("**Variables derivadas:** 2")

            st.write("**Variables totales evaluadas:** 12")

            st.write("**Salida del modelo:**")

            st.write("- Clasificación de riesgo")

            st.write("- Probabilidad de mora")

            st.write("**Precisión obtenida durante el entrenamiento:** 82.5 %")

            st.caption(
                "El modelo analiza simultáneamente las variables del socio para estimar la probabilidad de riesgo de morosidad."
            )