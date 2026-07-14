import os
import streamlit as st
import time
import pandas as pd
import joblib

st.set_page_config(page_title="Prevención - ACS", page_icon="🛡️", layout="wide")

st.markdown("""
    <style>
    .main-header {font-size: 2.5rem; color: #1a73e8; font-weight: bold;}
    .sub-header {font-size: 1.1rem; color: #757575; margin-bottom: 2rem;}
    </style>
""", unsafe_allow_html=True)

# --- BARRA LATERAL ---
with st.sidebar:
    st.markdown("### 🏢 Cooperativa ACS")
    st.write("👤 **Usuario:** Tesorería")
    st.divider()
    st.info("💡 **Sistema Predictivo:** Conectado al Árbol de Decisión V1.0 entrenado con historial de cobranzas.")

st.markdown('<p class="main-header">🛡️ Panel de Prevención de Morosidad</p>', unsafe_allow_html=True)

# --- FORMULARIO DE ENTRADA CON LAS VARIABLES REALES DEL APF3 ---
st.write("#### 📝 Ingrese los datos del socio")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**👤 Perfil del Socio**")
    sector = st.selectbox("Sector del Comercial", ['A', 'B', 'C', 'D'])
    tipo_puesto = st.selectbox("Giro de Negocio", ['alimentos', 'ropa', 'servicios', 'otros'])
    antiguedad = st.number_input("Antigüedad (meses)", min_value=1, value=12, step=1)
    canal_pago = st.selectbox("Canal preferido", ['presencial', 'transferencia', 'yape_plin'])

with col2:
    st.markdown("**📊 Historial (Últimos 6 meses)**")
    pagos_puntuales = st.number_input("Pagos puntuales", min_value=0, max_value=6, value=4, step=1)
    aportes_atrasados = st.number_input("Aportes atrasados", min_value=0, max_value=6, value=0, step=1)
    dias_retraso = st.number_input("Días de retraso promedio", min_value=0.0, value=0.0, step=1.0)
    recordatorio = st.checkbox("¿Se le envió recordatorio preventivo?", value=False)

with col3:
    st.markdown("**💰 Finanzas Actuales**")
    cuota_mensual = st.number_input("Monto Cuota Mensual (S/)", min_value=10.0, value=60.0, step=10.0)
    saldo_pendiente = st.number_input("Saldo Pendiente Total (S/)", min_value=0.0, value=0.0, step=10.0)

st.write("")
_, col_btn, _ = st.columns([1, 1, 1])
with col_btn:
    analizar = st.button("🚀 Ejecutar Análisis Predictivo", use_container_width=True, type="primary")

# --- LÓGICA CON EL MODELO REAL ---
if analizar:
    with st.spinner("Conectando con el Árbol de Decisión..."):
        time.sleep(1)
        
        # 1. Ingeniería de características automática (Igual que en tu Jupyter)
        indice_hist_pago = round(pagos_puntuales / (pagos_puntuales + aportes_atrasados + 1), 3)
        ratio_saldo = round(saldo_pendiente / cuota_mensual, 2)
        recibio_rec = 1 if recordatorio else 0
        
        # 2. Armar el DataFrame con una sola fila para el modelo
        datos_entrada = pd.DataFrame([{
            'sector': sector,
            'tipo_puesto': tipo_puesto,
            'antiguedad_meses': antiguedad,
            'monto_cuota_mensual': cuota_mensual,
            'pagos_puntuales_6m': pagos_puntuales,
            'aportes_atrasados_6m': aportes_atrasados,
            'dias_retraso_promedio': dias_retraso,
            'saldo_pendiente': saldo_pendiente,
            'recibio_recordatorio': recibio_rec,
            'canal_pago': canal_pago,
            'indice_historial_pago': indice_hist_pago,
            'ratio_saldo_cuota': ratio_saldo
        }])

        # 3. Intentar cargar el modelo `.joblib`
        directorio_actual = os.path.dirname(os.path.abspath(__file__))
        ruta_modelo = os.path.join(directorio_actual, 'modelo_morosidad_acs.joblib')
        
        if os.path.exists(ruta_modelo):
            modelo = joblib.load(ruta_modelo)
            prediccion = modelo.predict(datos_entrada)[0]
        else:
            # Fallback (Plan B) por si olvidan poner el archivo en la misma carpeta
            st.warning("⚠️ Archivo 'modelo_morosidad_acs.joblib' no encontrado. Mostrando predicción simulada por defecto.")
            if aportes_atrasados >= 2 and dias_retraso >= 10:
                prediccion = 1
            elif ratio_saldo > 2.0 and saldo_pendiente > 100:
                prediccion = 1
            else:
                prediccion = 0

    st.divider()
    st.write("#### 🎯 Resultado de la Evaluación")
    
    if prediccion == 1:
        st.error("🚨 **ALERTA ROJA: SOCIO CON RIESGO DE MORA DETECTADO**")
        c1, c2 = st.columns(2)
        with c1:
            st.warning("**Acción Inmediata Sugerida:**")
            st.write("📞 Contactar al socio de manera preventiva.")
        with c2:
            st.info("**Variables Críticas Detectadas:**")
            st.write(f"- Atrasos recientes: **{aportes_atrasados}**")
            st.write(f"- Ratio de endeudamiento: **{ratio_saldo}x**")
    else:
        st.success("✅ **SOCIO AL DÍA: BAJO RIESGO DE MORA**")
        c1, c2 = st.columns(2)
        with c1:
            st.info("**Acción Sugerida:**")
            st.write("👍 Mantener flujo de cobranza regular.")
        with c2:
            st.metric(label="Deuda Actual a Evaluar", value=f"S/ {saldo_pendiente:.2f}", delta="Nivel Saludable", delta_color="normal")