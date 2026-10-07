"""
mystreamlit.py
Autores: Pablo García Aparicio, Miguel Merino Sánchez
NIA: 100522190

Uso:
    streamlit run mystreamlit.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Configuración general de la página
st.set_page_config(
    page_title="Predictor de Depósito Bancario",
    page_icon="bank",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Estilos personalizados para la cabecera, las cajas de resultado y el botón principal
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .prediction-box-yes {
        background: linear-gradient(135deg, #0d7377, #14a085);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        font-size: 1.4rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .prediction-box-no {
        background: linear-gradient(135deg, #c0392b, #e74c3c);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        font-size: 1.4rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .info-box {
        background: #f8f9fa;
        border-left: 4px solid #0f3460;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #0f3460, #16213e);
        color: white;
        border: none;
        padding: 0.75rem;
        font-size: 1.1rem;
        border-radius: 8px;
        cursor: pointer;
        transition: transform 0.2s;
    }
    .stButton > button:hover {
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

# Cabecera principal de la aplicación
st.markdown("""
<div class="main-header">
    <h1>Predictor de Depósito Bancario</h1>
    <p>Modelo de Machine Learning para predecir si un cliente contratará un depósito a plazo</p>
    <small>Práctica 1 – Aprendizaje Automático 2025-26 | Pablo García & Miguel Merino</small>
</div>
""", unsafe_allow_html=True)

# Cargamos el modelo entrenado; usamos cache para no repetir la carga en cada interacción
@st.cache_resource
def load_model():
    try:
        model = joblib.load('modelo_final.joblib')
        return model
    except FileNotFoundError:
        st.error("No se encontró 'modelo_final.joblib'. Asegúrate de que esté en el mismo directorio.")
        st.stop()

model = load_model()

# Reconstruimos el LabelEncoder con las clases que usamos durante el entrenamiento
le = LabelEncoder()
le.classes_ = np.array(['no', 'yes'])

st.success("Modelo cargado correctamente. Listo para realizar predicciones.")

# Formulario principal con los datos del cliente
st.markdown("## Datos del Cliente")
st.markdown("Introduce los datos del cliente para obtener la predicción:")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Datos personales")
        age = st.number_input("Edad", min_value=18, max_value=100, value=35, step=1)
        job = st.selectbox("Tipo de trabajo", [
            'admin.', 'blue-collar', 'entrepreneur', 'housemaid',
            'management', 'retired', 'self-employed', 'services',
            'student', 'technician', 'unemployed', 'unknown'
        ])
        marital = st.selectbox("Estado marital", ['married', 'single', 'divorced'])
        education = st.selectbox("Nivel de educación", [
            'primary', 'secondary', 'tertiary', 'unknown'
        ])
        default = st.selectbox("¿Tiene crédito impagado?", ['no', 'yes'])
        balance = st.number_input("Balance anual medio (€)", value=1000, step=100)
        housing = st.selectbox("¿Tiene hipoteca?", ['yes', 'no'])
        loan = st.selectbox("¿Tiene préstamo personal?", ['no', 'yes'])

    with col2:
        st.markdown("#### Contacto y campaña")
        contact = st.selectbox("Tipo de contacto", ['cellular', 'telephone', 'unknown'])
        day = st.selectbox("Día del mes del último contacto", list(range(1, 32)), index=14)
        month = st.selectbox("Mes del último contacto", [
            'jan', 'feb', 'mar', 'apr', 'may', 'jun',
            'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
        ])
        duration = st.number_input("Duración del último contacto (segundos)", min_value=0, value=180, step=30)
        campaign = st.number_input("N contactos esta campaña", min_value=1, value=2, step=1)

        st.markdown("#### Campaña anterior")
        # Si el cliente fue contactado antes, pedimos cuántos días hace
        was_contacted = st.checkbox("¿Fue contactado en campaña anterior?", value=False)
        pdays_val = -1
        if was_contacted:
            pdays_val = st.number_input("Días desde el último contacto anterior", min_value=0, value=30, step=1)

        previous = st.number_input("N contactos campañas anteriores", min_value=0, value=0, step=1)
        poutcome = st.selectbox("Resultado campaña anterior", [
            'unknown', 'other', 'failure', 'success'
        ])

    submitted = st.form_submit_button("Predecir")

# Bloque de predicción: solo se ejecuta cuando el usuario pulsa el botón
if submitted:
    # Construimos el DataFrame con los mismos nombres de columnas que se usaron en el entrenamiento
    input_data = pd.DataFrame([{
        'age': age,
        'job': job,
        'marital': marital,
        'education': education,
        'default': default,
        'balance': balance,
        'housing': housing,
        'loan': loan,
        'contact': contact,
        'day': int(day),
        'month': month,
        'duration': duration,
        'campaign': campaign,
        # Si no fue contactado anteriormente, dejamos pdays como NaN (valor desconocido)
        'pdays': float(pdays_val) if pdays_val != -1 else np.nan,
        'previous': previous,
        'poutcome': poutcome,
        'pdays_contactado': 1 if was_contacted else 0
    }])

    # Obtenemos la predicción y la probabilidad asociada
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0, 1]
    label = le.inverse_transform([prediction])[0]

    st.markdown("---")
    st.markdown("## Resultado de la Predicción")
    st.markdown("El modelo ha analizado los datos del cliente y ha emitido el siguiente veredicto:")

    if label == 'yes':
        st.markdown(f"""
        <div class="prediction-box-yes">
            EL CLIENTE CONTRATARÁ EL DEPÓSITO<br>
            <small>Probabilidad: {probability:.1%}</small>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="prediction-box-no">
            EL CLIENTE NO CONTRATARÁ EL DEPÓSITO<br>
            <small>Probabilidad de SÍ: {probability:.1%}</small>
        </div>
        """, unsafe_allow_html=True)

    # Barra visual de probabilidad para facilitar la lectura al gestor
    st.markdown("#### Probabilidad de contratación del depósito")
    st.progress(float(probability))
    st.markdown(f"**{probability:.1%}** de probabilidad de contratar el depósito")

    # Tabla con los datos introducidos, útil para que el gestor pueda verificar la entrada
    with st.expander("Ver datos introducidos (para verificación del gestor)"):
        st.dataframe(input_data.T.rename(columns={0: 'Valor'}))

    st.markdown("""
    <div class="info-box">
    <strong>Nota de uso:</strong> Estas predicciones son orientativas y están basadas
    en el modelo entrenado con datos históricos. La decisión final debe ser tomada por el gestor bancario.
    </div>
    """, unsafe_allow_html=True)

# Pie de página
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:grey; font-size:0.85rem">
    Práctica 1 – Aprendizaje Automático 2025-26 |
    Pablo García Aparicio & Miguel Merino Sánchez |
    NIA: 100522190
</div>
""", unsafe_allow_html=True)