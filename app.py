# app.py
import streamlit as st
import pandas as pd
import altair as alt
from src.model_interface import get_best_model

st.set_page_config(page_title="Detector de Sentimientos", layout="wide")
st.title("Detector de Sentimientos Interactivo")
st.write("Este sistema utiliza un modelo de Machine Learning para analizar el sentimiento de textos en español.")

if st.checkbox("Mostrar distribución de sentimientos"):
    try:
        df = pd.read_csv('data/dataset.csv')
        st.write("Primeros registros del dataset:")
        st.write(df.head())
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X('sentimiento:O', title="Sentimiento (0 = Negativo, 1 = Positivo)"),
            y=alt.Y('count()', title="Cantidad")
        ).properties(width=400, height=300, title="Distribución de Sentimientos")
        st.altair_chart(chart)
    except Exception as e:
        st.error(f"Error al cargar el dataset: {e}")

st.subheader("Analiza tu comentario")
user_input = st.text_area("Ingresa un comentario:", value="Escribe aquí tu comentario...")

if st.button("Analizar"):
    if user_input.strip():
        model = get_best_model()  # Carga el modelo persistido o entrena si no existe
        prediction = model.predict([user_input])[0]
        sentiment = "Positivo" if prediction == 1 else "Negativo"
        st.success(f"El sentimiento es: **{sentiment}**")
    else:
        st.error("Ingresa un comentario para analizar.")
