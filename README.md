# practica_1_aprendizaje_automatico-100522156-100522190
Miguel Merino Sanchez,100522156
Pablo Garcia Aparicio,100522190


## Instalación y uso

```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar notebook 1 (EDA + entrenamiento)
jupyter notebook notebook_1_analisis.ipynb

# Ejecutar notebook 2 (predicciones)
jupyter notebook notebook_2_predicciones.ipynb

# Lanzar app Streamlit
streamlit run mystreamlit.py
```

## Notas

- La semilla usada es `100522190` (NIA).
- El escalado, imputación y codificación se realizan dentro de pipelines scikit-learn.
- La variable `pdays` se transforma: `-1 → NaN` y se añade `pdays_contactado` (indicador binario).
- Métrica principal: **ROC-AUC**.
- Evaluación inner: **StratifiedKFold(n_splits=5)**.
- Evaluación outer: **Holdout 2/3 train / 1/3 test**.
