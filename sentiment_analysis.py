# src/sentiment_analysis.py
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from .preprocess import preprocess_text

# Cargar el dataset
df = pd.read_csv('data/dataset.csv')
if 'texto' not in df.columns or 'sentimiento' not in df.columns:
    raise ValueError("El dataset debe contener las columnas 'texto' y 'sentimiento'.")

# Preprocesamiento
df['texto_limpio'] = df['texto'].apply(preprocess_text)
X = df['texto_limpio']
y = df['sentimiento']

# División de datos (80/20) con estratificación
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

n_splits = 3 if y_train.value_counts().min() >= 3 else 2
cv = StratifiedKFold(n_splits=n_splits)

# Pipeline para Regresión Logística
pipeline_lr = Pipeline([
    ('vectorizer', TfidfVectorizer(ngram_range=(1, 2), max_df=0.95)),
    ('classifier', LogisticRegression(max_iter=5000, solver='saga'))
])
param_grid_lr = {'classifier__C': [0.1, 1, 10, 100]}
grid_search_lr = GridSearchCV(pipeline_lr, param_grid_lr, cv=cv, scoring='accuracy', verbose=1)
grid_search_lr.fit(X_train, y_train)
cv_score_lr = grid_search_lr.best_score_

print("Logistic Regression - Mejores parámetros:", grid_search_lr.best_params_)
print("Logistic Regression - Mejor precisión en validación (CV):", cv_score_lr)

# Pipeline para Multinomial Naive Bayes
pipeline_nb = Pipeline([
    ('vectorizer', TfidfVectorizer(ngram_range=(1, 2), max_df=0.95)),
    ('classifier', MultinomialNB())
])
param_grid_nb = {'classifier__alpha': [0.1, 0.5, 1.0, 2.0]}
grid_search_nb = GridSearchCV(pipeline_nb, param_grid_nb, cv=cv, scoring='accuracy', verbose=1)
grid_search_nb.fit(X_train, y_train)
cv_score_nb = grid_search_nb.best_score_

print("\nMultinomial Naive Bayes - Mejores parámetros:", grid_search_nb.best_params_)
print("Multinomial Naive Bayes - Mejor precisión en validación (CV):", cv_score_nb)

if cv_score_lr >= cv_score_nb:
    best_model = grid_search_lr
    print("\nSeleccionando Logistic Regression como mejor modelo (CV: {:.2f})".format(cv_score_lr))
else:
    best_model = grid_search_nb
    print("\nSeleccionando Multinomial Naive Bayes como mejor modelo (CV: {:.2f})".format(cv_score_nb))

y_pred = best_model.predict(X_test)
print("\nPrecisión en Test:", accuracy_score(y_test, y_pred))
print("\nReporte de Clasificación en Test:\n", classification_report(y_test, y_pred))

# Probar con nuevos textos
nuevos_textos = [
    "¡Estoy absolutamente enamorado de este producto! Es fantástico y maravilloso.",
    "Fue una experiencia de la chingada; no recomiendo nada de esto.",
    "La experiencia fue buena, aunque con varios detalles, meh.",
    "Definitivamente excelente, no hay nada que criticar, ¡perfecto!",
    "Una decepción total, ni se diga, una completa basura."
]
predicciones = best_model.predict(nuevos_textos)
print("\nPredicciones para nuevos textos (0 = negativo, 1 = positivo):", predicciones)
