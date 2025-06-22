# src/model_interface.py
import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from .preprocess import preprocess_text

MODEL_FILENAME = "model.pkl"

def train_model():
    # Carga del dataset
    df = pd.read_csv('data/dataset.csv')
    df['texto_limpio'] = df['texto'].apply(preprocess_text)
    X = df['texto_limpio']
    y = df['sentimiento']
    
    # Dividir en entrenamiento y prueba (aunque aquí se usa solo el entrenamiento)
    X_train, _, y_train, _ = train_test_split(
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
    grid_search_lr = GridSearchCV(pipeline_lr, param_grid_lr, cv=cv, scoring='accuracy', verbose=0)
    grid_search_lr.fit(X_train, y_train)
    cv_score_lr = grid_search_lr.best_score_

    # Pipeline para Multinomial Naive Bayes
    pipeline_nb = Pipeline([
        ('vectorizer', TfidfVectorizer(ngram_range=(1, 2), max_df=0.95)),
        ('classifier', MultinomialNB())
    ])
    param_grid_nb = {'classifier__alpha': [0.1, 0.5, 1.0, 2.0]}
    grid_search_nb = GridSearchCV(pipeline_nb, param_grid_nb, cv=cv, scoring='accuracy', verbose=0)
    grid_search_nb.fit(X_train, y_train)
    cv_score_nb = grid_search_nb.best_score_
    
    if cv_score_lr >= cv_score_nb:
        best_model = grid_search_lr
    else:
        best_model = grid_search_nb

    return best_model

def get_best_model():
    """Carga el modelo si existe; de lo contrario, lo entrena y lo guarda."""
    if os.path.exists(MODEL_FILENAME):
        print("Cargando modelo desde disco...")
        model = joblib.load(MODEL_FILENAME)
    else:
        print("Entrenando el modelo...")
        model = train_model()
        joblib.dump(model, MODEL_FILENAME)
        print("Modelo guardado en", MODEL_FILENAME)
    return model
