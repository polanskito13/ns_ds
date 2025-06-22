import re
import nltk
from nltk.corpus import stopwords

# Descargar stopwords (solo la primera vez)
nltk.download('stopwords', quiet=True)

# Lista de palabras de negación en español
NEGATION_WORDS = {"no", "nunca", "jamás", "sin", "ni", "tampoco"}

def handle_negation(tokens):
    """
    Agrega el prefijo 'NEG_' a las palabras que siguen a una negación,
    hasta encontrar un signo de puntuación.
    """
    output = []
    negate = False
    punctuation = {'.', ',', ';', ':', '!', '?'}
    for token in tokens:
        if token in NEGATION_WORDS:
            negate = True
            output.append(token)
            continue
        if negate and token not in punctuation:
            output.append(f"NEG_{token}")
        else:
            output.append(token)
        if token in punctuation:
            negate = False
    return output

def preprocess_text(text):
    """
    Limpia y normaliza el texto:
      - Conserva letras (incluyendo acentos) y signos de puntuación importantes (!, ?, :, ;, ,, .),
      - Convierte a minúsculas,
      - Aplica manejo de negación,
      - Elimina stopwords en español.
    """
    # Permitir letras y ciertos signos de puntuación
    pattern = r'[^A-Za-záéíóúñÁÉÍÓÚÑ!?:;,.]+'
    text = re.sub(pattern, ' ', text)
    text = text.lower()
    tokens = text.split()
    tokens = handle_negation(tokens)
    stops = set(stopwords.words('spanish'))
    tokens = [token for token in tokens if token not in stops]
    return " ".join(tokens)
