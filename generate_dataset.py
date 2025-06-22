# generate_dataset.py
import pandas as pd
import random

# Listas base de reseñas positivas (formales e informales)
positive_formal = [
    "El servicio es excepcional y me dejó sin palabras.",
    "Estoy encantado con la calidad, supera mis expectativas.",
    "La experiencia fue increíble; definitivamente recomendable.",
    "Un producto de primera, muy satisfecho con la compra.",
    "Calidad y atención al cliente de 10, lo volveré a comprar.",
    "Realmente excelente, me hizo sentir muy valorado.",
    "Impresionante, cada detalle transmite profesionalismo.",
    "Inmejorable, servicio y calidad inigualables.",
    "Me sorprendió gratamente, la calidad es extraordinaria.",
    "La experiencia general fue sobresaliente."
]

positive_informal = [
    "El servicio está brutal, me flipó.",
    "Está de 10, lo recomiendo sin dudar.",
    "Super top, me encantó de verdad.",
    "Buenísimo, lo recomiendo un montón.",
    "Lo máximo, de verdad funcionó.",
    "Excelente, pa' mi gusto es lo mejor.",
    "Muy cool, ¡me rompió las bolas de lo bueno!",
    "Buen servicio, lo re-bueno.",
    "Está buenazo, me sacó una sonrisa.",
    "A todo dar, lo recomiendo."
]

# Listas base de reseñas negativas (formales e informales)
negative_formal = [
    "El servicio fue pésimo y decepcionante en todos los aspectos.",
    "Muy malo, no cumple con lo prometido ni la calidad esperada.",
    "Una experiencia desastrosa, me arrepentí inmediatamente.",
    "El producto es deficiente, un verdadero desperdicio de dinero.",
    "La atención fue negligente y el resultado, deplorable.",
    "No vale la pena, quedó muy por debajo de lo que esperaba.",
    "Una experiencia frustrante, lo recomiendo evitar a toda costa.",
    "Terrible, cada detalle demuestra falta de profesionalismo.",
    "El sabor era insípido y la presentación, descuidada.",
    "Muy insatisfecho, la calidad y el servicio son lamentables."
]

negative_informal = [
    "El servicio apesta, no lo recomiendo.",
    "Malísimo, ta' fatal.",
    "Una experiencia de la chingada.",
    "El producto es una cagada, desperdicio total.",
    "Pésimo, una decepción brutal.",
    "Neta, no vale la pena ni de broma.",
    "Cosa horrible, ni se diga.",
    "El trato fue chafa, muy mal.",
    "Una completa basura, hijo de la chingada.",
    "Desastre total, ni de coña lo compres."
]

# Algunos ejemplos irrelevantes o “ruido”
random_irrelevants = [
    "k", "lol", "asdf", "q", "x", "no se", "blabla", "jeje", "uff", "mmm", "??"
]

# Combinamos listas para tener más variedad
positives = positive_formal + positive_informal
negatives = negative_formal + negative_informal

num_examples_per_class = 300

data_text = []
data_sentiment = []

# Generar ejemplos positivos
for i in range(num_examples_per_class):
    base = random.choice(positives)
    # Agrega una variación (puede ser un extra irrelevante)
    extra = " " + random.choice(random_irrelevants) if random.random() < 0.3 else ""
    # Introducir errores de tipeo de forma aleatoria:
    if random.random() < 0.2:
        base = base.replace("a", "4").replace("e", "3")
    text = base + extra
    data_text.append(text)
    data_sentiment.append(1)

# Generar ejemplos negativos
for i in range(num_examples_per_class):
    base = random.choice(negatives)
    extra = " " + random.choice(random_irrelevants) if random.random() < 0.3 else ""
    if random.random() < 0.2:
        base = base.replace("o", "0").replace("e", "3")
    text = base + extra
    data_text.append(text)
    data_sentiment.append(0)

# Agregar 100 ejemplos adicionales de ruido (aleatorios)
for i in range(100):
    noise = random.choice(random_irrelevants)
    label = random.choice([0, 1])
    data_text.append(noise)
    data_sentiment.append(label)

# Mezclar el dataset
data = {'texto': data_text, 'sentimiento': data_sentiment}
df = pd.DataFrame(data).sample(frac=1, random_state=42).reset_index(drop=True)

# Guardar en data/dataset.csv
df.to_csv('data/dataset.csv', index=False)
print("Dataset generado exitosamente en 'data/dataset.csv' con {} ejemplos".format(len(df)))
