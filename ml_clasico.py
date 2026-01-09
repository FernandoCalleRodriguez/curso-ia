import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# 1. EL DATASET (Los datos de entrenamiento)
# En la vida real, cargarías esto de un CSV con pd.read_csv()
data = {
    'mensaje': [
        "Oferta increible gana dinero rapido", # Spam
        "Reunión de equipo a las 10am",        # Ham
        "Tu factura ya está disponible",       # Ham
        "Casino online gratis bono bienvenida",# Spam
        "Confirmación de cita médica",         # Ham
        "Pierde peso rápido sin dieta"         # Spam
    ],
    'etiqueta': ['spam', 'ham', 'ham', 'spam', 'ham', 'spam']
}
df = pd.DataFrame(data)

print("--- DATOS DE ENTRENAMIENTO ---")
print(df)

# 2. EL MODELO (Pipeline)
# Paso A: CountVectorizer convierte texto en números (Matriz de frecuencias)
# Paso B: MultinomialNB es un algoritmo clásico de probabilidad (Bayes)
model = make_pipeline(CountVectorizer(), MultinomialNB())

# 3. ENTRENAMIENTO (Aquí ocurre el aprendizaje)
model.fit(df['mensaje'], df['etiqueta'])
print("\n✅ Modelo entrenado con éxito.")

# 4. PREDICCIÓN (Inferencia)
# Probamos con frases que NUNCA ha visto
nuevos_mensajes = [
    "Hola mamá, llego tarde a comer",
    "Gana un iphone gratis haciendo click aqui",
    "Reporte mensual de ventas adjunto"
]

print("\n--- RESULTADOS DE LA PREDICCIÓN ---")
predicciones = model.predict(nuevos_mensajes)

for msg, pred in zip(nuevos_mensajes, predicciones):
    print(f"📝 '{msg}' \n   -> Clasificado como: {pred.upper()}")