"""
Parámetros ajustables de la Versión 2, todos en un solo lugar.

Cada parámetro dice qué controla, con qué concepto del curso se relaciona, qué pasa
si se sube o se baja, y cómo se eligió su valor. Para probar otros valores sin
editar este archivo:

    python barrido.py LR_C 1 10 100
    python barrido.py TAU 0.05 0.15 0.30

Las reglas del NEGOCIO (horario, máximo de personas, carta, sucursales) no están
aquí: son datos del restaurante y viven en conocimiento.py.
"""

# =============================================================================
# 1. INTENCIÓN  (intencion.py) — Clases 06 y 08
# =============================================================================

# Qué clasificador usa el chatbot. Opciones: ver intencion.MODELOS.
# Se eligió "hibrido_lr" (TF-IDF + embeddings) porque tuvo el mejor F1 en validación
# cruzada sobre el corpus de entrenamiento (experimento E1), sin mirar la prueba.
CLASIFICADOR = "hibrido_lr"

# Regresión logística: inverso de la regularización (C = 1/λ).
#   C alto  -> casi sin regularizar: se ajusta mucho a las 150 frases (riesgo de sobreajuste).
#   C bajo  -> regulariza fuerte: pesos pequeños, fronteras más "suaves" (riesgo de subajuste).
# 10 es un valor estándar para textos cortos con pocos datos; ver barrido.py LR_C.
LR_C = 10
LR_MAX_ITER = 3000        # iteraciones máximas del optimizador (solo afecta si no converge)

# Naive Bayes (modelo bow_nb): suavizado de Laplace/Lidstone.
#   Evita probabilidad 0 para palabras que no aparecieron en una clase. 1.0 = Laplace clásico.
NB_ALPHA = 0.5

# TF-IDF de caracteres (modelo tfidf_char_lr): tamaño mínimo y máximo de los n-gramas.
#   (2, 5) captura trozos como "res", "serv", "reserv": resiste faltas de ortografía y
#   formas chilenas ("resérvame", "pal"). Subir el máximo agrega más features (más lento).
CHAR_NGRAMAS = (2, 5)

# Vecinos más cercanos por similitud coseno (modelos tfidf_knn y emb_knn).
#   k chico -> sigue al ejemplo más parecido (sensible a ruido); k grande -> vota la mayoría.
KNN_VECINOS = 3

# LSA (modelo lsa_lr): cuántas dimensiones latentes conserva la SVD truncada (Clase 06).
#   Debe ser menor que el número de frases (150). Pocas dimensiones = más compresión.
LSA_DIMENSIONES = 60


# =============================================================================
# 2. EXTRACCIÓN DE DATOS  (extraccion.py) — Clases 04 y 05
# =============================================================================

# "regex", "ner_spacy" o "hibrido" (regex primero, NER de spaCy como respaldo).
# Elegido por E2: el híbrido iguala a la regex y cubre nombres que el patrón no ve.
EXTRACTOR = "hibrido"


# =============================================================================
# 3. DIÁLOGO  (dialogo.py) — Clases 02 y 07
# =============================================================================
# Las "confianzas" son las probabilidades del clasificador. Con 7 intenciones, el
# azar daría 1/7 ≈ 0.14; el clasificador entrenado con 150 frases rara vez supera 0.6.

# Para ABANDONAR una reserva/modificación/cancelación a medio llenar y empezar otra,
# el clasificador debe estar al menos así de seguro. Evita que "Cámbiala para las 22:00"
# (que suena a reserva) interrumpa una modificación en curso.
#   Subir  -> el bot se "aferra" más a la tarea actual.
#   Bajar  -> cambia de tarea con más facilidad (y más por error).
UMBRAL_CAMBIO = 0.75

# Un mensaje con varios actos de habla ("Hola, quiero reservar") se parte por la
# puntuación. Un segmento transaccional gana sobre el resto si su confianza llega a esto.
#   0.30 ≈ 2 veces el azar. Subir -> se ignoran más segmentos; bajar -> más falsos positivos.
UMBRAL_SEGMENTO = 0.30

# Datos obligatorios antes de registrar una reserva (el orden es el orden en que se piden).
CAMPOS_RESERVA = ["sucursal", "fecha", "hora", "cantidad_personas", "nombre_cliente"]
CAMPOS_CANCELACION = ["nombre_cliente", "fecha"]


# =============================================================================
# 4. RESPUESTAS FUNDAMENTADAS  (respuesta.py) — Clases 06, 10 y 12
# =============================================================================

# Similitud coseno mínima para usar un hecho de la base al responder.
#   Bajar -> responde con hechos menos relacionados (riesgo de contestar otra cosa).
#   Subir -> exige más parecido y se abstiene más ("no tengo información").
# Se fijó en 0.15 ANTES de evaluar; en E4 el resultado no cambia entre 0.05 y 0.20 y
# empieza a empeorar desde 0.30.
TAU = 0.15

# Cuántos hechos como máximo se incluyen en una respuesta.
MAX_HECHOS = 3

# Un hecho solo se incluye si su similitud es al menos esta fracción de la del mejor.
# Evita mezclar un hecho muy relevante (0.60) con otro apenas relacionado (0.16).
RATIO_MEJOR = 0.5

# Ablación de E4: False = RAG ingenuo que siempre responde con lo más parecido, sin
# decir "no tengo información". Dejar en True para el chatbot.
ABSTENCION_ACTIVA = True


# =============================================================================
# 5. EVALUACIÓN  (experimentos.py) — Clase 11
# =============================================================================

CV_PARTICIONES = 5        # validación cruzada estratificada sobre el corpus de entrenamiento
SEMILLA = 0               # para que las particiones y la SVD sean reproducibles
