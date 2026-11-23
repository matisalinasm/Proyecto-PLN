"""
Detección de intención — reemplazos clásicos del LLM (Clases 02, 06 y 08).

La intención es el ACTO DE HABLA del cliente (Clase 07): reservar/cancelar son
actos directivos (pedir que el local haga algo), consultar es un directivo de
pregunta, saludar/agradecer es un acto expresivo.

Modelos comparados en experimentos.py (E1):
  reglas         palabras clave escritas a mano (como un ELIZA, Clase 01)
  bow_nb         Bag-of-Words + Naive Bayes                   (Clase 06)
  tfidf_lr       TF-IDF de palabras + Regresión Logística     (Clase 06)
  tfidf_stem_lr  idem, con stemming Snowball                  (Clase 02)
  tfidf_lema_lr  idem, con lematización spaCy                 (Clase 02)
  tfidf_char_lr  TF-IDF de n-gramas de caracteres (2-5)       (Clase 02/06: subpalabras)
  tfidf_knn      TF-IDF + vecino más cercano por coseno       (Clase 06: similitud coseno)
  lsa_lr         TF-IDF + SVD truncada (LSA) + Reg. Logística (Clase 06: LSA)
  emb_lr         promedio de word embeddings (spaCy) + LR      (Clase 08)
  emb_knn        promedio de word embeddings + coseno kNN      (Clase 08)
  hibrido_lr     TF-IDF (stem) + embeddings concatenados + LR  (Clase 06/12: léxico + denso)
Todos se entrenan SOLO con datos/entrenamiento_intenciones.json.
"""
import json
import re
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import FunctionTransformer, Normalizer

import config
from preprocesamiento import analizar, lematizar, normalizar, stem

INTENCIONES = ["hacer_reserva", "modificar_reserva", "cancelar_reserva", "consultar_menu",
               "consultar_info_local", "saludo_despedida", "fuera_de_dominio"]
TRANSACCIONALES = {"hacer_reserva", "modificar_reserva", "cancelar_reserva"}
CONSULTAS = {"consultar_menu", "consultar_info_local"}
RUTA_ENTRENAMIENTO = Path(__file__).parent / "datos" / "entrenamiento_intenciones.json"


def cargar_entrenamiento(ruta=RUTA_ENTRENAMIENTO):
    datos = json.loads(Path(ruta).read_text(encoding="utf-8"))
    X, y = [], []
    for intencion, frases in datos.items():
        if not intencion.startswith("_"):
            X += frases
            y += [intencion] * len(frases)
    return X, y


# --------------------------------------------------------------------------
# Baseline de reglas (se escribió mirando las 50 conversaciones de prueba:
# sirve para mostrar el efecto del sobreajuste / data leakage, Clase 11)
# --------------------------------------------------------------------------
KW = {
    "cancelar": ["cancel", "anul", "dar de baja", "borra", "no podre asistir", "no vamos a poder",
                 "ya no vamos", "eliminar la reserva", "eliminar mi reserva"],
    "modificar": ["cambiar", "cambia", "modific", "reagend", "agregar", "agrega", "mover",
                  "actualizar", "dejala", "en vez de"],
    "reserva": ["reserv", "mesa", "agend", "anot", "ir a comer", "cupo"],
    "info_fuerte": ["maximo", "capacidad", "cuantas personas", "politica"],
    "menu": ["carta", "menu", "plato", "precio", "cuanto cuesta", "cuanto sale", "cuanto vale",
             "postre", "dulce", "jugo", "beber", "tomar", "trago", "vegan", "gluten", "lactosa",
             "diabet", "vegetarian", "pisco", "cerveza", "vino", "terremoto", "completo",
             "chorrillana", "empanada", "pastel", "cazuela", "churrasco", "ensalada", "mote",
             "comida", "promocion", "sabor"],
    "info": ["donde", "direccion", "ubica", "horario", "abren", "cierran", "atienden", "sucursal",
             "local", "estacionamiento", "parking", "despacho", "delivery", "domicilio", "uber",
             "pedidosya", "pago", "pagar", "tarjeta", "junaeb", "sodexo", "amipass", "efectivo",
             "terraza", "fumador", "telefono", "whatsapp", "contacto", "correo", "accesib", "wifi",
             "mascota", "silla"],
    "saludo": ["hola", "buenas", "buenos dias", "gracias", "chao", "adios", "nos vemos",
               "hasta luego", "saludos"],
}


def intencion_reglas(texto: str) -> str:
    t = normalizar(texto)
    hits = lambda g: sum(1 for k in KW[g] if k in t)  # noqa: E731
    if hits("cancelar"):
        return "cancelar_reserva"
    if hits("modificar") and re.search(r"reserv|mesa", t):
        return "modificar_reserva"
    if hits("reserva") and not hits("info_fuerte"):
        return "hacer_reserva"
    menu, info = hits("menu"), hits("info") + hits("info_fuerte")
    if menu or info:
        return "consultar_menu" if menu >= info else "consultar_info_local"
    if hits("saludo"):
        return "saludo_despedida"
    return "fuera_de_dominio"


class ClasificadorReglas:
    def fit(self, X, y):
        return self

    def predict(self, X):
        return [intencion_reglas(x) for x in X]


# --------------------------------------------------------------------------
# Representaciones
# --------------------------------------------------------------------------
class EmbeddingPromedio(BaseEstimator, TransformerMixin):
    """Documento = promedio de los vectores de sus palabras (Clase 08, 'documentos como
    el promedio de sus embeddings'). Vectores estáticos de 300 dimensiones de spaCy."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.vstack([analizar(x).vector for x in X])


def _pre(f):
    return FunctionTransformer(lambda X: [f(x) for x in X])


def construir(nombre: str):
    lr = lambda: LogisticRegression(max_iter=config.LR_MAX_ITER, C=config.LR_C)  # noqa: E731
    if nombre == "reglas":
        return ClasificadorReglas()
    if nombre == "bow_nb":
        return make_pipeline(_pre(normalizar), CountVectorizer(), MultinomialNB(alpha=config.NB_ALPHA))
    if nombre == "tfidf_lr":
        return make_pipeline(_pre(normalizar), TfidfVectorizer(sublinear_tf=True), lr())
    if nombre == "tfidf_stem_lr":
        return make_pipeline(_pre(stem), TfidfVectorizer(sublinear_tf=True), lr())
    if nombre == "tfidf_lema_lr":
        return make_pipeline(_pre(lematizar), TfidfVectorizer(sublinear_tf=True), lr())
    if nombre == "tfidf_char_lr":
        return make_pipeline(_pre(normalizar),
                             TfidfVectorizer(analyzer="char_wb", ngram_range=config.CHAR_NGRAMAS, sublinear_tf=True), lr())
    if nombre == "tfidf_knn":
        return make_pipeline(_pre(stem), TfidfVectorizer(sublinear_tf=True),
                             KNeighborsClassifier(n_neighbors=config.KNN_VECINOS, metric="cosine", weights="distance"))
    if nombre == "lsa_lr":
        return make_pipeline(_pre(stem), TfidfVectorizer(sublinear_tf=True),
                             TruncatedSVD(n_components=config.LSA_DIMENSIONES, random_state=config.SEMILLA), Normalizer(), lr())
    if nombre == "emb_lr":
        return make_pipeline(EmbeddingPromedio(), Normalizer(), lr())
    if nombre == "emb_knn":
        return make_pipeline(EmbeddingPromedio(),
                             KNeighborsClassifier(n_neighbors=config.KNN_VECINOS, metric="cosine", weights="distance"))
    if nombre == "hibrido_lr":
        # Representación híbrida léxica + densa (Clase 06/12): TF-IDF con stemming
        # concatenado al promedio de embeddings.
        lexica = make_pipeline(_pre(stem), TfidfVectorizer(sublinear_tf=True))
        densa = make_pipeline(EmbeddingPromedio(), Normalizer())
        return make_pipeline(FeatureUnion([("lexica", lexica), ("densa", densa)]), lr())
    raise ValueError(nombre)


MODELOS = ["reglas", "bow_nb", "tfidf_lr", "tfidf_stem_lr", "tfidf_lema_lr", "tfidf_char_lr",
           "tfidf_knn", "lsa_lr", "emb_lr", "emb_knn", "hibrido_lr"]


class Clasificador:
    """Envoltorio usado por el chatbot: entrena con el corpus y entrega (intención, confianza)."""

    def __init__(self, nombre: str = None):
        nombre = nombre or config.CLASIFICADOR   # elegido por validación cruzada (E1)
        self.nombre = nombre
        X, y = cargar_entrenamiento()
        self.modelo = construir(nombre).fit(X, y)

    def predecir(self, texto: str):
        if hasattr(self.modelo, "predict_proba"):
            p = self.modelo.predict_proba([texto])[0]
            i = int(np.argmax(p))
            return self.modelo.classes_[i], float(p[i])
        return self.modelo.predict([texto])[0], 1.0
