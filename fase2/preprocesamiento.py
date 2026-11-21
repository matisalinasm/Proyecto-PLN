"""
Clase 02 (Morfología) — normalización, tokenización, stemming y lematización.

Se usan como paso previo de los clasificadores (intencion.py) y de la
recuperación de información (respuesta.py), y se comparan entre sí en el
experimento E1:
  - normalizar(): minúsculas + sin tildes  ("Ñuñoa" -> "nunoa")
  - stemming Snowball (NLTK): corta sufijos aunque el tronco no sea una palabra
    válida ("reservaciones" -> "reserv"). Es el stemmer "intermedio" de la clase
    (Porter es más suave y Lancaster más radical).
  - lematización (spaCy, es_core_news_md): usa diccionario + reglas para llegar
    al lema válido ("quiero" -> "querer", "mesas" -> "mesa").
"""
import re
import unicodedata
from functools import lru_cache

from nltk.stem.snowball import SnowballStemmer

_STEMMER = SnowballStemmer("spanish")
_TOKEN = re.compile(r"\d{1,2}[:.]\d{2}|\d+|[a-zñ]+")


def normalizar(texto: str) -> str:
    """Minúsculas, sin tildes y espacios simples: 'Ñuñoa, ¿Mañana?' -> 'nunoa, ¿manana?'."""
    texto = unicodedata.normalize("NFD", texto.lower())
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", texto).strip()


def tokenizar(texto: str) -> list:
    """Tokenización por regex (Clase 02, slide 'Tokenización'): horas, números y palabras."""
    return _TOKEN.findall(normalizar(texto))


def stem(texto: str) -> str:
    return " ".join(_STEMMER.stem(t) for t in tokenizar(texto))


@lru_cache(maxsize=1)
def nlp():
    """Carga perezosa del modelo de spaCy (POS, lemas, NER y vectores de palabras)."""
    import spacy
    return spacy.load("es_core_news_md")


@lru_cache(maxsize=4096)
def analizar(texto: str):
    """Documento spaCy cacheado (se usa en varios módulos sobre el mismo texto)."""
    return nlp()(texto)


def lematizar(texto: str) -> str:
    return " ".join(normalizar(t.lemma_) for t in analizar(texto) if not t.is_punct and not t.is_space)


PREPROCESADORES = {
    "normalizado": normalizar,
    "stemming": stem,
    "lematizacion": lematizar,
}
