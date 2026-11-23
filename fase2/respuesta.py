"""
Generación de respuestas fundamentadas (grounded) — Clases 06, 07, 10 y 12.

La V1 dejaba que el LLM redactara libremente: así aparecieron sabores de jugo,
medios de pago y un correo que no existen. Aquí la respuesta solo puede contener
HECHOS de conocimiento.py, recuperados como en el RAG mínimo de la Clase 12:

  1. Retrieve: TF-IDF (con stemming) + similitud coseno entre la pregunta y
     cada hecho (Clase 06). Solo se usan hechos sobre un umbral (config.TAU).
  2. Augment: en vez de pasarle los hechos a un LLM, se insertan en una
     plantilla (no hay generación libre -> no hay alucinación posible).
  3. "Si no está, responde No lo sé" (Clase 10/12): las palabras de contenido
     de la pregunta (sustantivos y adjetivos, vía POS de spaCy, Clase 03) que
     no aparecen en ningún hecho se reportan explícitamente.

Mundo cerrado vs. abierto (Clase 07, pragmática: ser verdadero Y relevante):
  - Carta y sucursales están completas -> "No tenemos X en nuestra carta".
  - Servicios (estacionamiento, pagos...) no están en la base -> "No tengo
    información sobre X". Decir "no tenemos estacionamiento" sería afirmar algo
    que no sabemos (el error 'leve' de la V1).
"""
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import conocimiento as kb
import config
from extraccion import comunas_sin_local
from preprocesamiento import analizar, normalizar, stem, tokenizar

GENERICAS = {"opcion", "opciones", "algo", "cosa", "cosas", "tipo", "favor", "dia", "dias", "semana",
             "gente", "vez", "alguna", "algun", "alguno", "mesa", "reserva", "nombre", "tal", "lugar",
             "precio", "precios", "local", "locales", "restaurante", "senor", "senora", "hola", "chile",
             "santiago", "picada", "lang", "chi", "info", "informacion", "consulta", "pregunta", "ayuda",
             "bueno", "buena", "buenas", "favor", "rico", "rica", "mejor", "cuanto", "exacta", "exacto"}
POS_CONTENIDO = {"NOUN", "PROPN", "ADJ"}


def _stopwords():
    """Stopwords de spaCy (Clase 06: quitar palabras vacías antes de ponderar), sin 'sin' ni 'no',
    que en este dominio cambian el significado ('sin gluten')."""
    from spacy.lang.es.stop_words import STOP_WORDS
    return {normalizar(w) for w in STOP_WORDS} - {"sin", "no"}


_STOP = None


def analizador(texto: str):
    """normalizar -> tokenizar -> quitar stopwords -> stemming. El ORDEN importa: si se quitan las
    stopwords después del stemming, 'sábado' -> 'sab' choca con 'sabe' -> 'sab' y desaparece
    (sobre-stemming, Clase 02)."""
    global _STOP
    _STOP = _STOP or _stopwords()
    return [stem(t) for t in tokenizar(texto) if t not in _STOP]


class Recuperador:
    def __init__(self):
        self.ids = [h["id"] for h in kb.HECHOS]
        self.vec = TfidfVectorizer(sublinear_tf=True, analyzer=analizador)
        self.M = self.vec.fit_transform([h["indexar"] + " " + h["texto"] for h in kb.HECHOS])
        # La cobertura usa TODAS las palabras de la base (también stopwords como 'fin' de 'fin de semana')
        self.vocab = {stem(t) for h in kb.HECHOS for t in tokenizar(h["indexar"] + " " + h["texto"])}

    def buscar(self, consulta: str, tau: float = None):
        tau = config.TAU if tau is None else tau
        sims = cosine_similarity(self.vec.transform([consulta]), self.M)[0]
        orden = sims.argsort()[::-1]
        mejor = sims[orden[0]]
        return [(self.ids[i], float(sims[i])) for i in orden[:config.MAX_HECHOS]
                if sims[i] >= tau and sims[i] >= config.RATIO_MEJOR * mejor]

    def cubierto(self, palabra: str) -> bool:
        return stem(palabra) in self.vocab


def _lista(xs):
    xs = list(dict.fromkeys(xs))
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " ni " + xs[-1]


MODIFICADORES = {"amod", "flat", "compound", "fixed", "nmod"}


def _frase(raiz):
    """Frase nominal mínima alrededor de un sustantivo (Clase 05, chunking): núcleo + adjetivos
    + complementos ('menú infantil', 'sillas para guagua', 'cerveza artesanal')."""
    toks = [raiz]
    for c in raiz.children:
        if c.dep_ in MODIFICADORES and c.i > raiz.i:
            toks += [x for x in c.children if x.dep_ == "case"] + [c]
    return " ".join(x.text for x in sorted(toks, key=lambda x: x.i)).lower()


def terminos_no_cubiertos(texto: str, rec: Recuperador, excluir=()):
    """Sustantivos/adjetivos de la pregunta (POS de spaCy, Clase 03) que ningún hecho menciona,
    agrupados en su frase nominal. Devuelve [(frase, es_propiedad)]; 'propiedad' = la frase va tras
    'sin'/'para' (p. ej. 'sin lactosa'): describe una característica, no un producto."""
    doc = analizar(texto)
    en_entidad = {t.i for e in doc.ents if e.label_ in {"PER"} for t in e}
    excluir = {normalizar(x) for x in excluir}
    raices = {}
    for t in doc:
        n = normalizar(t.text)
        if (t.pos_ in POS_CONTENIDO and not t.is_stop and len(n) > 2 and not n.isdigit()
                and n not in GENERICAS and t.i not in en_entidad
                and not any(n in e for e in excluir) and not rec.cubierto(n)):
            raiz = t.head if t.dep_ in MODIFICADORES and t.head.pos_ in {"NOUN", "PROPN"} else t
            while raiz.dep_ in MODIFICADORES and raiz.head.pos_ in {"NOUN", "PROPN"} and raiz.head != raiz:
                raiz = raiz.head
            raices[raiz.i] = raiz
    salida = []
    for raiz in sorted(raices.values(), key=lambda x: x.i):
        previa = normalizar(doc[raiz.i - 1].text) if raiz.i > 0 else ""
        salida.append((_frase(raiz), previa in {"sin", "para"}))
    return salida


def responder_consulta(texto: str, intencion: str, datos: dict, rec: Recuperador,
                       referentes_previos=None, eliptica=False, mundo_cerrado_por_intencion=True):
    """Devuelve (respuesta, ids_de_hechos_usados). Los ids son la 'cita' de cada dato (Clase 12)."""
    t = normalizar(texto)
    hechos = [h for h, _ in rec.buscar(texto, tau=None if config.ABSTENCION_ACTIVA else 0.0)]
    sin_sustantivos = not any(x.pos_ in {"NOUN", "PROPN"} and normalizar(x.text) not in GENERICAS
                              for x in analizar(texto))
    # Resolución de referencias (Clase 07): "¿Y cuánto cuesta?" no nombra nada -> se usa lo último
    # que se mencionó (los referentes del discurso), como "reescribir la pregunta con el historial" (Clase 12)
    if eliptica and referentes_previos and sin_sustantivos:
        hechos = list(referentes_previos)
        especificos = [h for h in hechos if h.startswith(("plato_", "sucursal_"))]
    if "carta" in hechos and re.search(r"\b(carta|menu)\b", t):
        hechos = ["carta"] + [h for h in hechos if not h.startswith("plato_") and h != "carta"]   # la carta ya los incluye
    elif any(h.startswith("plato_") for h in hechos):
        hechos = [h for h in hechos if h != "carta"]
    if datos.get("sucursal"):
        suc = "sucursal_" + normalizar(datos["sucursal"])
        hechos = [h for h in hechos if not h.startswith("sucursal_")] + [suc]
    partes = []

    otras = comunas_sin_local(texto)
    if otras:
        partes.append(f"No tenemos local en {_lista(otras)}; solo en Providencia y Ñuñoa.")
        hechos = [h for h in hechos if not h.startswith("sucursal_")] + ["sucursal_providencia", "sucursal_nunoa"]
    if "horario" in hechos and (datos.get("fecha") == "lunes" or re.search(r"\blunes\b", t)):
        partes.append("Los lunes estamos cerrados.")

    # fecha/hora/personas/nombre/sucursal ya se entendieron; la restricción NO se excluye: si la base
    # no la cubre ("sin lactosa"), hay que decir que no se sabe
    valores_slots = [str(v) for k, v in datos.items() if v is not None and k != "restriccion_alimentaria"]
    faltantes = terminos_no_cubiertos(texto, rec, excluir=list(otras) + valores_slots) if config.ABSTENCION_ACTIVA else []
    hay_plato = any(h.startswith("plato_") for h in hechos)
    negar = [p for p, prop in faltantes if mundo_cerrado_por_intencion and intencion == "consultar_menu"
             and not hay_plato and not prop]
    sin_info = [p for p, _ in faltantes if p not in negar]
    if negar:
        partes.append(f"No tenemos {_lista(negar)} en nuestra carta.")
        if "carta" not in hechos:
            hechos.append("carta")

    hechos = list(dict.fromkeys(hechos))
    if hechos:
        partes.append(" | ".join(kb.HECHOS_POR_ID[h]["texto"] for h in hechos) + ".")
    if sin_info:
        partes.append(f"No tengo información sobre {_lista(sin_info)}; te recomiendo consultarlo en el local.")
    if not partes:
        partes.append("No tengo información sobre eso. Puedo ayudarte con la carta, horarios, "
                      "sucursales y reservas.")
    return " ".join(partes), hechos
