"""
Extracción de información (slots) — Clases 04, 05 y 06.

Tres formas de llenar el "template" de la reserva {fecha, hora, cantidad_personas,
nombre_cliente, sucursal, restriccion_alimentaria}:

  extraer_regex()   Clase 04: patrones de extracción con expresiones regulares
                    (activación por palabras gatillo + calce de patrones).
  extraer_ner()     Clase 05: NER estadístico de spaCy (PER, LOC) para nombre y
                    sucursal; el resto con regex.
  extraer_hibrido() Clase 05, "NER + reglas": primero el patrón explícito
                    ("a nombre de X"), si no calza, la entidad PER del NER.

Además, dos ambigüedades léxicas del dominio (Clase 06, WSD):
  - "mañana": DÍA ("reservar mañana") o FRANJA ("en la mañana").
  - "once":   NÚMERO ("somos once") o COMIDA ("tomar once").
Se resuelven con features COLOCACIONALES (la palabra inmediatamente anterior),
o con la categoría gramatical de spaCy (POS, Clase 03). experimentos.py compara
ambas contra una versión ingenua sin desambiguar.
"""
import re

from preprocesamiento import analizar, normalizar

CAMPOS = ["fecha", "hora", "cantidad_personas", "nombre_cliente", "sucursal", "restriccion_alimentaria"]

DIAS = {"lunes": "lunes", "martes": "martes", "miercoles": "miércoles", "jueves": "jueves",
        "viernes": "viernes", "sabado": "sábado", "domingo": "domingo"}
NUMEROS = {"un": 1, "una": 1, "uno": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6,
           "siete": 7, "ocho": 8, "nueve": 9, "diez": 10, "once": 11, "doce": 12, "trece": 13,
           "catorce": 14, "quince": 15}
_NUM = r"(\d{1,3}|" + "|".join(NUMEROS) + r")"

SUCURSALES_ALIAS = {"providencia": "Providencia", "nueva providencia": "Providencia",
                    "nunoa": "Ñuñoa", "irarrazaval": "Ñuñoa"}
OTRAS_COMUNAS = ["Las Condes", "Maipú", "Vitacura", "La Florida", "Puente Alto", "La Reina", "Macul",
                 "San Miguel", "Santiago Centro", "Lo Barnechea", "Pudahuel", "Peñalolén", "Recoleta",
                 "Estación Central", "Quilicura", "San Bernardo", "Viña del Mar", "Valparaíso",
                 "Concepción", "Temuco", "Antofagasta", "La Serena"]
# restricción -> raíces que la gatillan (concepto canónico, Clase 06: sinonimia)
RESTRICCIONES = {"sin gluten": ["celiac", "sin gluten", "gluten"], "vegano": ["vegan"],
                 "vegetariano": ["vegetarian"], "sin lactosa": ["lactosa"],
                 "para diabéticos": ["diabetic", "sin azucar"]}

_NOMBRE = r"([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)?)"
_PATRONES_NOMBRE = [r"(?i:a nombre de)\s+" + _NOMBRE, r"(?i:mi nombre es|me llamo)\s+" + _NOMBRE,
                    r"\b(?i:soy)\s+" + _NOMBRE, r"(?i:la reserva de)\s+" + _NOMBRE]
_SALUDOS = {"hola", "wena", "buenas", "buenos", "gracias", "oye", "ya", "ok", "perdon"}


def _num(tok: str):
    return int(tok) if tok.isdigit() else NUMEROS.get(tok)


def _ultima(patron: str, texto: str):
    """Se toma la ÚLTIMA mención: 'del sábado... mejor el domingo' -> domingo."""
    m = list(re.finditer(patron, texto))
    return m[-1] if m else None


# ------------------------------------------------------------- WSD: "mañana"
_FECHAS = (r"\b(pasado manana|manana|hoy|lunes|martes|miercoles|jueves|viernes|sabado|domingo"
           r"|\d{1,2}/\d{1,2}(?:/\d{2,4})?)\b")


def _fecha_desde(t: str):
    m = _ultima(_FECHAS, t)
    if not m:
        return None
    v = m.group(1)
    return {"manana": "mañana", "pasado manana": "pasado mañana"}.get(v, DIAS.get(v, v))


def fecha_ingenua(t: str):
    """Sin desambiguar: cualquier 'mañana' es un día."""
    return _fecha_desde(t)


def fecha_colocacional(t: str):
    """Feature colocacional: si a 'mañana' la antecede 'la', es la franja del día, no un día."""
    t = re.sub(r"\bde (lunes|martes|miercoles|jueves|viernes|sabado|domingo) a "
               r"(lunes|martes|miercoles|jueves|viernes|sabado|domingo)\b", " ", t)  # rangos
    t = re.sub(r"\b(en|por|de|a|para) la manana\b", " ", t)
    return _fecha_desde(t)


def fecha_pos(texto: str):
    """WSD con POS de spaCy: 'mañana' NOUN (la mañana) = franja; ADV (mañana voy) = día."""
    doc = analizar(texto)
    franjas = {t.i for t in doc if normalizar(t.text) == "manana" and t.pos_ == "NOUN"
               and t.i > 0 and normalizar(doc[t.i - 1].text) != "pasado"}
    limpio = " ".join("_" if t.i in franjas else t.text for t in doc)
    return _fecha_desde(normalizar(limpio))


# --------------------------------------------------------------- WSD: "once"
def personas_ingenua(t: str):
    """Sin desambiguar: cualquier número escrito cerca de la frase se toma como personas."""
    m = _ultima(r"\b" + _NUM + r"\b(?![:.]\d)(?! hrs)", re.sub(r"\ba las \S+", " ", t))
    return _num(m.group(1)) if m else None


def personas_colocacional(t: str):
    """'once' precedido de 'tomar' / 'la' / 'de' es la comida; 'N personas', 'somos N' son personas."""
    t = re.sub(r"\b(tomar|la|de|al) once\b", " ", t)
    for patron in [r"\b" + _NUM + r" en total\b", r"\bseriamos " + _NUM + r"\b",
                   r"\b" + _NUM + r" (?:personas?|pers\b|comensales|adultos)",
                   r"\b(?:somos|seremos|sean|vamos a ser|grupo de) " + _NUM + r"\b"]:
        m = _ultima(patron, t)
        if m:
            return _num(m.group(1))
    if re.search(r"\b(una pareja|para dos|pa dos)\b", t):
        return 2
    m = _ultima(r"\b(?:para|pa) " + _NUM + r"\b(?![:.]\d)", t)
    return _num(m.group(1)) if m else None


# ---------------------------------------------------------- resto de campos
def extraer_hora(t: str):
    m = _ultima(r"\b(\d{1,2})[:.](\d{2})\b", t)
    if m:
        return f"{int(m.group(1)):02d}:{m.group(2)}"
    m = _ultima(r"\ba las " + _NUM + r"(?:\s*(?:hrs?|horas))?\b", t)
    if m and _num(m.group(1)) is not None and _num(m.group(1)) <= 24:
        return f"{_num(m.group(1)):02d}:00"
    if re.search(r"\b(en la|de|por la|pa la|para la) noche\b", t):
        return "noche"
    if re.search(r"\b(hora de(l)? almuerzo|para (el )?almuerzo|pal almuerzo|a almorzar)\b", t):
        return "almuerzo"
    if re.search(r"\b(en la|por la|de la) tarde\b", t):
        return "tarde"
    return None


def nombre_regex(texto: str):
    encontrado = None
    for patron in _PATRONES_NOMBRE:
        for m in re.finditer(patron, texto):
            encontrado = m.group(1)
    return encontrado


def nombre_ner(texto: str):
    """Entidad PER de spaCy (se descartan saludos que el modelo confunde con nombres)."""
    personas = [e.text for e in analizar(texto).ents
                if e.label_ == "PER" and normalizar(e.text.split()[0]) not in _SALUDOS]
    return personas[-1] if personas else None


def sucursal_regex(t: str):
    m = _ultima(r"\b(nueva providencia|providencia|nunoa|irarrazaval)\b", t)
    return SUCURSALES_ALIAS[m.group(1)] if m else None


def sucursal_ner(texto: str):
    """Entidad LOC de spaCy, mapeada a nuestras sucursales."""
    for e in reversed(analizar(texto).ents):
        if e.label_ == "LOC":
            for alias, suc in SUCURSALES_ALIAS.items():
                if alias in normalizar(e.text):
                    return suc
    return None


def extraer_restriccion(t: str):
    encontradas = [r for r, claves in RESTRICCIONES.items() if any(k in t for k in claves)]
    return " y ".join(encontradas) if encontradas else None


def comunas_sin_local(texto: str):
    """Lugares mencionados donde NO hay local: entidades LOC del NER + diccionario (gazetteer)."""
    t = normalizar(texto)
    lugares = [c for c in OTRAS_COMUNAS if re.search(r"\b" + normalizar(c) + r"\b", t)]
    # Activación por palabras gatillo (Clase 04): una entidad LOC solo cuenta como lugar si la
    # pregunta habla de ubicación. Sin esto, el NER etiqueta "pisco sour" o "Mercado Pago" como LOC.
    if not re.search(r"\b(local|locales|sucursal|sucursales|comuna|queda|ubicad|donde|llegar)", t):
        return lugares
    for e in analizar(texto).ents:
        n = normalizar(e.text)
        if e.label_ == "LOC" and not any(a in n for a in SUCURSALES_ALIAS) \
                and not any(normalizar(c) in n or n in normalizar(c) for c in lugares) \
                and n not in {"santiago", "chile"}:
            lugares.append(e.text)
    return lugares


# --------------------------------------------------------------- extractores
def extraer_regex(texto: str) -> dict:
    t = normalizar(texto)
    return {"fecha": fecha_colocacional(t), "hora": extraer_hora(t),
            "cantidad_personas": personas_colocacional(t), "nombre_cliente": nombre_regex(texto),
            "sucursal": sucursal_regex(t), "restriccion_alimentaria": extraer_restriccion(t)}


def extraer_ner(texto: str) -> dict:
    d = extraer_regex(texto)
    d["nombre_cliente"] = nombre_ner(texto)
    d["sucursal"] = sucursal_ner(texto)
    return d


def extraer_hibrido(texto: str) -> dict:
    d = extraer_regex(texto)
    d["nombre_cliente"] = d["nombre_cliente"] or nombre_ner(texto)
    d["sucursal"] = d["sucursal"] or sucursal_ner(texto)
    return d


EXTRACTORES = {"regex": extraer_regex, "ner_spacy": extraer_ner, "hibrido": extraer_hibrido}
