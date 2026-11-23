"""
Guardrails con expresiones regulares — Clase 04 (casos de uso 2 y 5) y Clase 10.

  enmascarar_sensibles()  caso 5: RUT, tarjetas, correos y teléfonos no deben
                          quedar en el historial ni en logs (ni llegar a un LLM).
  parece_injection()      caso 5 / Clase 10: detección simple de prompt injection
                          directa ("ignora las instrucciones..."). Primera línea de
                          defensa, no un sistema de seguridad completo.
  validar_salida_llm()    caso 2: validar el JSON del LLM antes de aceptarlo:
                          claves exactas, intención dentro del enum, precios que
                          existan en la carta y nada de datos de contacto inventados.
"""
import json
import re

import conocimiento as kb

INTENCIONES = ["hacer_reserva", "modificar_reserva", "cancelar_reserva", "consultar_menu",
               "consultar_info_local", "saludo_despedida", "fuera_de_dominio"]
CAMPOS = ["fecha", "hora", "cantidad_personas", "nombre_cliente", "sucursal", "restriccion_alimentaria"]

SENSIBLES = {
    "RUT": r"\b\d{1,2}\.?\d{3}\.?\d{3}-[\dkK]\b",                 # patrón de la Clase 04
    "TARJETA": r"\b(?:\d{4}[ -]?){3}\d{4}\b",
    "EMAIL": r"[\w.+-]+@[\w-]+\.[\w.]+",
    "TELEFONO": r"(?:\+56\s?)?9\s?\d{4}\s?\d{4}\b",               # patrón de la Clase 04
}
_INJECTION = re.compile(
    r"(ignora|olvida|omite)\s+(todas?\s+)?(las|tus|el|todo)?\s*(instrucciones|reglas|anterior|lo anterior|todo)"
    r"|eres un asistente sin reglas|system\s*:|system prompt|prompt del sistema|act[uú]a como", re.I)
PRECIOS_KB = {v[0] for v in kb.CARTA.values()}


def enmascarar_sensibles(texto: str) -> str:
    for etiqueta, patron in SENSIBLES.items():
        texto = re.sub(patron, f"[{etiqueta}]", texto)
    return texto


def parece_injection(texto: str) -> bool:
    return bool(_INJECTION.search(texto))


def precios_sin_respaldo(respuesta: str):
    """Montos "$N" de la respuesta que no son precios de la carta (una suma inventada también se marca)."""
    montos = [int(m.replace(".", "")) for m in re.findall(r"\$\s?(\d{1,3}(?:\.\d{3})+|\d+)", respuesta or "")]
    return [m for m in montos if m not in PRECIOS_KB]


def validar_salida_llm(contenido: str):
    """Devuelve (dict_valido_o_None, lista_de_problemas)."""
    problemas = []
    if not re.search(r"^\s*\{.*\}\s*$", contenido or "", re.S):   # guardrail barato antes de json.loads
        return None, ["no parece JSON"]
    try:
        d = json.loads(contenido)
    except json.JSONDecodeError:
        return None, ["JSON inválido"]
    if set(d) != {"intencion", "datos_extraidos", "respuesta"}:
        problemas.append(f"claves inesperadas: {sorted(d)}")
    if d.get("intencion") not in INTENCIONES:
        problemas.append(f"intención fuera del enum: {d.get('intencion')}")
    datos = d.get("datos_extraidos")
    if not isinstance(datos, dict):
        problemas.append("datos_extraidos no es un objeto")
        d["datos_extraidos"] = {}
    else:
        extra = [k for k in datos if k not in CAMPOS]
        if extra:
            problemas.append(f"campos fuera del esquema: {extra}")
        d["datos_extraidos"] = {k: datos.get(k) for k in CAMPOS}
    if precios_sin_respaldo(d.get("respuesta", "")):
        problemas.append(f"precios sin respaldo: {precios_sin_respaldo(d['respuesta'])}")
    if re.search(r"[\w.]+@[\w.]+|https?://|www\.", d.get("respuesta", "")):
        problemas.append("contacto inventado")
    return d, problemas
