"""
Gestor de diálogo de la Versión 2 — Clase 02 (autómatas) y Clase 07 (discurso).

1) Autómata de estados finitos (FSA, Clase 02: "interacciones de un chatbot",
   "¿cómo pedimos una pizza?"). La V1 no tenía estados: re-enviaba todo el
   historial y el LLM debía "acordarse". Aquí el estado es explícito:

        INICIO --transacción incompleta--> RECOLECTANDO --completa--> REGISTRADA
          |  ^                               |   ^   |                   |
          |  +-------- consulta/social ------+   +---+ (faltan datos o    |
          |                                         rechazo de política)  |
          +---------------------- transacción completa ------------------>+
   (la tabla TRANSICIONES es la función de transición δ(estado, evento)).

2) Resolución de referencias y elipsis (Clase 07):
   - "Somos 5", "A las 21:00": no dicen 'reservar' pero CONTINÚAN el acto de habla
     en curso -> se mantiene la intención (slot filling).
   - "Perdón, mejor que sean 6" / "Ah, y una es celíaca" justo después de registrar:
     marcadores de discurso ('perdón', 'ah, y', 'mejor') + datos = correferencia con
     la reserva recién hecha -> modificar_reserva.
   - "¿Y el pastel de choclo?" / "¿Y el lunes?": elipsis, como "¿Y mañana?" en el
     ejemplo del clima -> se hereda la intención de consulta anterior.
   - "¿Y cuánto cuesta?": anáfora sin antecedente explícito -> se usan los hechos
     de la respuesta anterior (los referentes del discurso).
   - "a la misma hora": el valor ya está en el estado; no se borra.

3) Reglas de negocio (horario, lunes, 1-12 personas) en código: se GARANTIZAN.
4) Las respuestas a reservas dicen "registré tu solicitud": es lo único que el
   sistema realmente hace (Clase 07: no emitir actos declarativos que no se
   pueden cumplir, como el "Ya cancelé tu reserva" de la V1).
"""
import re

import conocimiento as kb
from extraccion import CAMPOS, EXTRACTORES
from guardrails import enmascarar_sensibles
from intencion import CONSULTAS, TRANSACCIONALES, Clasificador
from preprocesamiento import normalizar
from respuesta import Recuperador, responder_consulta

INICIO, RECOLECTANDO, REGISTRADA = "INICIO", "RECOLECTANDO", "REGISTRADA"
TRANSICIONES = {
    (INICIO, "incompleta"): RECOLECTANDO, (INICIO, "completa"): REGISTRADA,
    (INICIO, "rechazo"): RECOLECTANDO, (INICIO, "consulta"): INICIO, (INICIO, "social"): INICIO,
    (RECOLECTANDO, "incompleta"): RECOLECTANDO, (RECOLECTANDO, "completa"): REGISTRADA,
    (RECOLECTANDO, "rechazo"): RECOLECTANDO, (RECOLECTANDO, "consulta"): RECOLECTANDO,
    (RECOLECTANDO, "social"): RECOLECTANDO,
    (REGISTRADA, "incompleta"): RECOLECTANDO, (REGISTRADA, "completa"): REGISTRADA,
    (REGISTRADA, "rechazo"): RECOLECTANDO, (REGISTRADA, "consulta"): REGISTRADA,
    (REGISTRADA, "social"): REGISTRADA,
}
MARCADOR = re.compile(r"^\W*(y|ah|oh|perdon|mejor|entonces|ya|ok|pero|bueno|al final)\b")
ELIPSIS = re.compile(r"^\W*y\b")
UMBRAL_CONFIANZA = 0.40
UMBRAL_CAMBIO = 0.75      # para abandonar una transacción en curso se exige evidencia fuerte
UMBRAL_SEGMENTO = 0.30    # con 7 clases, 0.30 ya es ~2 veces el azar (1/7)
NOMBRE_CAMPO = {"fecha": "la fecha", "hora": "la hora", "cantidad_personas": "cuántas personas son",
                "nombre_cliente": "el nombre para la reserva",
                "sucursal": "la sucursal (Providencia o Ñuñoa)"}


def _minutos(hora):
    m = re.fullmatch(r"(\d{2}):(\d{2})", hora or "")
    return int(m.group(1)) * 60 + int(m.group(2)) if m else None


def resumen(d: dict) -> str:
    p = []
    if d.get("fecha"): p.append(d["fecha"])
    if d.get("hora"): p.append(f"a las {d['hora']}" if ":" in d["hora"] else f"({d['hora']})")
    if d.get("cantidad_personas"): p.append(f"{d['cantidad_personas']} persona(s)")
    if d.get("sucursal"): p.append(f"sucursal {d['sucursal']}")
    if d.get("nombre_cliente"): p.append(f"a nombre de {d['nombre_cliente']}")
    if d.get("restriccion_alimentaria"): p.append(f"restricción: {d['restriccion_alimentaria']}")
    return ", ".join(p)


def validar_politicas(d: dict):
    """Reglas de negocio en código (en la V1 eran solo un pedido en el prompt)."""
    if d.get("fecha") == "lunes":
        return f"Los lunes estamos cerrados (atendemos {kb.TEXTO_HORARIO}). ¿Te sirve otro día?"
    m = _minutos(d.get("hora"))
    abre = kb.HORA_APERTURA[0] * 60 + kb.HORA_APERTURA[1]
    cierra = kb.HORA_CIERRE[0] * 60 + kb.HORA_CIERRE[1]
    if m is not None and not abre <= m < cierra:
        return f"A las {d['hora']} estamos cerrados (atendemos {kb.TEXTO_HORARIO}). ¿Qué otro horario te acomoda?"
    n = d.get("cantidad_personas")
    if n is not None and not kb.RESERVA_MIN_PERSONAS <= n <= kb.RESERVA_MAX_PERSONAS:
        return (f"Solo reservamos para grupos de {kb.RESERVA_MIN_PERSONAS} a {kb.RESERVA_MAX_PERSONAS} "
                f"personas, así que no puedo reservar para {n}.")
    return None


class Asistente:
    """Versión 2. Cada componente es configurable para los experimentos de ablación."""

    def __init__(self, clasificador="hibrido_lr", extractor="hibrido", seguimiento=True,
                 mundo_cerrado=True, recuperador=None, clf=None):
        self.clf = clf or Clasificador(clasificador)
        self.extraer = EXTRACTORES[extractor]
        self.seguimiento = seguimiento
        self.mundo_cerrado = mundo_cerrado
        self.rec = recuperador or Recuperador()
        self.estado = INICIO
        self.intencion_activa = None
        self.slots = {c: None for c in CAMPOS}
        self.ultima_reserva = None
        self.ultima_intencion = None
        self.ultimos_hechos = []
        self.solicitudes = []   # lo único que el bot puede "confirmar": su propio registro

    # --------------------------------------------------------- discurso
    def _resolver_intencion(self, texto, pred, conf, datos):
        if not self.seguimiento:
            return pred
        t = normalizar(texto)
        n_datos = sum(v is not None for v in datos.values())
        if self.estado == RECOLECTANDO and (n_datos or pred in TRANSACCIONALES) and (
                pred not in TRANSACCIONALES or pred == self.intencion_activa or conf < UMBRAL_CAMBIO):
            return self.intencion_activa                                   # slot filling
        if self.estado == REGISTRADA and n_datos and pred != "cancelar_reserva" and (
                (MARCADOR.match(t) and not texto.strip().endswith("?"))
                or pred == "modificar_reserva" or (pred == "hacer_reserva" and n_datos < 3)):
            return "modificar_reserva"                                     # correferencia
        if ELIPSIS.match(t) and self.ultima_intencion in CONSULTAS:
            return self.ultima_intencion                                   # elipsis
        return pred

    def _clasificar(self, mensaje):
        """Un turno puede traer varios actos de habla ("Hola, quiero reservar" = expresivo +
        directivo). Se segmenta por puntuación (segmentación simple del discurso, Clase 07)
        y, si algún segmento es un acto directivo transaccional (reservar/modificar/cancelar),
        ese acto prevalece sobre el resto; si no, se usa la clasificación del mensaje completo."""
        segmentos = [s for s in re.split(r"[,.;!?¡¿]+", mensaje) if len(s.strip()) > 1] or [mensaje]
        if len(segmentos) == 1 or not self.seguimiento:
            return self.clf.predecir(mensaje)
        completo = self.clf.predecir(mensaje)
        directivos = [c for c in (self.clf.predecir(s) for s in segmentos)
                      if c[0] in TRANSACCIONALES and c[1] >= UMBRAL_SEGMENTO]
        if completo[0] in TRANSACCIONALES or not directivos:
            return completo
        return max(directivos, key=lambda c: c[1])

    # ------------------------------------------------------------ turno
    def procesar(self, mensaje: str) -> dict:
        mensaje = enmascarar_sensibles(mensaje)                            # guardrail (Clase 04)
        datos = self.extraer(mensaje)
        pred, conf = self._clasificar(mensaje)
        intencion = self._resolver_intencion(mensaje, pred, conf, datos)
        hechos = []
        if intencion in TRANSACCIONALES:
            respuesta, evento = self._transaccion(intencion, datos)
            salida_datos = dict(self.slots)
        elif intencion in CONSULTAS:
            eliptica = bool(ELIPSIS.match(normalizar(mensaje)))
            respuesta, hechos = responder_consulta(mensaje, intencion, datos, self.rec,
                                                   referentes_previos=self.ultimos_hechos,
                                                   eliptica=eliptica and self.seguimiento,
                                                   mundo_cerrado_por_intencion=self.mundo_cerrado)
            self.ultimos_hechos = hechos or self.ultimos_hechos
            evento, salida_datos = "consulta", {k: v for k, v in datos.items() if v is not None}
        else:
            respuesta = self._social(intencion, normalizar(mensaje))
            evento, salida_datos = "social", {k: v for k, v in datos.items() if v is not None}
        self.estado = TRANSICIONES[(self.estado, evento)]
        self.ultima_intencion = intencion
        return {"intencion": intencion, "datos_extraidos": salida_datos, "respuesta": respuesta,
                "estado": self.estado, "hechos": hechos, "intencion_clasificador": pred,
                "confianza": round(conf, 2)}

    # ---------------------------------------------------- transacciones
    def _transaccion(self, intencion, datos):
        if intencion != self.intencion_activa or self.estado != RECOLECTANDO:
            base = self.ultima_reserva if (intencion == "modificar_reserva" and self.estado == REGISTRADA) else None
            self.slots = dict(base) if base else {c: None for c in CAMPOS}
            self.intencion_activa = intencion
        self.slots.update({k: v for k, v in datos.items() if v is not None})
        d = self.slots
        error = validar_politicas(d)
        if error:
            return error, "rechazo"
        if intencion == "hacer_reserva":
            faltan = [c for c in ["sucursal", "fecha", "hora", "cantidad_personas", "nombre_cliente"] if not d.get(c)]
            aprox = bool(d.get("hora")) and ":" not in d["hora"]
            if aprox:
                faltan.append("hora")
            if faltan:
                tengo = resumen(d)
                pedir = ", ".join(NOMBRE_CAMPO[c] for c in dict.fromkeys(faltan))
                pref = f"Perfecto, tengo: {tengo}. " if tengo else "¡Con gusto te ayudo con la reserva! "
                return pref + f"Para completarla me falta {pedir}" + (" (necesito una hora exacta)." if aprox else "."), "incompleta"
            return self._registrar("reserva", f"Listo, registré tu solicitud de reserva: {resumen(d)}. ¿Necesitas algo más?")
        if intencion == "modificar_reserva":
            if not d.get("nombre_cliente"):
                return "Para modificar tu reserva necesito el nombre con que la hiciste.", "incompleta"
            if not any(d.get(c) for c in ["fecha", "hora", "cantidad_personas", "sucursal", "restriccion_alimentaria"]):
                return f"¿Qué quieres cambiar de la reserva a nombre de {d['nombre_cliente']}?", "incompleta"
            return self._registrar("modificacion", f"Registré tu solicitud de modificación: {resumen(d)}.")
        faltan = [NOMBRE_CAMPO[c] for c in ["nombre_cliente", "fecha"] if not d.get(c)]
        if faltan:
            return "Para cancelar necesito " + " y ".join(faltan) + " de la reserva.", "incompleta"
        return self._registrar("cancelacion", f"Registré tu solicitud de cancelación: {resumen(d)}.")

    def _registrar(self, tipo, texto):
        self.solicitudes.append({"tipo": tipo, **self.slots})
        self.ultima_reserva = dict(self.slots) if tipo != "cancelacion" else None
        self.intencion_activa = None
        return texto, "completa"

    @staticmethod
    def _social(intencion, t):
        if intencion == "saludo_despedida":
            if re.search(r"gracias|chao|adios|nos vemos|hasta luego|que estes bien", t):
                return f"¡Gracias a ti! Te esperamos en {kb.NOMBRE_LOCAL}."
            return f"¡Hola! Bienvenido/a a {kb.NOMBRE_LOCAL}. Puedo ayudarte con reservas, la carta, horarios y sucursales."
        return "Disculpa, solo puedo ayudarte con temas del restaurante: reservas, la carta, horarios y sucursales."
