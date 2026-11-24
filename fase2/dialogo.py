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
import config
from extraccion import CAMPOS, EXTRACTORES, NUMEROS, _SALUDOS
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
# Saludo al inicio del mensaje ("Hola tienen delivery?"): se separa como un segmento aparte
SALUDO_INICIAL = re.compile(r"^\W*(hola+|holi|wena+|al[oó]|buen[oa]s(\s+(d[ií]as|tardes|noches))?)\b[\s,.!¡]*",
                            re.IGNORECASE)
NOMBRE_CAMPO = {"fecha": "la fecha", "hora": "la hora", "cantidad_personas": "cuántas personas son",
                "nombre_cliente": "el nombre para la reserva",
                "sucursal": "la sucursal (Providencia o Ñuñoa)"}


def _minutos(hora):
    m = re.fullmatch(r"(\d{2}):(\d{2})", hora or "")
    return int(m.group(1)) * 60 + int(m.group(2)) if m else None


def enumerar(items) -> str:
    """'a, b y c' en vez de 'a, b, c'."""
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " y " + items[-1]


def resumen(d: dict) -> str:
    p = []
    if d.get("fecha"): p.append(d["fecha"])
    if d.get("hora"): p.append(f"a las {d['hora']}" if ":" in d["hora"] else f"({d['hora']})")
    if d.get("cantidad_personas"):
        n = d["cantidad_personas"]
        p.append(f"{n} persona" if str(n) == "1" else f"{n} personas")
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

    def __init__(self, clasificador=None, extractor=None, seguimiento=True,
                 mundo_cerrado=True, recuperador=None, clf=None):
        self.clf = clf or Clasificador(clasificador)
        self.extraer = EXTRACTORES[extractor or config.EXTRACTOR]
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
        # Una PREGUNTA sobre la carta o el local en medio de una reserva ("¿Tienen opciones veganas?")
        # se responde como consulta, aunque mencione un dato; la reserva queda en pausa (el estado
        # RECOLECTANDO no cambia) y sigue con el próximo dato.
        # Corrección (Clase 07, reparación): "no, a nombre de Camila" corrige un dato; el "no" no es cancelar.
        if self.estado == RECOLECTANDO and n_datos and re.match(r"\W*(n+o+)+\b", normalizar(texto)):
            return self.intencion_activa
        es_pregunta = "?" in texto or "¿" in texto
        if self.estado == RECOLECTANDO and pred in CONSULTAS and es_pregunta:
            return pred
        # Si la transacción en curso todavía no tiene ningún dato, no hay nada que proteger: un pedido
        # explícito de otra transacción ("quiero reservar" después de un "reserva" mal clasificado como
        # cancelar) cambia de tarea sin exigir UMBRAL_CAMBIO.
        sin_datos_aun = not any(self.slots.values())
        if (self.estado == RECOLECTANDO and sin_datos_aun and pred in TRANSACCIONALES
                and pred != self.intencion_activa and conf >= config.UMBRAL_SEGMENTO):
            return pred
        if self.estado == RECOLECTANDO and (n_datos or pred in TRANSACCIONALES) and (
                pred not in TRANSACCIONALES or pred == self.intencion_activa or conf < config.UMBRAL_CAMBIO):
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
        ese acto prevalece sobre el resto. Si el mensaje parte con un saludo o trae una
        pregunta ("Hola, ¿tienen delivery?") y otro segmento no es social, gana ese segmento:
        el saludo es un acto expresivo y no debe tapar lo que el cliente quiere saber. Una
        despedida sin pregunta ("Ya, muchas gracias, que estés bien") sigue siendo despedida."""
        segmentos = [s for s in re.split(r"[,.;!?¡¿]+", mensaje) if len(s.strip()) > 1] or [mensaje]
        resto = SALUDO_INICIAL.sub("", mensaje, count=1).strip()
        if len(resto) > 1 and resto != mensaje.strip() and len(segmentos) == 1:
            segmentos.append(resto)                      # "Hola tienen delivery?" sin coma
        if len(segmentos) == 1 or not self.seguimiento:
            return self.clf.predecir(mensaje)
        completo = self.clf.predecir(mensaje)
        por_segmento = [self.clf.predecir(s) for s in segmentos]
        directivos = [c for c in por_segmento
                      if c[0] in TRANSACCIONALES and c[1] >= config.UMBRAL_SEGMENTO]
        if completo[0] in TRANSACCIONALES:
            return completo
        if directivos:
            return max(directivos, key=lambda c: c[1])
        es_pregunta = "?" in mensaje or "¿" in mensaje
        if completo[0] == "saludo_despedida" and (SALUDO_INICIAL.match(mensaje) or es_pregunta):
            otros = [c for c in por_segmento
                     if c[0] != "saludo_despedida" and c[1] >= config.UMBRAL_SEGMENTO]
            if otros:
                return max(otros, key=lambda c: c[1])
        return completo

    # ------------------------------------------------------------ turno
    def procesar(self, mensaje: str) -> dict:
        mensaje = enmascarar_sensibles(mensaje)                            # guardrail (Clase 04)
        datos = self.extraer(mensaje)
        # Par pregunta-respuesta (Clase 07): si el bot está pidiendo la cantidad de personas y el
        # cliente responde solo un número ("7", "siete"), ese número es la respuesta a lo que se preguntó.
        suelto = re.fullmatch(r"\W*(\d{1,3}|" + "|".join(NUMEROS) + r")\W*", normalizar(mensaje))
        if (suelto and self.seguimiento and self.estado == RECOLECTANDO and self.intencion_activa == "hacer_reserva"
                and not self.slots.get("cantidad_personas") and datos.get("cantidad_personas") is None):
            g = suelto.group(1)
            datos["cantidad_personas"] = int(g) if g.isdigit() else NUMEROS[g]
        # Lo mismo con el nombre: si es lo único que falta (el bot acaba de pedirlo) y el cliente
        # responde solo "Pedro" o "Ana María", eso es el nombre. Se excluyen saludos ("hola", "gracias").
        requeridos = {"hacer_reserva": config.CAMPOS_RESERVA,
                      "cancelar_reserva": config.CAMPOS_CANCELACION}.get(self.intencion_activa, [])
        solo = re.fullmatch(r"\W*([A-Za-zÁÉÍÓÚÑáéíóúñü]{2,}(?:\s+[A-Za-zÁÉÍÓÚÑáéíóúñü]{2,}){0,2})\W*", mensaje)
        if (solo and self.seguimiento and self.estado == RECOLECTANDO and not any(datos.values())
                and [c for c in requeridos if not self.slots.get(c)] == ["nombre_cliente"]
                and normalizar(solo.group(1).split()[0]) not in _SALUDOS | {"si", "no", "chao", "adios"}):
            datos["nombre_cliente"] = solo.group(1).title()
        # Reparación: en "no, mejor quiero reservar" el "no" rechaza lo anterior; lo que importa es lo que
        # viene después. Se clasifica solo esa parte (si no, el "no" empuja hacia "cancelar").
        # Par pregunta-respuesta (Clase 07): tras "¿Necesitas algo más?", un "no, gracias" cierra la
        # conversación. No es un rechazo de la reserva ni una cancelación (el "no" engañaba al clasificador).
        cierre = r"(n+o+|nop|nada|nada mas|eso es todo|eso seria todo|asi esta bien|estoy bien|todo bien|por ahora no)"
        if (self.seguimiento and self.estado != RECOLECTANDO and re.fullmatch(
                r"\W*" + cierre + r"(?:\W+(?:" + cierre[1:-1] + r"|gracias|muchas gracias|por ahora))*\W*",
                normalizar(mensaje))):
            return {"intencion": "saludo_despedida", "datos_extraidos": {},
                    "respuesta": f"¡Perfecto! Gracias por preferir {kb.NOMBRE_LOCAL}. ¡Te esperamos!",
                    "estado": self.estado, "hechos": [], "intencion_clasificador": "cierre", "confianza": 1.0}
        # Negación del acto de habla (Clase 07): "no quiero cancelar" NO es una cancelación. La cláusula
        # negada se quita antes de clasificar ("no quiero cancelar, quiero reservar" -> "quiero reservar").
        texto_clf = mensaje
        niega = re.search(r"\bno (?:quiero|voy a|necesito|deseo|era) (?:\w+ )?(cancel|anul|modific|cambi|reserv)\w*"
                          r"(?: (?:la|mi|una|de) reserva)?", normalizar(mensaje))
        if niega and self.seguimiento:
            resto = re.sub(r"^\W*(y|sino|mejor|pero)?\W*", "", normalizar(mensaje).replace(niega.group(0), " ").strip())
            if len(re.sub(r"\W", "", resto)) >= 3:
                texto_clf = resto
            else:
                return self._negacion(niega.group(1))
        reparo = re.match(r"\W*(?:n+o+)+\W+(?:mejor\s+|en realidad\s+)?(.{3,})", texto_clf, re.I)
        pred, conf = self._clasificar(reparo.group(1) if reparo and self.estado == RECOLECTANDO else texto_clf)
        intencion = self._resolver_intencion(mensaje, pred, conf, datos)
        hechos = []
        if intencion in TRANSACCIONALES:
            respuesta, evento = self._transaccion(intencion, datos)
            salida_datos = dict(self.slots)
        elif intencion in CONSULTAS:
            eliptica = bool(ELIPSIS.match(normalizar(mensaje)))
            consulta = SALUDO_INICIAL.sub("", mensaje, count=1).strip() or mensaje  # sin "buenas tardes"
            respuesta, hechos = responder_consulta(consulta, intencion, datos, self.rec,
                                                   referentes_previos=self.ultimos_hechos,
                                                   eliptica=eliptica and self.seguimiento,
                                                   mundo_cerrado_por_intencion=self.mundo_cerrado)
            self.ultimos_hechos = hechos or self.ultimos_hechos
            evento, salida_datos = "consulta", {k: v for k, v in datos.items() if v is not None}
        elif self.seguimiento and self.estado == RECOLECTANDO and self.intencion_activa:
            # Algo sin datos en medio de una reserva ("ok", "jaja"): no se saluda de nuevo como si la
            # conversación partiera de cero; se recuerda qué falta para completarla.
            respuesta = self._recordatorio(intencion, normalizar(mensaje))
            evento, salida_datos = "social", dict(self.slots)
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
            faltan = [c for c in config.CAMPOS_RESERVA if not d.get(c)]
            aprox = bool(d.get("hora")) and ":" not in d["hora"]
            if aprox:
                faltan.append("hora")
            if faltan:
                tengo = resumen(d)
                pedir = enumerar(NOMBRE_CAMPO[c] for c in dict.fromkeys(faltan))
                pref = f"Perfecto, tengo: {tengo}. " if tengo else "¡Con gusto te ayudo con la reserva! "
                return pref + f"Para completarla me falta {pedir}" + (" (necesito una hora exacta)." if aprox else "."), "incompleta"
            return self._registrar("reserva", f"Listo, registré tu solicitud de reserva: {resumen(d)}. ¿Necesitas algo más?")
        if intencion == "modificar_reserva":
            if not d.get("nombre_cliente"):
                return "Para modificar tu reserva necesito el nombre con que la hiciste.", "incompleta"
            if not any(d.get(c) for c in ["fecha", "hora", "cantidad_personas", "sucursal", "restriccion_alimentaria"]):
                return f"¿Qué quieres cambiar de la reserva a nombre de {d['nombre_cliente']}?", "incompleta"
            return self._registrar("modificacion", f"Registré tu solicitud de modificación: {resumen(d)}.")
        faltan = [("el nombre con que la hiciste" if c == "nombre_cliente" else NOMBRE_CAMPO[c])
                  for c in config.CAMPOS_CANCELACION if not d.get(c)]
        if faltan:
            return "Para cancelar la reserva necesito " + enumerar(faltan) + ".", "incompleta"
        return self._registrar("cancelacion", f"Registré tu solicitud de cancelación: {resumen(d)}.")

    def _negacion(self, raiz):
        """El cliente niega una tarea ("no quiero cancelar"). Si es la tarea en curso, se abandona sin
        registrar nada; si no, se aclara que no se hará y se sigue con lo que había."""
        negada = {"cancel": "cancelar_reserva", "anul": "cancelar_reserva", "modific": "modificar_reserva",
                  "cambi": "modificar_reserva", "reserv": "hacer_reserva"}[raiz]
        nombre = {"cancelar_reserva": "cancelación", "modificar_reserva": "modificación",
                  "hacer_reserva": "reserva"}[negada]
        if self.estado == RECOLECTANDO and self.intencion_activa == negada:
            self.estado, self.intencion_activa = INICIO, None
            self.slots = {c: None for c in CAMPOS}
            respuesta = (f"Entendido, dejo la {nombre} sin efecto; no registré nada. "
                         "¿Quieres hacer una reserva, ver la carta o saber de horarios y sucursales?")
        elif self.estado == RECOLECTANDO and self.intencion_activa:
            cuerpo = self._recordatorio("", "")[len("Seguimos con tu solicitud: "):]
            respuesta = f"Entendido, no haré ninguna {nombre}. " + cuerpo[0].upper() + cuerpo[1:]
        else:
            respuesta = (f"Entendido, no haré ninguna {nombre}. "
                         "¿Te ayudo con una reserva, la carta o los horarios y sucursales?")
        return {"intencion": "saludo_despedida", "datos_extraidos": dict(self.slots), "respuesta": respuesta,
                "estado": self.estado, "hechos": [], "intencion_clasificador": "negacion", "confianza": 1.0}

    def _recordatorio(self, intencion, t):
        if self.intencion_activa == "hacer_reserva":
            faltan = [c for c in config.CAMPOS_RESERVA if not self.slots.get(c)]
            cuerpo = f"para completar la reserva me falta {enumerar(NOMBRE_CAMPO[c] for c in faltan)}."
        elif self.intencion_activa == "cancelar_reserva":
            faltan = [NOMBRE_CAMPO[c] for c in config.CAMPOS_CANCELACION if not self.slots.get(c)]
            cuerpo = "para cancelar la reserva necesito " + enumerar(faltan) + "."
        else:
            cuerpo = "¿qué quieres cambiar de tu reserva?"
        if re.fullmatch(r"\W*((n+o+)+|no es asi|esta mal|eso no)\W*", t):
            return ("Entendido. Si algún dato quedó mal, escríbeme el correcto (por ejemplo \"a nombre de Ana\" "
                    "o \"sucursal Ñuñoa\") y lo reemplazo. " + cuerpo[0].upper() + cuerpo[1:])
        if intencion == "saludo_despedida" and re.search(r"gracias|chao|adios|nos vemos|hasta luego", t):
            return "¡De nada! Antes de irte, " + cuerpo
        return "Seguimos con tu solicitud: " + cuerpo

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
            if re.search(r"\b(rico|bueno|buenisimo|genial|excelente|perfecto|bacan|wena onda)\b", t):
                return "¡Qué bueno que te guste! ¿Te ayudo con una reserva o quieres saber algo más?"
            return f"¡Hola! Bienvenido/a a {kb.NOMBRE_LOCAL}. Puedo ayudarte con reservas, la carta, horarios y sucursales."
        return "Disculpa, solo puedo ayudarte con temas del restaurante: reservas, la carta, horarios y sucursales."
