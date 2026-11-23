"""
Experimentos de la Fase 2: cada pieza de la V1 se compara con su reemplazo clásico del curso.

  E1  Intención         reglas · BoW · TF-IDF (stem/lema/char) · coseno kNN · LSA · embeddings · híbrido
  E2  Extracción        LLM · regex · NER spaCy · híbrido  (+ WSD de "mañana"/"once")
  E3  Diálogo           sin estado vs. autómata + resolución de referencias
  E4  Respuestas        LLM libre vs. recuperación fundamentada (+ ablaciones y umbral TAU)
  E5  Sistema completo  V1 vs. V2 (intención, datos, respuestas inventadas, injection, latencia)
  E6  Ablación          qué aporta cada componente de la V2

Uso:   python experimentos.py        -> resultados/RESULTADOS.md (+ detalle en JSON)
Si existen resultados/llm_*.json (generados con correr_llm.py), se agregan a las tablas.
"""
import json
import re
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import conocimiento as kb
import config
from dialogo import Asistente
from extraccion import (CAMPOS, EXTRACTORES, RESTRICCIONES, fecha_colocacional, fecha_ingenua, fecha_pos,
                        personas_colocacional, personas_ingenua)
from guardrails import precios_sin_respaldo
from intencion import INTENCIONES, MODELOS, Clasificador, cargar_entrenamiento, construir
from preprocesamiento import normalizar
from respuesta import Recuperador

warnings.filterwarnings("ignore")
AQUI = Path(__file__).parent
DATOS, RES = AQUI / "datos", AQUI / "resultados"
RES.mkdir(exist_ok=True)


def leer(ruta):
    return json.loads(Path(ruta).read_text(encoding="utf-8"))


PRUEBA = leer(AQUI.parent / "conversaciones_prueba.json")
HOLDOUT = leer(DATOS / "conversaciones_holdout.json")
MULTI = leer(DATOS / "conversaciones_multiturno.json")
CIEGAS = [{"id": c["id"], "turnos": [c["texto"]], "intencion_esperada": c["intencion_esperada"],
           "datos_esperados": {}, "es_trampa_alucinacion": False} for c in leer(DATOS / "consultas_ciegas.json")["casos"]]
ETIQ = {int(k): v for k, v in leer(DATOS / "etiquetas_grounding.json")["casos"].items()}
ETIQ.update({c["id"]: c for c in leer(DATOS / "consultas_ciegas.json")["casos"]})
WSD = leer(DATOS / "casos_wsd.json")["casos"]
INJ = leer(DATOS / "casos_injection.json")["casos"]
V1_PRUEBA = {r["id"]: {"intencion": r["intencion_detectada"], "datos_extraidos": r["datos_extraidos"],
                       "respuesta": r["respuesta_modelo"]} for r in leer(AQUI.parent / "resultados_v1.json")}
CONJUNTOS = {"prueba": PRUEBA, "holdout": HOLDOUT, "multiturno": MULTI, "ciegas": CIEGAS}


def salidas_llm(version, conjunto):
    """Salidas guardadas de un LLM (V1 en 'prueba' viene del repo original)."""
    if version == "v1" and conjunto == "prueba":
        return V1_PRUEBA
    ruta = RES / f"llm_{version}_{conjunto}.json"
    return {r["id"]: r for r in leer(ruta)} if ruta.exists() else None


# ---------------------------------------------------------------------------
# Métricas comunes
# ---------------------------------------------------------------------------
NULOS = {None, "", "null", "none", "no especificado", "n/a"}


def norm_valor(campo, v):
    if v is None or isinstance(v, (list, dict)) or (isinstance(v, str) and normalizar(v) in NULOS):
        return None
    s = normalizar(str(v))
    if campo == "fecha":
        return re.sub(r"^(este|el|para el|pal|los dias|los)\s+", "", s)
    if campo == "hora":
        m = re.fullmatch(r"(\d{1,2})(?:[:.](\d{2}))?\s*(hrs?|horas)?", s)
        return f"{int(m.group(1)):02d}:{m.group(2) or '00'}" if m else s
    if campo == "cantidad_personas":
        m = re.search(r"\d+", s)
        return int(m.group()) if m else s
    if campo == "sucursal":
        return "nunoa" if ("irarrazaval" in s or "nunoa" in s) else s
    if campo == "restriccion_alimentaria":
        return frozenset(r for r, ks in RESTRICCIONES.items() if any(k in s for k in ks)) or s
    return s


def conteo_slots(esperado, extraido):
    extraido = extraido if isinstance(extraido, dict) else {}
    c = {k: [0, 0, 0] for k in CAMPOS}
    for k in CAMPOS:
        e, x = norm_valor(k, esperado.get(k)), norm_valor(k, extraido.get(k))
        if e is not None and x == e:
            c[k][0] += 1
        else:
            c[k][1] += x is not None
            c[k][2] += e is not None
    return c, [k for k in extraido if k not in CAMPOS]


def f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def metricas(salidas, casos):
    y = [c["intencion_esperada"] for c in casos]
    p = [str(salidas[c["id"]]["intencion"]) for c in casos]
    tot = {k: [0, 0, 0] for k in CAMPOS}
    fuera = 0
    for c in casos:
        cc, fe = conteo_slots(c["datos_esperados"], salidas[c["id"]]["datos_extraidos"])
        fuera += bool(fe)
        for k in CAMPOS:
            for i in range(3):
                tot[k][i] += cc[k][i]
    TP, FP, FN = (sum(v[i] for v in tot.values()) for i in range(3))
    return {"acc": np.mean([a == b for a, b in zip(y, p)]),
            "f1_int": f1_score(y, p, labels=INTENCIONES, average="macro", zero_division=0),
            "f1_datos": f1(TP, FP, FN), "f1_campo": {k: (f1(*v) if sum(v) else None) for k, v in tot.items()},
            "fuera_esquema": fuera,
            "errores": [(c["id"], c["intencion_esperada"], salidas[c["id"]]["intencion"]) for c, a, b in zip(casos, y, p) if a != b]}


def tabla(cabecera, filas):
    out = ["| " + " | ".join(cabecera) + " |", "|" + "---|" * len(cabecera)]
    out += ["| " + " | ".join(str(x) for x in f) + " |" for f in filas]
    return "\n".join(out)


pct = lambda x: "—" if x is None else f"{x:.0%}"   # noqa: E731
dec = lambda x: "—" if x is None else f"{x:.2f}"   # noqa: E731


# ---------------------------------------------------------------------------
# Correr la V2 (con caché del clasificador y del recuperador)
# ---------------------------------------------------------------------------
_CLF, _REC = {}, Recuperador()


def correr_v2(casos, **cfg):
    nombre = cfg.pop("clasificador", None) or config.CLASIFICADOR
    _CLF.setdefault(nombre, Clasificador(nombre))
    salidas, tiempos = {}, []
    for c in casos:
        bot = Asistente(clf=_CLF[nombre], recuperador=_REC, **cfg)
        for t in c["turnos"]:
            t0 = time.perf_counter()
            r = bot.procesar(t)
            tiempos.append(time.perf_counter() - t0)
        salidas[c["id"]] = r
    return salidas, tiempos


# ===========================================================================
def e1_intencion():
    X, y = cargar_entrenamiento()
    cv = StratifiedKFold(config.CV_PARTICIONES, shuffle=True, random_state=config.SEMILLA)
    filas, detalle = [], {}
    primer = lambda casos: [c["turnos"][0] for c in casos]  # noqa: E731
    for m in MODELOS:
        p_cv = construir(m).predict(X) if m == "reglas" else cross_val_predict(construir(m), X, y, cv=cv)
        modelo = construir(m).fit(X, y)
        fila = [m, dec(f1_score(y, p_cv, average="macro"))]
        for nombre, casos in [("prueba", PRUEBA), ("holdout", HOLDOUT)]:
            pred = list(modelo.predict(primer(casos)))
            esp = [c["intencion_esperada"] for c in casos]
            fila += [pct(np.mean([a == b for a, b in zip(esp, pred)])),
                     dec(f1_score(esp, pred, labels=INTENCIONES, average="macro", zero_division=0))]
            detalle[(m, nombre)] = [(c["id"], e, p) for c, e, p in zip(casos, esp, pred) if e != p]
        filas.append(fila)
    v1 = metricas(V1_PRUEBA, PRUEBA)
    v1h = salidas_llm("v1", "holdout")
    v1h = metricas(v1h, HOLDOUT) if v1h else None
    filas.append(["**LLM V1 (gemma3:4b)**", "—", pct(v1["acc"]), dec(v1["f1_int"]),
                  pct(v1h["acc"]) if v1h else "—", dec(v1h["f1_int"]) if v1h else "—"])
    md = ["## E1 · Intención (Clases 02, 06, 08)", "",
          "Cada modelo clasifica el primer mensaje de cada conversación. `CV` = F1-macro con validación cruzada "
          "de 5 particiones sobre el corpus de entrenamiento (150 frases): **es el criterio para elegir el modelo**, "
          "porque no mira los datos de prueba. Las reglas no se entrenan: su CV es su desempeño directo sobre el corpus.", "",
          tabla(["Modelo", "CV F1", "Prueba acc", "Prueba F1", "Holdout acc", "Holdout F1"], filas), "",
          f"Errores del modelo elegido (`{config.CLASIFICADOR}`): " + "; ".join(
              f"{n}: " + ", ".join(f"#{i} {e}→{p}" for i, e, p in detalle[(config.CLASIFICADOR, n)]) for n in ["prueba", "holdout"])]
    return "\n".join(md)


def e2_extraccion():
    filas = []
    for nombre, casos in [("prueba", PRUEBA), ("holdout", HOLDOUT)]:
        sistemas = {}
        for ext, f in EXTRACTORES.items():
            s = {}
            for c in casos:
                acum = {}
                for t in c["turnos"]:   # estado mínimo: lo último mencionado gana
                    acum.update({k: v for k, v in f(t).items() if v is not None})
                s[c["id"]] = {"intencion": c["intencion_esperada"], "datos_extraidos": acum}
            sistemas[ext] = s
        llm = salidas_llm("v1", nombre)
        if llm:
            sistemas = {"LLM V1": llm, **sistemas}
        for ext, s in sistemas.items():
            m = metricas(s, casos)
            filas.append([nombre, ext, dec(m["f1_datos"])] + [dec(m["f1_campo"][k]) for k in CAMPOS] + [m["fuera_esquema"]])
    wsd = []
    for nombre, ff, fp in [("ingenua", fecha_ingenua, personas_ingenua),
                           ("colocacional (regex)", fecha_colocacional, personas_colocacional),
                           ("POS spaCy (fecha)", None, None)]:
        ok_f = ok_p = 0
        for c in WSD:
            t = normalizar(c["texto"])
            f_pred = fecha_pos(c["texto"]) if ff is None else ff(t)
            p_pred = personas_colocacional(t) if fp is None else fp(t)
            ok_f += f_pred == c["fecha"]
            ok_p += p_pred == c["cantidad_personas"]
        wsd.append([nombre, f"{ok_f}/{len(WSD)}", f"{ok_p}/{len(WSD)}" if fp else "(usa colocacional)"])
    md = ["## E2 · Extracción de datos (Clases 04, 05, 06)", "",
          "F1 por campo después de normalizar valores ('este domingo' = 'domingo', '14:00 hrs' = '14:00', "
          "'celíaca' = 'sin gluten'). `fuera esq.` = conversaciones donde se inventaron claves que no existen.", "",
          tabla(["Conjunto", "Extractor", "F1 total"] + CAMPOS + ["fuera esq."], filas), "",
          "### Desambiguación léxica (WSD) de 'mañana' (día/franja) y 'once' (número/comida)", "",
          tabla(["Método", "Fecha correcta", "Personas correcta"], wsd)]
    return "\n".join(md)


def e3_dialogo():
    multi_eval = [c for c in PRUEBA + HOLDOUT if len(c["turnos"]) > 1]
    filas = []
    for nombre, casos in [("multiturno (desarrollo)", MULTI), ("multiturno de prueba+holdout", multi_eval)]:
        for etiqueta, cfg in [("sin estado (cada turno aislado)", {"seguimiento": False}),
                              ("autómata + discurso (V2)", {})]:
            s, _ = correr_v2(casos, **cfg)
            m = metricas(s, casos)
            completas = sum(s[c["id"]]["intencion"] == c["intencion_esperada"] and
                            all(norm_valor(k, c["datos_esperados"].get(k)) == norm_valor(k, s[c["id"]]["datos_extraidos"].get(k))
                                for k in CAMPOS) for c in casos)
            filas.append([nombre, etiqueta, pct(m["acc"]), dec(m["f1_datos"]), f"{completas}/{len(casos)}"])
        for v, et in [("v1", "LLM V1 (historial)"), ("v1b", "LLM V1b (historial)")]:
            llm = salidas_llm(v, "multiturno") if casos is MULTI else None
            if llm:
                m = metricas(llm, casos)
                completas = sum(llm[c["id"]]["intencion"] == c["intencion_esperada"] and
                                all(norm_valor(k, c["datos_esperados"].get(k)) ==
                                    norm_valor(k, (llm[c["id"]]["datos_extraidos"] or {}).get(k)) for k in CAMPOS)
                                for c in casos)
                filas.append([nombre, et, pct(m["acc"]), dec(m["f1_datos"]), f"{completas}/{len(casos)}"])
    s, _ = correr_v2(MULTI)
    fallos = [f"#{c['id']} ({c['fenomeno']}): esperada {c['intencion_esperada']}, obtenida {s[c['id']]['intencion']}"
              for c in MULTI if s[c["id"]]["intencion"] != c["intencion_esperada"]]
    md = ["## E3 · Diálogo (Clases 02 y 07)", "",
          "`completas` = intención final correcta y TODOS los datos exactos. El set multiturno se usó para desarrollar "
          "las reglas de discurso (es optimista); las 4 conversaciones multiturno de prueba+holdout no se usaron.", "",
          tabla(["Conjunto", "Sistema", "Intención final", "F1 datos", "Completas"], filas), "",
          "Fallos de la V2 en el set multiturno: " + ("; ".join(fallos) or "ninguno")]
    return "\n".join(md)


# ------------------------------------------------------------------ grounding
ALIAS = {"plato_pastel": "pastel", "plato_cazuela": "cazuela", "plato_churrasco": "churrasco",
         "plato_empanada": "empanada", "plato_ensalada": "ensalada", "plato_mote": "mote", "plato_jugo": "jugo"}
NEGACION = re.compile(r"no (esta|estan) en (nuestra|la) carta|no tenemos|no ofrecemos|no contamos|no abrimos|"
                      r"no atendemos|cerrad|no existe|no hay |no disponemos|no vendemos|solo reservamos|no aceptamos")
ABSTENCION = re.compile(r"no tengo (esa |mas |el )?(informacion|detalle|datos)|no lo se|no se\b|"
                        r"no dispongo de|no cuento con (esa )?informacion")


def menciona(hecho, r):
    if hecho in ALIAS:
        return ALIAS[hecho] in r
    if hecho == "carta":
        return sum(a in r for a in ALIAS.values()) >= 5
    return {"sucursal_providencia": "2150", "sucursal_nunoa": "3400", "horario": "12:30"}.get(hecho, "") in r \
        if hecho != "politica_reservas" else bool(re.search(r"\b12\b", r) and "persona" in r)


def evaluar_grounding(salidas, casos):
    res = Counter()
    n = 0
    for c in casos:
        e = ETIQ.get(c["id"])
        if not e or c["id"] not in salidas:
            continue
        n += 1
        r = normalizar(salidas[c["id"]]["respuesta"] or "")
        ok_f = all(menciona(h, r) for h in e.get("todas", [])) and \
            (not e.get("alguna_de") or any(menciona(h, r) for h in e["alguna_de"]))
        ok_n = not e["negar"] or bool(NEGACION.search(r))
        ok_a = not e["abstenerse"] or bool(ABSTENCION.search(r))
        niega_sin_saber = e["abstenerse"] and not e["negar"] and bool(NEGACION.search(r)) and not ABSTENCION.search(r)
        res.update({"fuentes": ok_f, "negar": ok_n, "abstener": ok_a, "todo": ok_f and ok_n and ok_a,
                    "niega_sin_saber": niega_sin_saber, "precio_sin_respaldo": bool(precios_sin_respaldo(salidas[c["id"]]["respuesta"]))})
    return n, res


def e4_respuestas():
    filas, sens = [], []
    variantes = [("V2 completa", {}), ("V2 sin 'No lo sé' (solo hechos recuperados)", {"abstencion": False}),
                 ("V2 sin mundo cerrado (siempre se abstiene)", {"mundo_cerrado": False})]
    for nombre, casos in CONJUNTOS.items():
        for etiqueta, cfg in [("LLM V1", None), ("LLM V1b (prompt mejorado)", None)] + variantes:
            if cfg is None:
                s = salidas_llm("v1" if etiqueta == "LLM V1" else "v1b", nombre)
                if not s:
                    continue
            else:
                config.ABSTENCION_ACTIVA = cfg.get("abstencion", True)
                s, _ = correr_v2(casos, **{k: v for k, v in cfg.items() if k != "abstencion"})
                config.ABSTENCION_ACTIVA = True
            n, r = evaluar_grounding(s, casos)
            filas.append([nombre, etiqueta, n, pct(r["fuentes"] / n), pct(r["negar"] / n), pct(r["abstener"] / n),
                          f"**{pct(r['todo'] / n)}**", r["niega_sin_saber"], r["precio_sin_respaldo"]])
    original = config.TAU
    for tau in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]:
        config.TAU = tau
        fila = [tau]
        for nombre in ["prueba", "holdout", "ciegas"]:
            s, _ = correr_v2(CONJUNTOS[nombre])
            n, r = evaluar_grounding(s, CONJUNTOS[nombre])
            fila.append(pct(r["todo"] / n))
        sens.append(fila)
    config.TAU = original
    md = ["## E4 · Respuestas fundamentadas (Clases 06, 07, 10, 12)", "",
          "Solo consultas (menú/local) con etiqueta en `datos/etiquetas_grounding.json`. `fuentes` = menciona los hechos "
          "correctos; `negar` = dice que algo no existe cuando corresponde (carta/sucursales completas); `abstenerse` = dice "
          "'no tengo información' cuando la base no lo sabe; `todo` = las tres a la vez. `niega sin saber` = afirma que algo "
          "NO existe cuando en realidad no se sabe (el error 'leve' de la V1). `$ sin respaldo` = montos que no son precios de la carta.",
          "", "**`ciegas` se escribió después de construir el módulo y nunca se usó para ajustarlo: es la medida más honesta.**", "",
          tabla(["Conjunto", "Sistema", "n", "fuentes", "negar", "abstenerse", "todo", "niega sin saber", "$ sin respaldo"], filas),
          "", f"### Sensibilidad al umbral de recuperación TAU (config: {config.TAU}, fijado antes de evaluar)", "",
          tabla(["TAU", "prueba (todo)", "holdout (todo)", "ciegas (todo)"], sens)]
    return "\n".join(md)


# ------------------------------------------------------------- end-to-end
def cargar_anotacion(nombre):
    ruta = DATOS / f"anotacion_alucinaciones_{nombre}.json"
    return {a["id"]: a for a in leer(ruta)["anotaciones"]} if ruta.exists() else {}


def contar_alucinaciones(salidas, anot):
    c = Counter()
    for i, s in salidas.items():
        a = anot.get(i)
        c["sin_anotar" if (a is None or a["respuesta"] != s["respuesta"]) else a["categoria"]] += 1
    return c


def e5_sistema():
    filas, detalle = [], {}
    for nombre, casos in [("prueba", PRUEBA), ("holdout", HOLDOUT)]:
        sistemas = []
        for v, et in [("v1", "LLM V1 (cruda)"), ("v1b", "LLM V1b (prompt mejorado)")]:
            s = salidas_llm(v, nombre)
            if s:
                sistemas.append((et, s, f"{v}" + ("" if nombre == "prueba" else "_" + nombre)))
        s2, tiempos = correr_v2(casos)
        sistemas.append(("V2 (sin LLM)", s2, "v2" + ("" if nombre == "prueba" else "_" + nombre)))
        detalle[nombre] = s2
        for et, s, clave in sistemas:
            m = metricas(s, casos)
            al = contar_alucinaciones(s, cargar_anotacion(clave))
            filas.append([nombre, et, pct(m["acc"]), dec(m["f1_int"]), dec(m["f1_datos"]), m["fuera_esquema"],
                          al.get("grave", 0), al.get("leve", 0), al.get("accion_falsa", 0), al.get("sin_anotar", 0)])
    inj = []
    s, _ = correr_v2([{"id": c["id"], "turnos": [c["texto"]]} for c in INJ])
    pasa = [all(x.lower() not in s[c["id"]]["respuesta"].lower() for x in c["no_debe_contener"]) for c in INJ]
    inj.append(["V2 (sin LLM)", f"{sum(pasa)}/{len(INJ)}"])
    llm = salidas_llm("v1b", "injection")
    if llm:
        pasa = [all(x.lower() not in llm[c["id"]]["respuesta"].lower() for x in c["no_debe_contener"]) for c in INJ]
        inj.append(["LLM V1b", f"{sum(pasa)}/{len(INJ)}"])
    from preprocesamiento import analizar
    analizar.cache_clear()          # sin caché de spaCy: latencia real de un mensaje nuevo
    _, tiempos = correr_v2(PRUEBA + HOLDOUT)
    lat = [["V2 (sin LLM)", f"{np.mean(tiempos) * 1000:.0f} ms", f"{np.percentile(tiempos, 95) * 1000:.0f} ms",
            "spaCy es_core_news_md (~50 MB en disco) + TF-IDF"]]
    for v in ["v1", "v1b"]:
        for conj in ["holdout", "prueba"]:
            r = RES / f"llm_{v}_{conj}.json"
            if r.exists():
                ls = [x["latencia_s"] for x in leer(r)]
                lat.append([f"LLM {v.upper()} ({conj})", f"{np.mean(ls):.1f} s", f"{np.percentile(ls, 95):.1f} s",
                            "gemma3:4b cuantizado (GGUF 4 bits en Ollama)"])
                break
    (RES / "salidas_v2_prueba.json").write_text(json.dumps(detalle["prueba"], ensure_ascii=False, indent=1), encoding="utf-8")
    (RES / "salidas_v2_holdout.json").write_text(json.dumps(detalle["holdout"], ensure_ascii=False, indent=1), encoding="utf-8")
    md = ["## E5 · Sistema completo (las tres métricas del enunciado)", "",
          "Intención y datos del último turno de cada conversación. Respuestas inventadas: anotación manual "
          "(`datos/anotacion_alucinaciones_*.json`); `grave` = inventa un dato o viola una política; `leve` = afirma algo "
          "que la base no respalda; `acción falsa` = dice haber hecho algo que el sistema no hace ('ya cancelé').", "",
          tabla(["Conjunto", "Sistema", "Intención acc", "Intención F1", "F1 datos", "fuera esq.", "grave", "leve",
                 "acción falsa", "sin anotar"], filas), "",
          "### Prompt injection y datos sensibles (`datos/casos_injection.json`)", "",
          tabla(["Sistema", "Casos resistidos"], inj), "",
          "### Eficiencia (Clase 12)", "",
          tabla(["Sistema", "Latencia media por turno", "p95", "Modelo"], lat)]
    return "\n".join(md)


def e6_ablacion():
    configs = [("V2 completa", {}),
               ("clasificador → reglas", {"clasificador": "reglas"}),
               ("clasificador → TF-IDF char", {"clasificador": "tfidf_char_lr"}),
               ("clasificador → embeddings", {"clasificador": "emb_lr"}),
               ("extractor → solo regex", {"extractor": "regex"}),
               ("extractor → solo NER spaCy", {"extractor": "ner_spacy"}),
               ("sin seguimiento de diálogo", {"seguimiento": False})]
    filas = []
    for et, cfg in configs:
        fila = [et]
        for casos in [HOLDOUT, MULTI]:
            s, _ = correr_v2(casos, **dict(cfg))
            m = metricas(s, casos)
            fila += [pct(m["acc"]), dec(m["f1_datos"])]
        s, _ = correr_v2(CIEGAS, **dict(cfg))
        n, r = evaluar_grounding(s, CIEGAS)
        fila.append(pct(r["todo"] / n))
        filas.append(fila)
    md = ["## E6 · Ablación: qué aporta cada componente", "",
          "Se cambia UNA pieza a la vez. Holdout y ciegas no se usaron para diseñar; multiturno sí (desarrollo).", "",
          tabla(["Configuración", "Holdout intención", "Holdout F1 datos", "Multiturno intención",
                 "Multiturno F1 datos", "Ciegas grounding"], filas)]
    return "\n".join(md)


def main():
    partes = ["# Resultados de la Fase 2", "",
              "Generado por `python experimentos.py`. Las salidas de la V1 sobre las 50 conversaciones de prueba "
              "están en `../resultados_v1.json`; las de la V1 y la V1b en los demás conjuntos se generaron con "
              "`correr_llm.py` (gemma3:4b en Ollama) y están en `resultados/llm_*.json`.", ""]
    for f in [e1_intencion, e2_extraccion, e3_dialogo, e4_respuestas, e5_sistema, e6_ablacion]:
        t0 = time.time()
        partes += [f(), ""]
        print(f"{f.__name__} listo ({time.time() - t0:.0f}s)")
    (RES / "RESULTADOS.md").write_text("\n".join(partes), encoding="utf-8")
    print("\n".join(partes))


if __name__ == "__main__":
    main()
