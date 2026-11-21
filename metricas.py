"""
Calcula las tres métricas que pide el enunciado a partir de los resultados
de una versión del asistente (no vuelve a llamar al modelo):

  1. Intenciones detectadas   -> accuracy global y por intención
  2. Datos extraídos          -> % de campos esperados que se extrajeron bien
  3. Respuestas inventadas    -> % de conversaciones con alucinación,
                                 según la revisión manual de esa versión

Uso:
    python metricas.py v1
    python metricas.py v2        (cuando exista resultados_v2.json)

Lee:     resultados_<version>.json y revision_alucinaciones_<version>.json
Escribe: metricas_<version>.json
"""

import json
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

CAMPOS_VALIDOS = {
    "fecha", "hora", "cantidad_personas",
    "nombre_cliente", "sucursal", "restriccion_alimentaria",
}


def normalizar(valor):
    """Pasa a minúsculas, quita tildes y espacios para comparar textos."""
    if valor is None:
        return None
    texto = str(valor).strip().lower()
    texto = unicodedata.normalize("NFD", texto)
    return "".join(c for c in texto if unicodedata.category(c) != "Mn")


def metricas_intencion(resultados):
    aciertos = sum(r["intencion_detectada"] == r["intencion_esperada"] for r in resultados)
    por_clase = defaultdict(lambda: [0, 0])  # intención -> [aciertos, total]
    confusiones = Counter()
    for r in resultados:
        esperada, detectada = r["intencion_esperada"], r["intencion_detectada"]
        por_clase[esperada][1] += 1
        if esperada == detectada:
            por_clase[esperada][0] += 1
        else:
            confusiones[f"{esperada} -> {detectada}"] += 1
    return {
        "aciertos": aciertos,
        "total": len(resultados),
        "accuracy": round(aciertos / len(resultados), 3),
        "por_intencion": {k: {"aciertos": v[0], "total": v[1]} for k, v in sorted(por_clase.items())},
        "confusiones": dict(confusiones),
    }


def metricas_datos(resultados):
    correctos, total = 0, 0
    por_campo = defaultdict(lambda: [0, 0])  # campo -> [correctos, total]
    errores = []
    claves_fuera_de_esquema = Counter()
    sin_datos = 0

    for r in resultados:
        extraidos = r.get("datos_extraidos")
        if not isinstance(extraidos, dict):
            sin_datos += 1
            extraidos = {}
        for clave in extraidos:
            if clave not in CAMPOS_VALIDOS:
                claves_fuera_de_esquema[clave] += 1

        for campo, esperado in r["datos_esperados"].items():
            obtenido = extraidos.get(campo)
            ok = normalizar(obtenido) == normalizar(esperado)
            total += 1
            por_campo[campo][1] += 1
            if ok:
                correctos += 1
                por_campo[campo][0] += 1
            else:
                errores.append({"id": r["id"], "campo": campo, "esperado": esperado, "obtenido": obtenido})

    return {
        "correctos": correctos,
        "total": total,
        "accuracy": round(correctos / total, 3) if total else None,
        "por_campo": {k: {"correctos": v[0], "total": v[1]} for k, v in sorted(por_campo.items())},
        "errores": errores,
        "claves_fuera_de_esquema": dict(claves_fuera_de_esquema),
        "conversaciones_sin_datos_extraidos": sin_datos,
    }


def metricas_alucinacion(resultados, revision):
    marcados = {c["id"]: c for c in revision["casos"]}
    trampas = [r for r in resultados if r["es_trampa_alucinacion"]]
    normales = [r for r in resultados if not r["es_trampa_alucinacion"]]
    alucina_trampa = [r["id"] for r in trampas if r["id"] in marcados]
    alucina_normal = [r["id"] for r in normales if r["id"] in marcados]
    total_alucina = len(alucina_trampa) + len(alucina_normal)
    return {
        "conversaciones_con_alucinacion": total_alucina,
        "total": len(resultados),
        "tasa_global": round(total_alucina / len(resultados), 3),
        "trampas_con_alucinacion": len(alucina_trampa),
        "total_trampas": len(trampas),
        "tasa_en_trampas": round(len(alucina_trampa) / len(trampas), 3) if trampas else None,
        "casos_normales_con_alucinacion": alucina_normal,
        "por_tipo": dict(Counter(c["tipo"] for c in revision["casos"])),
    }


def main():
    version = sys.argv[1] if len(sys.argv) > 1 else "v1"
    resultados = json.loads(Path(f"resultados_{version}.json").read_text(encoding="utf-8"))
    revision = json.loads(Path(f"revision_alucinaciones_{version}.json").read_text(encoding="utf-8"))

    salida = {
        "version": version,
        "intencion": metricas_intencion(resultados),
        "datos": metricas_datos(resultados),
        "alucinacion": metricas_alucinacion(resultados, revision),
    }
    Path(f"metricas_{version}.json").write_text(
        json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    i, d, a = salida["intencion"], salida["datos"], salida["alucinacion"]
    print(f"=== MÉTRICAS {version.upper()} ===")
    print(f"1. Intenciones correctas:  {i['aciertos']}/{i['total']} ({i['accuracy']:.0%})")
    print(f"2. Datos bien extraídos:   {d['correctos']}/{d['total']} ({d['accuracy']:.0%})")
    print(f"   Claves inventadas fuera del esquema: {sum(d['claves_fuera_de_esquema'].values())}")
    print(f"3. Conversaciones con alucinación: {a['conversaciones_con_alucinacion']}/{a['total']} ({a['tasa_global']:.0%})")
    print(f"   En las trampas: {a['trampas_con_alucinacion']}/{a['total_trampas']} ({a['tasa_en_trampas']:.0%})")
    print(f"\nDetalle guardado en metricas_{version}.json")


if __name__ == "__main__":
    main()
