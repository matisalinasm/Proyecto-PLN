import json
import ollama
from Version1 import MODELO, SYSTEM_PROMPT


def evaluar_dataset():
    with open("conversaciones_prueba.json", "r", encoding="utf-8") as f:
        pruebas = json.load(f)

    resultados = []
    intenciones_correctas = 0
    errores_json = 0

    print(f"Iniciando evaluación automática de {len(pruebas)} conversaciones en {MODELO}...\n")

    for caso in pruebas:
        historial = [{"role": "system", "content": SYSTEM_PROMPT}]
        ultimo_json = {}

        for turno in caso["turnos"]:
            historial.append({"role": "user", "content": turno})
            resp = ollama.chat(
                model=MODELO,
                messages=historial,
                format="json",
                options={"temperature": 0.2},
            )
            contenido = resp["message"]["content"]
            historial.append({"role": "assistant", "content": contenido})

            try:
                ultimo_json = json.loads(contenido)
            except json.JSONDecodeError:
                errores_json += 1
                ultimo_json = {"intencion": "ERROR_JSON", "datos_extraidos": {}, "respuesta": contenido}

        intencion_predicha = ultimo_json.get("intencion")
        acierto_intencion = intencion_predicha == caso["intencion_esperada"]
        if acierto_intencion:
            intenciones_correctas += 1

        registro = {
            "id": caso["id"],
            "turnos": caso["turnos"],
            "intencion_esperada": caso["intencion_esperada"],
            "intencion_detectada": intencion_predicha,
            "acierto_intencion": acierto_intencion,
            "datos_esperados": caso["datos_esperados"],
            "datos_extraidos": ultimo_json.get("datos_extraidos", {}),
            "es_trampa_alucinacion": caso["es_trampa_alucinacion"],
            "motivo_trampa": caso.get("motivo_trampa", ""),
            "respuesta_modelo": ultimo_json.get("respuesta", ""),
        }
        resultados.append(registro)
        print(f"Caso #{caso['id']:02d} | Esperada: {caso['intencion_esperada']} | Detectada: {intencion_predicha} | OK: {acierto_intencion}")

    with open("resultados_v1.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    print("\n=== RESUMEN MÉTRICAS VERSIÓN 1 CRUDA ===")
    print(f"Total conversaciones evaluadas: {len(pruebas)}")
    print(f"Precisión en detección de intenciones: {intenciones_correctas}/{len(pruebas)} ({intenciones_correctas/len(pruebas)*100:.1f}%)")
    print(f"Errores de formato JSON: {errores_json}")
    print("Detalle completo guardado en 'resultados_v1.json' para revisar datos extraídos y respuestas inventadas.")


if __name__ == "__main__":
    evaluar_dataset()