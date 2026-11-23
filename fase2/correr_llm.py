"""
Ejecuta las versiones con LLM sobre los conjuntos de evaluación y guarda sus salidas
en resultados/ para que experimentos.py las compare con la V2. REQUIERE Ollama con gemma3:4b
(los resultados ya están guardados en resultados/llm_*.json; solo hay que volver a correrlo si se cambia
el prompt o el modelo).

    python correr_llm.py v1  holdout        # V1 cruda (prompt original) sobre el holdout
    python correr_llm.py v1  multiturno
    python correr_llm.py v1b prueba         # V1b (prompt mejorado) sobre las 50 de prueba
    python correr_llm.py v1b todos          # V1b sobre prueba, holdout, multiturno, ciegas e injection

La V1 sobre 'prueba' ya existe: es ../resultados_v1.json (la generó Evaluar_V1.py).
"""
import json
import sys
import time
from pathlib import Path

AQUI = Path(__file__).parent
sys.path.insert(0, str(AQUI.parent))   # para importar ../Version1.py sin modificarlo


def cargar(conjunto):
    if conjunto == "prueba":
        return json.loads((AQUI.parent / "conversaciones_prueba.json").read_text(encoding="utf-8"))
    if conjunto in {"holdout", "multiturno"}:
        nombre = "conversaciones_holdout.json" if conjunto == "holdout" else "conversaciones_multiturno.json"
        return json.loads((AQUI / "datos" / nombre).read_text(encoding="utf-8"))
    if conjunto == "ciegas":
        return [{"id": c["id"], "turnos": [c["texto"]]} for c in
                json.loads((AQUI / "datos" / "consultas_ciegas.json").read_text(encoding="utf-8"))["casos"]]
    if conjunto == "injection":
        return [{"id": c["id"], "turnos": [c["texto"]]} for c in
                json.loads((AQUI / "datos" / "casos_injection.json").read_text(encoding="utf-8"))["casos"]]
    raise ValueError(conjunto)


class AsistenteV1:
    """La V1 tal cual (mismo prompt, mismos parámetros que Evaluar_V1.py)."""

    def __init__(self, cliente=None):
        from Version1 import MODELO, SYSTEM_PROMPT
        if cliente is None:
            import ollama
            cliente = ollama
        self.ollama, self.modelo = cliente, MODELO
        self.historial = [{"role": "system", "content": SYSTEM_PROMPT}]

    def procesar(self, mensaje):
        t0 = time.time()
        self.historial.append({"role": "user", "content": mensaje})
        r = self.ollama.chat(model=self.modelo, messages=self.historial, format="json",
                             options={"temperature": 0.2})
        contenido = r["message"]["content"]
        self.historial.append({"role": "assistant", "content": contenido})
        try:
            d = json.loads(contenido)
        except json.JSONDecodeError:
            d = {"intencion": "ERROR_JSON", "datos_extraidos": {}, "respuesta": contenido}
        d["latencia_s"] = round(time.time() - t0, 2)
        return d


def correr(version, conjunto, cliente=None):
    from Version1b_prompt import AsistenteV1b
    clase = AsistenteV1 if version == "v1" else AsistenteV1b
    salidas = []
    for caso in cargar(conjunto):
        bot = clase(cliente)
        lat = []
        for turno in caso["turnos"]:
            r = bot.procesar(turno)
            lat.append(r.get("latencia_s", 0))
        salidas.append({"id": caso["id"], "intencion": r.get("intencion"),
                        "datos_extraidos": r.get("datos_extraidos", {}), "respuesta": r.get("respuesta", ""),
                        "latencia_s": round(sum(lat) / len(lat), 2),
                        "problemas_guardrail": r.get("problemas_guardrail", [])})
        print(f"#{caso['id']} -> {r.get('intencion')}")
    destino = AQUI / "resultados" / f"llm_{version}_{conjunto}.json"
    destino.write_text(json.dumps(salidas, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Guardado {destino}")


if __name__ == "__main__":
    version, conjunto = sys.argv[1], sys.argv[2]
    conjuntos = ["prueba", "holdout", "multiturno", "ciegas", "injection"] if conjunto == "todos" else [conjunto]
    for c in conjuntos:
        if version == "v1" and c == "prueba":
            print("La V1 sobre 'prueba' ya está en ../resultados_v1.json; se omite.")
            continue
        correr(version, c)
