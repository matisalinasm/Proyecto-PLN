"""
Sondeo (Clase 08): ¿sirven los word embeddings estáticos para decidir si una palabra
desconocida es COMIDA (mundo cerrado: "no está en la carta") o un SERVICIO (mundo
abierto: "no tengo información")?

Se compara la similitud coseno de cada palabra con dos prototipos (promedio de
vectores). Resultado: falla en casos claros ("terraza", "perro", "horno" salen como
comida), porque los vectores estáticos reflejan co-ocurrencia ("perro caliente",
"terraza del restaurante"), no categorías. Por eso respuesta.py decide con la
intención detectada y no con embeddings.

    python sondeo_embeddings.py
"""
import numpy as np

from preprocesamiento import nlp

PROTO_COMIDA = "comida plato bebida postre trago menú"
PROTO_SERVICIO = "servicio instalación pago lugar atención mueble"
PALABRAS = {  # palabra -> categoría correcta
    "sopaipillas": "comida", "porotos": "comida", "chorrillana": "comida", "pisco": "comida",
    "cerveza": "comida", "terremoto": "comida", "completo": "comida", "lomo": "comida",
    "estacionamiento": "servicio", "despacho": "servicio", "terraza": "servicio", "teléfono": "servicio",
    "whatsapp": "servicio", "perro": "servicio", "sillas": "servicio", "promoción": "servicio",
    "horno": "servicio", "wifi": "servicio",
}


def coseno(a, b):
    return float(a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    comida, servicio = nlp()(PROTO_COMIDA).vector, nlp()(PROTO_SERVICIO).vector
    aciertos = 0
    print(f"{'palabra':16s} {'sim comida':>10s} {'sim servicio':>12s}  predicción  correcta")
    for w, real in PALABRAS.items():
        v = nlp()(w).vector
        sc, ss = coseno(v, comida), coseno(v, servicio)
        pred = "comida" if sc > ss else "servicio"
        aciertos += pred == real
        print(f"{w:16s} {sc:10.2f} {ss:12.2f}  {pred:10s}  {'✓' if pred == real else '✗ (' + real + ')'}")
    print(f"\nAciertos: {aciertos}/{len(PALABRAS)}")


if __name__ == "__main__":
    main()
