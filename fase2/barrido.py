"""
Prueba varios valores de UN parámetro de config.py y muestra su efecto, sin editar archivos.

    python barrido.py LR_C 0.1 1 10 100
    python barrido.py TAU 0.05 0.15 0.30 0.50
    python barrido.py UMBRAL_CAMBIO 0.5 0.75 0.9
    python barrido.py CLASIFICADOR tfidf_lr emb_lr hibrido_lr
    python barrido.py CHAR_NGRAMAS "(1,3)" "(2,5)" "(3,6)"

Columnas:
  CV F1        validación cruzada sobre el corpus de ENTRENAMIENTO. Es la única columna que
               se puede usar para ELEGIR un valor sin hacer trampa (Clase 11: data leakage).
  Holdout      intención y F1 de datos en las 25 conversaciones nuevas.
  Multiturno   intención final en las 12 conversaciones de varios turnos.
  Ciegas       % de consultas nunca vistas respondidas bien (hechos + negar + abstenerse).
Las tres últimas sirven para ENTENDER el efecto, no para escoger el valor mirando el
resultado: si se elige así, esas métricas dejan de ser una medida honesta.
"""
import ast
import sys
import warnings

from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import config
import experimentos as E
from intencion import cargar_entrenamiento, construir

warnings.filterwarnings("ignore")


def interpretar(valor: str, actual):
    """Convierte el texto de la terminal al tipo del valor actual ('10' -> 10, '(2,5)' -> (2, 5))."""
    if isinstance(actual, str):
        return valor
    return ast.literal_eval(valor)


def evaluar():
    X, y = cargar_entrenamiento()
    cv = StratifiedKFold(config.CV_PARTICIONES, shuffle=True, random_state=config.SEMILLA)
    modelo = construir(config.CLASIFICADOR)
    pred = modelo.predict(X) if config.CLASIFICADOR == "reglas" else cross_val_predict(modelo, X, y, cv=cv)
    E._CLF.clear()   # el chatbot se reentrena con el valor nuevo
    s_h, _ = E.correr_v2(E.HOLDOUT)
    s_m, _ = E.correr_v2(E.MULTI)
    s_c, _ = E.correr_v2(E.CIEGAS)
    m_h, m_m = E.metricas(s_h, E.HOLDOUT), E.metricas(s_m, E.MULTI)
    n, r = E.evaluar_grounding(s_c, E.CIEGAS)
    return [f"{f1_score(y, pred, average='macro'):.2f}", f"{m_h['acc']:.0%}", f"{m_h['f1_datos']:.2f}",
            f"{m_m['acc']:.0%}", f"{r['todo'] / n:.0%}"]


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return
    nombre, valores = sys.argv[1], sys.argv[2:]
    if not hasattr(config, nombre):
        print(f"'{nombre}' no existe en config.py. Disponibles: "
              + ", ".join(k for k in vars(config) if k.isupper()))
        return
    original = getattr(config, nombre)
    filas = []
    for v in valores:
        setattr(config, nombre, interpretar(v, original))
        filas.append([v + (" (actual)" if interpretar(v, original) == original else "")] + evaluar())
        print(f"  {nombre} = {v} listo")
    setattr(config, nombre, original)
    print()
    print(E.tabla([nombre, "CV F1", "Holdout intención", "Holdout F1 datos", "Multiturno intención",
                   "Ciegas (todo bien)"], filas))


if __name__ == "__main__":
    main()
