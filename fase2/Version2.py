"""
Versión 2 (sin LLM): chat por terminal del asistente de "La Picada de Lang-Chi".

    python Version2.py            # componentes por defecto (los elegidos en los experimentos)
    python Version2.py --debug    # muestra intención, confianza, estado del autómata y hechos citados
"""
import sys
import warnings

from dialogo import Asistente

warnings.filterwarnings("ignore")


def main():
    debug = "--debug" in sys.argv
    bot = Asistente()
    print("=== Asistente 'La Picada de Lang-Chi' (Versión 2, sin LLM) ===")
    print("Escribe 'salir' para terminar.\n")
    while True:
        try:
            msg = input("Cliente: ").strip()
        except EOFError:
            break
        if not msg:
            continue
        if msg.lower() in {"salir", "exit", "quit"}:
            break
        r = bot.procesar(msg)
        if debug:
            print(f"  [intención: {r['intencion']} (clasificador: {r['intencion_clasificador']}, "
                  f"conf {r['confianza']}) | estado: {r['estado']} | datos: {r['datos_extraidos']} "
                  f"| hechos: {r['hechos']}]")
        print(f"Asistente: {r['respuesta']}\n")


if __name__ == "__main__":
    main()
