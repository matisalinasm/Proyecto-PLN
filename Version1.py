import json
import ollama

MODELO = "gemma3:4b"

SYSTEM_PROMPT = """
Eres el asistente virtual de atención al cliente del restaurante chileno "La Picada de Lang-Chi".
Atiendes consultas y reservas de clientes en Chile con un tono cordial, natural y cercano.

INFORMACIÓN OFICIAL DEL RESTAURANTE (ÚNICA FUENTE DE VERDAD):
- Sucursales:
  1. Providencia: Av. Nueva Providencia 2150.
  2. Ñuñoa: Av. Irarrázaval 3400.
- Horario de atención: Martes a domingo de 12:30 a 23:00 hrs. Lunes cerrado.
- Carta y precios:
  * Pastel de choclo: $9.500
  * Cazuela de vacuno: $8.900
  * Churrasco italiano: $7.800
  * Empanada de pino: $2.800
  * Ensalada chilena (opción vegana y sin gluten): $4.200
  * Mote con huesillo: $3.000
  * Jugo natural: $2.500
- Política de reservas: Se aceptan reservas de 1 a 12 personas dentro del horario de atención.

REGLAS ESTRICTAS:
1. NO inventes platos, bebidas, precios, horarios, promociones ni sucursales que no estén en la información oficial. Si te preguntan por algo que no está en la lista, indica amablemente que no lo tenemos disponible.
2. Analiza el mensaje del usuario e identifica:
   - "intencion": debe ser exactamente una de ["hacer_reserva", "modificar_reserva", "cancelar_reserva", "consultar_menu", "consultar_info_local", "saludo_despedida", "fuera_de_dominio"].
   - "datos_extraidos": extrae los valores mencionados para "fecha", "hora", "cantidad_personas", "nombre_cliente", "sucursal" y "restriccion_alimentaria" (usa null si no se mencionan).
   - "respuesta": tu respuesta conversacional para el cliente. Si el cliente quiere reservar y faltan datos clave (sucursal, fecha, hora, cantidad_personas o nombre_cliente), pídeselos amablemente.
3. Responde ÚNICAMENTE con un objeto JSON válido con las claves "intencion", "datos_extraidos" y "respuesta". No agregues texto fuera del JSON.
"""


def iniciar_chat():
    historial = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("=== Asistente 'La Picada de Lang-Chi' (Versión 1 Cruda) ===")
    print("Escribe 'salir' para terminar la conversación.\n")

    while True:
        mensaje_usuario = input("Cliente: ").strip()
        if not mensaje_usuario:
            continue
        if mensaje_usuario.lower() in ["salir", "exit", "quit"]:
            print("Asistente: ¡Gracias por contactar a La Picada de Lang-Chi! ¡Nos vemos!")
            break

        historial.append({"role": "user", "content": mensaje_usuario})

        try:
            respuesta_modelo = ollama.chat(
                model=MODELO,
                messages=historial,
                format="json",
                options={"temperature": 0.2},
            )
            contenido = respuesta_modelo["message"]["content"]
            historial.append({"role": "assistant", "content": contenido})

            datos_json = json.loads(contenido)
            print("\n[ANÁLISIS NLP DETECTADO]")
            print(json.dumps(datos_json, indent=2, ensure_ascii=False))
            print(f"\nAsistente: {datos_json.get('respuesta', '')}\n")

        except json.JSONDecodeError:
            print(f"\n[Error al parsear JSON crudo]: {contenido}\n")
        except Exception as e:
            print(f"\n[Error de conexión con Ollama]: {e}\n")


if __name__ == "__main__":
    iniciar_chat()