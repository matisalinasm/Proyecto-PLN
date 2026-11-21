# Asistente de atención "La Picada de Lang-Chi"

Proyecto del curso Procesamiento del Lenguaje Natural (UAI, 2026). Elegimos el proyecto 3, un chatbot en Python que atiende a clientes de un restaurante chileno ficticio. El bot detecta qué quiere el cliente (reservar, cancelar, consultar la carta, etc.), extrae los datos que entrega y responde sin inventar información.

Usamos solo modelos locales con [Ollama](https://ollama.com), en particular `gemma3:4b`.

## Cómo correrlo

```bash
pip install -r requirements.txt
ollama pull gemma3:4b

python Version1.py          # chatear con la versión 1
python Evaluar_V1.py        # correr las 50 conversaciones de prueba (genera resultados_v1.json)
python metricas.py v1       # calcular las 3 métricas a partir de esos resultados
```

## Archivos

| Archivo | Para qué sirve |
|---|---|
| `Version1.py` | Versión 1 cruda, tal como la entregó la IA. No se modifica |
| `Evaluar_V1.py` | Evaluador generado junto con la versión 1. Corre las conversaciones de prueba contra el modelo |
| `conversaciones_prueba.json` | 50 conversaciones de prueba con la intención y los datos esperados. 15 son "trampas" para ver si el bot inventa cosas |
| `resultados_v1.json` | Lo que respondió la versión 1 en cada conversación |
| `revision_alucinaciones_v1.json` | Revisión manual de qué respuestas de la versión 1 inventan información |
| `metricas.py` | Calcula las 3 métricas del enunciado sin volver a llamar al modelo |
| `metricas_v1.json` | Métricas de la versión 1 con el detalle de cada error |

## Etapas del proyecto

| Etapa | Estado |
|---|---|
| 1. Primera versión con IA, solo con prompt engineering | Lista |
| 2. Explicar el código y comparar versiones simplificadas | En curso |
| 3. Repositorio con el historial de cambios | En curso |
| 4. Presentación de 15 minutos | Pendiente |

## Resultados de la versión 1 (línea base)

| Métrica | Resultado |
|---|---|
| Intenciones correctas | 46/50 (92%) |
| Datos bien extraídos | 85/95 campos (90%) |
| Conversaciones con alucinación | 10/50 (20%) |
| Alucinaciones en las trampas | 9/15 (60%) |

Lo que más falla es la alucinación. Aunque el prompt dice explícitamente "NO inventes", el modelo inventó sabores de jugo, precios de platos que no existen, un correo de contacto y medios de pago. También aceptó una reserva para 18 personas cuando el máximo es 12.

En la extracción de datos el problema principal es la fecha (18 de 24 bien). El modelo a veces no la guarda, sobre todo en cancelaciones y modificaciones. Además inventó 11 claves que no están en el esquema, como `plato` o `consulta`.
