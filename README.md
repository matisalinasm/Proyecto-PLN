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
| `ORIGEN_V1.md` | Cómo se generó la versión 1 con Gemini, con el link a la conversación |
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
| 2. Explicar el código y comparar versiones simplificadas | Lista: V1 vs V1b (prompt mejorado) vs V2 (sin LLM), ver `fase2/` |
| 3. Repositorio con el historial de cambios | Lista |
| 4. Presentación de 15 minutos | Pendiente |

## Resultados de la versión 1 (línea base)

| Métrica | Resultado |
|---|---|
| Intenciones correctas | 46/50 (92%) |
| Datos bien extraídos (comparación exacta) | 85/95 campos (90%) |
| Conversaciones con alucinación | 12/50 (24%): 7 graves y 5 leves |
| Alucinaciones en las trampas | 9/15 (60%) |
| Acciones falsas ("ya cancelé tu reserva") | 6/50 |

Lo que más falla es la alucinación. Aunque el prompt dice explícitamente "NO inventes", el modelo inventó sabores de jugo, precios de platos que no existen, un correo de contacto y medios de pago. También aceptó una reserva para 18 personas cuando el máximo es 12.

Para contar las alucinaciones usamos un criterio estricto. Cuenta como alucinación todo lo que el bot afirma y no está en la información oficial. Las separamos en dos tipos. Las **graves** inventan un dato o rompen una regla del local, como los precios inventados o la reserva para 18. Las **leves** afirman algo que la información oficial no respalda, como decir "no tenemos estacionamiento" sin tener cómo saberlo, o se contradicen. Usamos el mismo criterio para todas las versiones, para que la comparación sea justa.

Aparte de eso encontramos otro problema. En 6 conversaciones el bot dice "tu reserva está hecha" o "ya cancelé tu reserva", pero no tiene ningún sistema de reservas detrás. No lo contamos como alucinación porque no inventa un dato del restaurante, pero le promete al cliente algo que no hizo.

En la extracción de datos el problema principal es la fecha (18 de 24 bien). Esta cifra compara los valores tal cual; en la fase 2 se normalizan antes de comparar ("este domingo" = "domingo"), por eso ahí la V1 sube a un F1 de 0,95. El modelo a veces no la guarda, sobre todo en cancelaciones y modificaciones. Además inventó 11 claves que no están en el esquema, como `plato` o `consulta`.

## Fase 2: explicación del código y versiones simplificadas

Todo está en [`fase2/`](fase2/). El documento principal es [`fase2/EXPLICACION_FASE2.md`](fase2/EXPLICACION_FASE2.md): explica cada parte de la V1 con los contenidos del curso y la compara con su reemplazo clásico (TF-IDF, embeddings, regex, NER, autómata de diálogo, recuperación de hechos). Las tablas están en [`fase2/resultados/RESULTADOS.md`](fase2/resultados/RESULTADOS.md). Para revisar el código y los parámetros ajustables (todos en `fase2/config.py`), ver [`fase2/GUIA_REVISION.md`](fase2/GUIA_REVISION.md).

```bash
pip install -r requirements.txt
python -m spacy download es_core_news_md
cd fase2
python Version2.py --debug       # chat de la V2 (sin LLM)
python interfaz_chat.py          # el mismo chat, en el navegador (http://127.0.0.1:7860)
python experimentos.py           # regenera todas las tablas
python correr_llm.py v1b todos   # V1 con prompt mejorado (requiere Ollama)
```
