# Resultados de la Fase 2

Generado por `python experimentos.py`. Las salidas de la V1 sobre las 50 conversaciones de prueba están en `../resultados_v1.json`; las de la V1 y la V1b en los demás conjuntos se generaron con `correr_llm.py` (gemma3:4b en Ollama) y están en `resultados/llm_*.json`.

## E1 · Intención (Clases 02, 06, 08)

Cada modelo clasifica el primer mensaje de cada conversación. `CV` = F1-macro con validación cruzada de 5 particiones sobre el corpus de entrenamiento (179 frases): **es el criterio para elegir el modelo**, porque no mira los datos de prueba. Las reglas no se entrenan: su CV es su desempeño directo sobre el corpus.

| Modelo | CV F1 | Prueba acc | Prueba F1 | Holdout acc | Holdout F1 |
|---|---|---|---|---|---|
| reglas | 0.81 | 100% | 1.00 | 48% | 0.47 |
| bow_nb | 0.55 | 84% | 0.79 | 84% | 0.78 |
| tfidf_lr | 0.60 | 94% | 0.93 | 72% | 0.64 |
| tfidf_stem_lr | 0.68 | 94% | 0.93 | 76% | 0.68 |
| tfidf_lema_lr | 0.64 | 88% | 0.84 | 64% | 0.57 |
| tfidf_char_lr | 0.63 | 88% | 0.87 | 76% | 0.74 |
| tfidf_knn | 0.61 | 86% | 0.85 | 64% | 0.55 |
| lsa_lr | 0.60 | 86% | 0.81 | 72% | 0.66 |
| emb_lr | 0.64 | 84% | 0.77 | 64% | 0.65 |
| emb_knn | 0.53 | 54% | 0.50 | 56% | 0.54 |
| hibrido_lr | 0.70 | 96% | 0.95 | 92% | 0.92 |
| **LLM V1 (gemma3:4b)** | — | 92% | 0.90 | 80% | 0.73 |

Errores del modelo elegido (`hibrido_lr`): prueba: #18 cancelar_reserva→hacer_reserva, #19 cancelar_reserva→modificar_reserva; holdout: #118 consultar_info_local→fuera_de_dominio, #125 fuera_de_dominio→consultar_info_local

## E2 · Extracción de datos (Clases 04, 05, 06)

F1 por campo después de normalizar valores ('este domingo' = 'domingo', '14:00 hrs' = '14:00', 'celíaca' = 'sin gluten'). `fuera esq.` = conversaciones donde se inventaron claves que no existen.

| Conjunto | Extractor | F1 total | fecha | hora | cantidad_personas | nombre_cliente | sucursal | restriccion_alimentaria | fuera esq. |
|---|---|---|---|---|---|---|---|---|---|
| prueba | LLM V1 | 0.95 | 0.86 | 0.92 | 1.00 | 1.00 | 0.97 | 0.86 | 7 |
| prueba | regex | 0.99 | 0.98 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0 |
| prueba | ner_spacy | 0.97 | 0.98 | 1.00 | 1.00 | 0.88 | 1.00 | 1.00 | 0 |
| prueba | hibrido | 0.99 | 0.98 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0 |
| holdout | LLM V1 | 0.97 | 0.90 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 4 |
| holdout | regex | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0 |
| holdout | ner_spacy | 0.97 | 1.00 | 1.00 | 1.00 | 0.86 | 1.00 | 1.00 | 0 |
| holdout | hibrido | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0 |

### Desambiguación léxica (WSD) de 'mañana' (día/franja) y 'once' (número/comida)

| Método | Fecha correcta | Personas correcta |
|---|---|---|
| ingenua | 10/12 | 9/12 |
| colocacional (regex) | 12/12 | 12/12 |
| POS spaCy (fecha) | 12/12 | (usa colocacional) |

## E3 · Diálogo (Clases 02 y 07)

`completas` = intención final correcta y TODOS los datos exactos. El set multiturno se usó para desarrollar las reglas de discurso (es optimista); las 4 conversaciones multiturno de prueba+holdout no se usaron.

| Conjunto | Sistema | Intención final | F1 datos | Completas |
|---|---|---|---|---|
| multiturno (desarrollo) | sin estado (cada turno aislado) | 25% | 0.52 | 3/12 |
| multiturno (desarrollo) | autómata + discurso (V2) | 92% | 1.00 | 11/12 |
| multiturno (desarrollo) | LLM V1 (historial) | 92% | 0.76 | 7/12 |
| multiturno (desarrollo) | LLM V1b (historial) | 75% | 0.91 | 8/12 |
| multiturno de prueba+holdout | sin estado (cada turno aislado) | 25% | 0.69 | 1/4 |
| multiturno de prueba+holdout | autómata + discurso (V2) | 100% | 1.00 | 4/4 |

Fallos de la V2 en el set multiturno: #207 (cambio de acto de habla (saludo -> directivo) y luego slot filling): esperada cancelar_reserva, obtenida hacer_reserva

## E4 · Respuestas fundamentadas (Clases 06, 07, 10, 12)

Solo consultas (menú/local) con etiqueta en `datos/etiquetas_grounding.json`. `fuentes` = menciona los hechos correctos; `negar` = dice que algo no existe cuando corresponde (carta/sucursales completas); `abstenerse` = dice 'no tengo información' cuando la base no lo sabe; `todo` = las tres a la vez. `niega sin saber` = afirma que algo NO existe cuando en realidad no se sabe (el error 'leve' de la V1). `$ sin respaldo` = montos que no son precios de la carta.

**`ciegas` se escribió después de construir el módulo y nunca se usó para ajustarlo: es la medida más honesta.**

| Conjunto | Sistema | n | fuentes | negar | abstenerse | todo | niega sin saber | $ sin respaldo |
|---|---|---|---|---|---|---|---|---|
| prueba | LLM V1 | 22 | 100% | 95% | 64% | **59%** | 6 | 1 |
| prueba | LLM V1b (prompt mejorado) | 22 | 95% | 86% | 91% | **73%** | 1 | 0 |
| prueba | V2 completa | 22 | 100% | 95% | 95% | **91%** | 0 | 0 |
| prueba | V2 sin 'No lo sé' (solo hechos recuperados) | 22 | 100% | 91% | 64% | **59%** | 3 | 0 |
| prueba | V2 sin mundo cerrado (siempre se abstiene) | 22 | 100% | 86% | 100% | **86%** | 0 | 0 |
| holdout | LLM V1 | 10 | 90% | 90% | 80% | **60%** | 0 | 0 |
| holdout | LLM V1b (prompt mejorado) | 10 | 90% | 90% | 100% | **80%** | 0 | 0 |
| holdout | V2 completa | 10 | 100% | 100% | 90% | **90%** | 0 | 0 |
| holdout | V2 sin 'No lo sé' (solo hechos recuperados) | 10 | 100% | 100% | 70% | **70%** | 1 | 0 |
| holdout | V2 sin mundo cerrado (siempre se abstiene) | 10 | 100% | 90% | 90% | **80%** | 0 | 0 |
| multiturno | LLM V1 | 4 | 75% | 100% | 100% | **75%** | 0 | 0 |
| multiturno | LLM V1b (prompt mejorado) | 4 | 75% | 100% | 100% | **75%** | 0 | 0 |
| multiturno | V2 completa | 4 | 100% | 100% | 100% | **100%** | 0 | 0 |
| multiturno | V2 sin 'No lo sé' (solo hechos recuperados) | 4 | 100% | 100% | 100% | **100%** | 0 | 0 |
| multiturno | V2 sin mundo cerrado (siempre se abstiene) | 4 | 100% | 100% | 100% | **100%** | 0 | 0 |
| ciegas | LLM V1 | 15 | 87% | 93% | 60% | **40%** | 5 | 0 |
| ciegas | LLM V1b (prompt mejorado) | 15 | 87% | 93% | 93% | **73%** | 0 | 0 |
| ciegas | V2 completa | 15 | 100% | 100% | 87% | **87%** | 1 | 0 |
| ciegas | V2 sin 'No lo sé' (solo hechos recuperados) | 15 | 100% | 100% | 60% | **60%** | 3 | 0 |
| ciegas | V2 sin mundo cerrado (siempre se abstiene) | 15 | 100% | 87% | 93% | **80%** | 0 | 0 |

### Sensibilidad al umbral de recuperación TAU (config: 0.15, fijado antes de evaluar)

| TAU | prueba (todo) | holdout (todo) | ciegas (todo) |
|---|---|---|---|
| 0.05 | 91% | 90% | 87% |
| 0.1 | 91% | 90% | 87% |
| 0.15 | 91% | 90% | 87% |
| 0.2 | 91% | 90% | 87% |
| 0.3 | 91% | 80% | 80% |
| 0.4 | 77% | 70% | 60% |

## E5 · Sistema completo (las tres métricas del enunciado)

Intención y datos del último turno de cada conversación. Respuestas inventadas: anotación manual (`datos/anotacion_alucinaciones_*.json`); `grave` = inventa un dato o viola una política; `leve` = afirma algo que la base no respalda; `acción falsa` = dice haber hecho algo que el sistema no hace ('ya cancelé').

| Conjunto | Sistema | Intención acc | Intención F1 | F1 datos | fuera esq. | grave | leve | acción falsa | sin anotar |
|---|---|---|---|---|---|---|---|---|---|
| prueba | LLM V1 (cruda) | 92% | 0.90 | 0.95 | 7 | 7 | 5 | 6 | 0 |
| prueba | LLM V1b (prompt mejorado) | 92% | 0.91 | 0.97 | 0 | 4 | 1 | 20 | 0 |
| prueba | V2 (sin LLM) | 94% | 0.93 | 0.99 | 0 | 0 | 1 | 0 | 0 |
| holdout | LLM V1 (cruda) | 80% | 0.73 | 0.97 | 4 | 4 | 2 | 5 | 0 |
| holdout | LLM V1b (prompt mejorado) | 92% | 0.91 | 1.00 | 0 | 2 | 0 | 8 | 0 |
| holdout | V2 (sin LLM) | 92% | 0.92 | 1.00 | 0 | 0 | 0 | 0 | 0 |

### Prompt injection y datos sensibles (`datos/casos_injection.json`)

| Sistema | Casos resistidos |
|---|---|
| V2 (sin LLM) | 5/5 |
| LLM V1b | 3/5 |

### Eficiencia (Clase 12)

| Sistema | Latencia media por turno | p95 | Modelo |
|---|---|---|---|
| V2 (sin LLM) | 13 ms | 24 ms | spaCy es_core_news_md (~50 MB en disco) + TF-IDF |
| LLM V1 (holdout) | 2.0 s | 6.4 s | gemma3:4b cuantizado (GGUF 4 bits en Ollama) |
| LLM V1B (holdout) | 1.0 s | 1.1 s | gemma3:4b cuantizado (GGUF 4 bits en Ollama) |

## E6 · Ablación: qué aporta cada componente

Se cambia UNA pieza a la vez. Holdout y ciegas no se usaron para diseñar; multiturno sí (desarrollo).

| Configuración | Holdout intención | Holdout F1 datos | Multiturno intención | Multiturno F1 datos | Ciegas grounding |
|---|---|---|---|---|---|
| V2 completa | 92% | 1.00 | 92% | 1.00 | 87% |
| clasificador → reglas | 48% | 0.95 | 100% | 1.00 | 47% |
| clasificador → TF-IDF char | 76% | 1.00 | 100% | 1.00 | 73% |
| clasificador → embeddings | 72% | 1.00 | 75% | 1.00 | 80% |
| extractor → solo regex | 92% | 1.00 | 92% | 1.00 | 87% |
| extractor → solo NER spaCy | 92% | 0.97 | 75% | 0.94 | 87% |
| sin seguimiento de diálogo | 84% | 0.95 | 25% | 0.52 | 87% |
