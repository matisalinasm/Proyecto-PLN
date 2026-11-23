# Resultados de la Fase 2

Generado por `python experimentos.py`. La V1 solo tiene salidas sobre las 50 conversaciones de prueba (`resultados_v1.json`); para sumarla en los otros conjuntos, correr `correr_llm.py` con Ollama.

## E1 · Intención (Clases 02, 06, 08)

Cada modelo clasifica el primer mensaje de cada conversación. `CV` = F1-macro con validación cruzada de 5 particiones sobre el corpus de entrenamiento (150 frases): **es el criterio para elegir el modelo**, porque no mira los datos de prueba. Las reglas no se entrenan: su CV es su desempeño directo sobre el corpus.

| Modelo | CV F1 | Prueba acc | Prueba F1 | Holdout acc | Holdout F1 |
|---|---|---|---|---|---|
| reglas | 0.80 | 100% | 1.00 | 48% | 0.47 |
| bow_nb | 0.58 | 82% | 0.77 | 80% | 0.74 |
| tfidf_lr | 0.58 | 90% | 0.86 | 72% | 0.64 |
| tfidf_stem_lr | 0.65 | 94% | 0.91 | 80% | 0.70 |
| tfidf_lema_lr | 0.57 | 90% | 0.85 | 72% | 0.64 |
| tfidf_char_lr | 0.61 | 88% | 0.84 | 84% | 0.82 |
| tfidf_knn | 0.54 | 84% | 0.84 | 68% | 0.62 |
| lsa_lr | 0.62 | 84% | 0.79 | 68% | 0.62 |
| emb_lr | 0.68 | 84% | 0.78 | 68% | 0.69 |
| emb_knn | 0.57 | 52% | 0.48 | 60% | 0.52 |
| hibrido_lr | 0.72 | 94% | 0.91 | 88% | 0.85 |
| **LLM V1 (gemma3:4b)** | — | 92% | 0.90 | — | — |

Errores del modelo elegido (`hibrido_lr`): prueba: #16 cancelar_reserva→modificar_reserva, #18 cancelar_reserva→hacer_reserva, #19 cancelar_reserva→modificar_reserva; holdout: #110 cancelar_reserva→saludo_despedida, #118 consultar_info_local→fuera_de_dominio, #125 fuera_de_dominio→consultar_info_local

## E2 · Extracción de datos (Clases 04, 05, 06)

F1 por campo después de normalizar valores ('este domingo' = 'domingo', '14:00 hrs' = '14:00', 'celíaca' = 'sin gluten'). `fuera esq.` = conversaciones donde se inventaron claves que no existen.

| Conjunto | Extractor | F1 total | fecha | hora | cantidad_personas | nombre_cliente | sucursal | restriccion_alimentaria | fuera esq. |
|---|---|---|---|---|---|---|---|---|---|
| prueba | LLM V1 | 0.95 | 0.86 | 0.92 | 1.00 | 1.00 | 0.97 | 0.86 | 7 |
| prueba | regex | 0.99 | 0.98 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0 |
| prueba | ner_spacy | 0.97 | 0.98 | 1.00 | 1.00 | 0.88 | 1.00 | 1.00 | 0 |
| prueba | hibrido | 0.99 | 0.98 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0 |
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
| multiturno (desarrollo) | sin estado (cada turno aislado) | 17% | 0.45 | 2/12 |
| multiturno (desarrollo) | autómata + discurso (V2) | 92% | 1.00 | 11/12 |
| multiturno de prueba+holdout | sin estado (cada turno aislado) | 25% | 0.69 | 1/4 |
| multiturno de prueba+holdout | autómata + discurso (V2) | 100% | 1.00 | 4/4 |

Fallos de la V2 en el set multiturno: #207 (cambio de acto de habla (saludo -> directivo) y luego slot filling): esperada cancelar_reserva, obtenida hacer_reserva

## E4 · Respuestas fundamentadas (Clases 06, 07, 10, 12)

Solo consultas (menú/local) con etiqueta en `datos/etiquetas_grounding.json`. `fuentes` = menciona los hechos correctos; `negar` = dice que algo no existe cuando corresponde (carta/sucursales completas); `abstenerse` = dice 'no tengo información' cuando la base no lo sabe; `todo` = las tres a la vez. `niega sin saber` = afirma que algo NO existe cuando en realidad no se sabe (el error 'leve' de la V1). `$ sin respaldo` = montos que no son precios de la carta.

**`ciegas` se escribió después de construir el módulo y nunca se usó para ajustarlo: es la medida más honesta.**

| Conjunto | Sistema | n | fuentes | negar | abstenerse | todo | niega sin saber | $ sin respaldo |
|---|---|---|---|---|---|---|---|---|
| prueba | LLM V1 | 22 | 100% | 95% | 64% | **59%** | 6 | 1 |
| prueba | V2 completa | 22 | 100% | 95% | 95% | **91%** | 0 | 0 |
| prueba | V2 sin 'No lo sé' (solo hechos recuperados) | 22 | 100% | 91% | 64% | **59%** | 3 | 0 |
| prueba | V2 sin mundo cerrado (siempre se abstiene) | 22 | 100% | 86% | 100% | **86%** | 0 | 0 |
| holdout | V2 completa | 10 | 100% | 100% | 90% | **90%** | 0 | 0 |
| holdout | V2 sin 'No lo sé' (solo hechos recuperados) | 10 | 100% | 100% | 70% | **70%** | 1 | 0 |
| holdout | V2 sin mundo cerrado (siempre se abstiene) | 10 | 100% | 90% | 90% | **80%** | 0 | 0 |
| multiturno | V2 completa | 4 | 100% | 100% | 100% | **100%** | 0 | 0 |
| multiturno | V2 sin 'No lo sé' (solo hechos recuperados) | 4 | 100% | 100% | 100% | **100%** | 0 | 0 |
| multiturno | V2 sin mundo cerrado (siempre se abstiene) | 4 | 100% | 100% | 100% | **100%** | 0 | 0 |
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
| prueba | V2 (sin LLM) | 92% | 0.90 | 0.99 | 0 | 0 | 1 | 0 | 0 |
| holdout | V2 (sin LLM) | 92% | 0.92 | 1.00 | 0 | 0 | 0 | 0 | 0 |

### Prompt injection y datos sensibles (`datos/casos_injection.json`)

| Sistema | Casos resistidos |
|---|---|
| V2 (sin LLM) | 5/5 |

### Eficiencia (Clase 12)

| Sistema | Latencia media por turno | p95 | Modelo |
|---|---|---|---|
| V2 (sin LLM) | 16 ms | 29 ms | spaCy es_core_news_md (~50 MB en disco) + TF-IDF |

## E6 · Ablación: qué aporta cada componente

Se cambia UNA pieza a la vez. Holdout y ciegas no se usaron para diseñar; multiturno sí (desarrollo).

| Configuración | Holdout intención | Holdout F1 datos | Multiturno intención | Multiturno F1 datos | Ciegas grounding |
|---|---|---|---|---|---|
| V2 completa | 92% | 1.00 | 92% | 1.00 | 87% |
| clasificador → reglas | 48% | 0.95 | 100% | 1.00 | 47% |
| clasificador → TF-IDF char | 88% | 1.00 | 100% | 1.00 | 80% |
| clasificador → embeddings | 76% | 1.00 | 67% | 0.94 | 73% |
| extractor → solo regex | 92% | 1.00 | 92% | 1.00 | 87% |
| extractor → solo NER spaCy | 92% | 0.97 | 67% | 0.81 | 87% |
| sin seguimiento de diálogo | 84% | 0.95 | 17% | 0.45 | 87% |
