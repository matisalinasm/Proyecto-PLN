# Origen de la versión 1 (etapa 1)

La versión 1 la generó **Gemini** el 19 de noviembre de 2026, solo con prompt engineering. La conversación completa se puede ver acá:

https://share.gemini.google/jlwPQqCXJgrc

La conversación se hizo dentro de un Proyecto de Gemini, por lo que el modelo también tenía las instrucciones personalizadas de ese proyecto.

## Cómo se llegó al código

No le dimos un único prompt con todo. Fuimos conversando con Gemini paso a paso. Primero le contamos que teníamos que hacer el proyecto 3 y le subimos el enunciado. Después decidimos juntos que el servicio sería un restaurante y que el bot ayudaría a gestionar reservas. Luego nos guió para instalar Ollama, gemma3:4b, Python y VS Code, y finalmente nos entregó el código completo del chatbot.

Estos fueron los mensajes que le escribimos, en orden:

1. debo hacer un proyecto en mi ramo de procesamiento del lenguaje natural, me interesa hacer el proyecto numero 3
2. podria ser un restaurante el servicio elegido? que el modelo ayude a gestionar reservas o algo asi?
3. okay, ahora que definimos que usaremos un restaurante, tengo que poder ejecutar el modelo de forma local, no? como uso gemma3: 4b o qwen3: 4b? o cual es el siguiente paso que debo hacer ahora que defini el servicio de restaurante?
4. wait, y como se si es mejor utilizar qwen o gemma?
5. okay, instale ollama pero no inicie sesion, ya que lo usaremos de forma local, no?
6. como abro el terminal en mi pc?
7. okay, estoy instalando gemma en el terminal, quedara guardada la informacion el computador o debo descargarlo cada vez?
8. okay, ya verifique que ollama quedo instalado con gemma localmente, ahora que sigue
9. puede ser un doc de google colab?
10. y visual studio code?
11. puedo cerrar la terminal el pc?
12. okay, acabo de instalar visual studio code e inicie sesion, que hago ahora?
13. para instalar ollama es new file? y ahi ejecuto el code?
14. *(capturas de pantalla de errores al instalar Python y ollama)* → aquí Gemini entregó el código de la versión 1
15. pero esas cosas que le pregunte contaran para las 50 preguntas que debo hacerle, no?
16. me esta funcionando bien por ahora, tengo que hacerle un par de preguntas mas
17. me invento un monton de informacion que no estan en sus fuentes...
18. pasemos a la etapa de creacion del archivo con las preguntas → aquí Gemini entregó `conversaciones_prueba.json` y el script de evaluación

## Verificación

Comparamos el código que aparece en la conversación con `Version1.py` del repositorio y son idénticos. La única diferencia son líneas en blanco entre funciones. Gemini nombró el archivo `v1_cruda.py` y en el repo quedó como `Version1.py`, pero el contenido no se tocó.

## Observaciones de las pruebas manuales

Al probar la versión 1 a mano, antes de armar las 50 conversaciones, ya vimos alucinaciones. El modelo inventó sabores de jugo (naranja, mango, maracuyá, yuyo y lúcuma) y propuso dividir un grupo grande en dos reservas, algo que no está en la política oficial.
