"""
Interfaz de chat en el navegador para probar la Versión 2 (sin LLM).

    pip install gradio
    python interfaz_chat.py          # abre http://127.0.0.1:7860

No cambia nada de la lógica: usa el mismo Asistente de dialogo.py que Version2.py.
El panel de la derecha muestra lo que el bot entendió en cada turno (intención,
estado del autómata, datos extraídos y hechos de la base que usó para responder).
"""
import warnings

import gradio as gr

import conocimiento as kb
from dialogo import Asistente
from intencion import Clasificador
from respuesta import Recuperador

warnings.filterwarnings("ignore")

# El clasificador y el buscador se cargan una sola vez; cada pestaña del navegador
# tiene su propio Asistente (su propia conversación y estado).
CLF = Clasificador()
REC = Recuperador()

NOMBRES_INTENCION = {
    "hacer_reserva": "Hacer una reserva", "modificar_reserva": "Modificar una reserva",
    "cancelar_reserva": "Cancelar una reserva", "consultar_menu": "Consultar la carta",
    "consultar_info_local": "Consultar sobre el local", "saludo_despedida": "Saludo / despedida",
    "fuera_de_dominio": "Fuera de tema",
}
NOMBRES_CAMPO = {"fecha": "Fecha", "hora": "Hora", "cantidad_personas": "Personas",
                 "nombre_cliente": "Nombre", "sucursal": "Sucursal", "restriccion_alimentaria": "Restricción"}
ESTADOS = ["INICIO", "RECOLECTANDO", "REGISTRADA"]
EJEMPLOS = [
    "Hola, quiero reservar para 4 el viernes a las 21:00 en Ñuñoa",
    "¿Cuánto cuesta el pastel de choclo?",
    "¿Tienen estacionamiento?",
    "¿Tienen opciones veganas?",
    "Quiero reservar para 15 personas el sábado",
    "¿Abren los lunes?",
]
PANEL_VACIO = "<div class='panel-vacio'>Escribe un mensaje para ver cómo lo interpreta el asistente.</div>"


def nuevo_asistente():
    return Asistente(clf=CLF, recuperador=REC)


def panel(r):
    """HTML del panel 'Qué entendió el bot'."""
    conf = int(round(r["confianza"] * 100))
    if r["intencion"] == r["intencion_clasificador"]:
        origen = f"Clasificador · confianza {conf}%"
    else:  # la decidió el seguimiento del diálogo (continuación, corrección o elipsis)
        origen = "Deducida del contexto de la conversación"
    pasos = "".join(
        f"<span class='estado {'activo' if e == r['estado'] else ''}'>{e.title()}</span>"
        + ("<span class='flecha'>→</span>" if i < len(ESTADOS) - 1 else "")
        for i, e in enumerate(ESTADOS))
    datos = {k: v for k, v in r["datos_extraidos"].items() if v is not None}
    filas = "".join(f"<tr><td>{NOMBRES_CAMPO.get(k, k)}</td><td><b>{v}</b></td></tr>" for k, v in datos.items())
    tabla = f"<table class='datos'>{filas}</table>" if filas else "<div class='nada'>Ninguno en este mensaje</div>"
    hechos = "".join(f"<li>{kb.HECHOS_POR_ID[h]['texto']}</li>" for h in r["hechos"])
    hechos = f"<ul class='hechos'>{hechos}</ul>" if hechos else "<div class='nada'>No usó la base en este turno</div>"
    return f"""
    <div class='panel'>
      <div class='seccion'>Intención detectada</div>
      <div class='intencion'>{NOMBRES_INTENCION.get(r['intencion'], r['intencion'])}</div>
      <div class='nada'>{origen}</div>
      <div class='seccion'>Etapa de la conversación</div>
      <div class='estados'>{pasos}</div>
      <div class='seccion'>Datos extraídos</div>
      {tabla}
      <div class='seccion'>Información usada de la base</div>
      {hechos}
    </div>"""


def responder(mensaje, historial, bot):
    mensaje = (mensaje or "").strip()
    if not mensaje:
        return "", historial, bot, gr.update()
    bot = bot or nuevo_asistente()
    r = bot.procesar(mensaje)
    historial = historial + [{"role": "user", "content": mensaje},
                             {"role": "assistant", "content": r["respuesta"]}]
    return "", historial, bot, panel(r)


def reiniciar():
    return [], nuevo_asistente(), PANEL_VACIO


CSS = """
.gradio-container {max-width: 1150px !important; margin: auto;}
#cabecera {text-align:center; padding: 18px 0 6px 0;}
#cabecera h1 {font-size: 2em; margin: 0; color: #1F4E79;}
#cabecera p {margin: 4px 0 0 0; color: #666;}
.panel {font-size: 0.95em;}
.seccion {font-weight: 700; color: #1F4E79; margin: 14px 0 6px 0; text-transform: uppercase;
          font-size: 0.78em; letter-spacing: .05em;}
.seccion:first-child {margin-top: 0;}
.intencion {font-size: 1.25em; font-weight: 700; margin-bottom: 2px;}
.estados {display:flex; align-items:center; flex-wrap: wrap; gap: 4px;}
.estado {padding: 3px 10px; border-radius: 12px; background: #E3E8EE; color: #666; font-size: 0.88em;}
.estado.activo {background: #1F4E79; color: white; font-weight: 700;}
.flecha {color: #999;}
.panel table.datos, .panel table.datos tr, .panel table.datos tbody {width: 100%; border: 0 !important;
    border-collapse: collapse; box-shadow: none !important;}
.panel table.datos td {padding: 5px 6px; border: 0 !important; border-bottom: 1px solid #E3E8EE !important;
    background: transparent !important;}
table.datos td:first-child {color: #666; width: 40%;}
ul.hechos {margin: 0; padding-left: 18px;}
.nada, .panel-vacio {color: #888; font-size: 0.9em;}
footer {display: none !important;}
"""


def construir():
    tema = gr.themes.Soft(primary_hue="blue", secondary_hue="green", neutral_hue="slate",
                          font=[gr.themes.Font(f) for f in ("Segoe UI", "Helvetica Neue", "Arial", "sans-serif")])
    with gr.Blocks(title="La Picada de Lang-Chi · Asistente") as demo:
        bot = gr.State(None)
        gr.HTML(f"<div id='cabecera'><h1>🍽️ {kb.NOMBRE_LOCAL}</h1>"
                "<p>Asistente virtual · Versión 2 (sin LLM) · Proyecto PLN, UAI</p></div>")
        with gr.Row():
            with gr.Column(scale=3):
                chat = gr.Chatbot(height=560, show_label=False, buttons=[], placeholder=(
                    "<b>¡Hola!</b> Puedo ayudarte con reservas, la carta, horarios y sucursales."))
                with gr.Row():
                    caja = gr.Textbox(placeholder="Escribe tu mensaje…", show_label=False, scale=5,
                                      autofocus=True, container=False)
                    enviar = gr.Button("Enviar", variant="primary", scale=1)
                gr.Examples(EJEMPLOS, inputs=caja, label="Prueba con…")
                limpiar = gr.Button("Nueva conversación", variant="secondary", size="sm")
            with gr.Column(scale=2):
                with gr.Accordion("¿Qué entendió el asistente?", open=True):
                    analisis = gr.HTML(PANEL_VACIO)
        entradas, salidas = [caja, chat, bot], [caja, chat, bot, analisis]
        caja.submit(responder, entradas, salidas)
        enviar.click(responder, entradas, salidas)
        limpiar.click(reiniciar, None, [chat, bot, analisis])
        demo.load(lambda: nuevo_asistente(), None, bot)
    return demo, tema


if __name__ == "__main__":
    demo, tema = construir()
    demo.launch(theme=tema, css=CSS, inbrowser=True)
