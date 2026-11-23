"""
Base de conocimiento de "La Picada de Lang-Chi".

Es la MISMA información que la V1 tenía escrita dentro del SYSTEM_PROMPT, pero
como datos. Así el código (y no el LLM) decide qué se responde.

Dos formas de usarla:
  1. Datos estructurados (SUCURSALES, CARTA, horario, política) para validar
     reglas de negocio en código.
  2. HECHOS: la base partida en "chunks" (un hecho por trozo) para recuperarlos
     con TF-IDF + coseno, como en el RAG mínimo de la Clase 12. El campo
     "indexar" agrega sinónimos e hiperónimos (Clase 06, semántica léxica):
     una mini-ontología del dominio para que "postre" encuentre el mote con
     huesillo o "dónde queda" encuentre una dirección.
"""

NOMBRE_LOCAL = "La Picada de Lang-Chi"

SUCURSALES = {
    "Providencia": "Av. Nueva Providencia 2150",
    "Ñuñoa": "Av. Irarrázaval 3400",
}

DIAS_ABIERTO = ["martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
HORA_APERTURA = (12, 30)
HORA_CIERRE = (23, 0)
TEXTO_HORARIO = "martes a domingo de 12:30 a 23:00 hrs; lunes cerrado"

# nombre -> (precio, categoría, etiquetas de restricción)
CARTA = {
    "Pastel de choclo": (9500, "plato", []),
    "Cazuela de vacuno": (8900, "plato", []),
    "Churrasco italiano": (7800, "plato", []),
    "Empanada de pino": (2800, "plato", []),
    "Ensalada chilena": (4200, "plato", ["vegana", "sin gluten"]),
    "Mote con huesillo": (3000, "postre/bebida", []),
    "Jugo natural": (2500, "bebida", []),
}

RESERVA_MIN_PERSONAS = 1
RESERVA_MAX_PERSONAS = 12


def precio_fmt(precio: int) -> str:
    """9500 -> '$9.500' (formato chileno)."""
    return "$" + f"{precio:,}".replace(",", ".")


def texto_carta() -> str:
    return "; ".join(f"{p} {precio_fmt(v[0])}" for p, v in CARTA.items())


# ---------------------------------------------------------------------------
# Hechos (chunks) para recuperación. "indexar" = texto que se vectoriza;
# "texto" = lo que se le dice al cliente (lo único que puede aparecer).
# ---------------------------------------------------------------------------
HECHOS = [
    {"id": "plato_pastel", "texto": "Pastel de choclo: $9.500",
     "indexar": "pastel de choclo plato comida precio cuesta vale sale"},
    {"id": "plato_cazuela", "texto": "Cazuela de vacuno: $8.900",
     "indexar": "cazuela de vacuno carne plato comida precio cuesta vale sale"},
    {"id": "plato_churrasco", "texto": "Churrasco italiano: $7.800",
     "indexar": "churrasco italiano sandwich carne plato comida precio cuesta vale sale"},
    {"id": "plato_empanada", "texto": "Empanada de pino: $2.800",
     "indexar": "empanada de pino carne plato comida precio cuesta vale sale"},
    {"id": "plato_ensalada", "texto": "Ensalada chilena (vegana y sin gluten): $4.200",
     "indexar": "ensalada chilena vegana vegano vegetariana vegetariano sin gluten celiaca celiaco "
                "plato comida opcion precio cuesta vale sale"},
    {"id": "plato_mote", "texto": "Mote con huesillo (postre/bebida): $3.000",
     "indexar": "mote con huesillo postre dulce bebida tomar refresco precio cuesta vale sale"},
    {"id": "plato_jugo", "texto": "Jugo natural: $2.500",
     "indexar": "jugo natural bebida tomar refresco precio cuesta vale sale"},
    {"id": "carta", "texto": "Nuestra carta: " + texto_carta(),
     "indexar": "carta menu platos comida comer almorzar almuerzo recomiendan ofrecen preparan"},
    {"id": "sucursal_providencia", "texto": "Sucursal Providencia: " + SUCURSALES["Providencia"],
     "indexar": "sucursal local restaurante providencia nueva providencia direccion donde queda ubicado "
                "ubicacion calle llegar comuna"},
    {"id": "sucursal_nunoa", "texto": "Sucursal Ñuñoa: " + SUCURSALES["Ñuñoa"],
     "indexar": "sucursal local restaurante nunoa irarrazaval direccion donde queda ubicado ubicacion "
                "calle llegar comuna"},
    {"id": "horario", "texto": "Horario: " + TEXTO_HORARIO,
     "indexar": "horario hora abren abierto abiertos cierran cerrado atienden atencion dias "
                "martes miercoles jueves viernes sabado domingo lunes fin de semana"},
    {"id": "politica_reservas", "texto": f"Reservas: de {RESERVA_MIN_PERSONAS} a {RESERVA_MAX_PERSONAS} "
                                         "personas, dentro del horario de atención",
     "indexar": "reserva reservas maximo minimo personas grupo capacidad caben cuantas politica"},
]
HECHOS_POR_ID = {h["id"]: h for h in HECHOS}

# Mundo cerrado vs. mundo abierto (Clase 07: una respuesta debe ser verdadera Y relevante)
#  - La carta y las sucursales están COMPLETAS: si algo no aparece, se puede decir "no existe".
#  - De servicios (estacionamiento, pagos, delivery...) NO hay información: hay que abstenerse.
CATEGORIAS_MUNDO_CERRADO = ["comida", "bebida", "plato", "postre", "menú", "trago"]
