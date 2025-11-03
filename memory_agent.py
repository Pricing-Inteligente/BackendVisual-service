from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from dash import Dash, html, dcc, callback, Output, Input, State
import dash_ag_grid as dag
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os
import re
from dotenv import load_dotenv, find_dotenv
from typing import Optional

from flask import request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import json
from plotly.utils import PlotlyJSONEncoder

from sqlalchemy import create_engine, text

# Supabase client (opcional, para guardar mensajes via REST)
try:
    from supabase import create_client, Client as SupabaseClient
except Exception:  # paquete no instalado o no disponible
    SupabaseClient = None  # type: ignore
    def create_client(*args, **kwargs):  # type: ignore
        raise RuntimeError("Supabase client no disponible. Instala el paquete 'supabase'.")

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_PRODS = os.getenv("DB_PRODS")
DB_VARS = os.getenv("DB_VARS")
DB_MESSAGES = os.getenv("DB_MESSAGES", "viz_messages")
DB_SCHEMA = os.getenv("DB_SCHEMA", "public")
# SSL: autodetectar por defecto segun host; se puede forzar con DB_SSLMODE
_env_sslmode = os.getenv("DB_SSLMODE")
if _env_sslmode:
    DB_SSLMODE = _env_sslmode
else:
    _host_l = (os.getenv("DB_HOST") or "").lower()
    DB_SSLMODE = "require" if "supabase.co" in _host_l else "disable"
CREATE_MESSAGES_TABLE_ON_START = os.getenv("CREATE_MESSAGES_TABLE_ON_START", "false").lower() == "true"

# Config Supabase (para almacenar mensajes en la tabla `messages` vía API REST)
SUPABASE_URL = os.getenv("SUPABASE_URL") or os.getenv("VITE_SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

# Orígenes permitidos para CORS/Socket.IO (separados por coma en ALLOWED_ORIGINS)
_default_origins = [
    "http://localhost:4200",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "http://localhost:5173",
    "http://localhost:3000",
]
ALLOWED_ORIGINS = [o.strip() for o in (os.getenv("ALLOWED_ORIGINS") or ",".join(_default_origins)).split(",") if o.strip()]

productos_string = ""
variables_string = ""

_base_url = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
DATABASE_URL = f"{_base_url}?sslmode={DB_SSLMODE}" if DB_SSLMODE else _base_url
try:
    connect_timeout = int(os.getenv("DB_CONNECT_TIMEOUT", "5"))
except Exception:
    connect_timeout = 5
engine = create_engine(DATABASE_URL, pool_pre_ping=True, connect_args={"connect_timeout": connect_timeout})

# Cliente Supabase opcional (si hay credenciales)
# Cliente supabase (evitamos anotación para compatibilidad si el tipo no está disponible)
supabase = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY and SupabaseClient is not None:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    except Exception as e:
        print(f"[supabase] No se pudo inicializar el cliente: {e}")
        supabase = None

def fetch_table(table_name: str, limit: int = None):
    # try:
    #     conn = psycopg2.connect(
    #         dbname=DB_NAME,
    #         user=DB_USER,
    #         password=DB_PASSWORD,
    #         host=DB_HOST,
    #         port=DB_PORT
    #     )
    #     if limit:
    #         query = f"SELECT * FROM {table_name} LIMIT {limit};"
    #     else:
    #         query = f"SELECT * FROM {table_name};"
    #     df = pd.read_sql(query, conn)
    #     conn.close()
    #     return df
    # except Exception as e:
    #     print(f"Error fetching data from {table_name}: {e}")
    #     return pd.DataFrame()
    try:
        with engine.connect() as conn:
            if limit:
                query = text(f"SELECT * FROM {table_name} LIMIT :limit")
                df = pd.read_sql(query, conn, params={"limit": limit})
            else:
                query = text(f"SELECT * FROM {table_name}")
                df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        print(f"Error fetching data from {table_name}: {e}")
        return pd.DataFrame()

SKIP_DB_ON_START = (os.getenv("SKIP_DB_ON_START", "false").lower() == "true")
if SKIP_DB_ON_START:
    df_productos = pd.DataFrame()
    df_variables = pd.DataFrame()
    df_productos_all = pd.DataFrame()
    df_variables_all = pd.DataFrame()
else:
    # Sample 5 rows for the model to understand the data structure
    df_productos = fetch_table(DB_PRODS, 5)
    df_variables = fetch_table(DB_VARS, 5)

    # All rows from PostgreSQL
    df_productos_all = fetch_table(DB_PRODS)
    df_variables_all = fetch_table(DB_VARS)

def smart_cast_numeric(series: pd.Series) -> pd.Series:
    """
    Limpia y convierte una serie numérica en float, adaptando formatos locales.
    Detecta y corrige:
      - Separadores de miles (., o ,)
      - Separadores decimales ambiguos
      - Casos mixtos (1.234,56 o 12,345.67)
    """
    cleaned = series.astype(str).str.replace(r'[^\d.,-]', '', regex=True)

    def normalize(val: str) -> str:
        if not val or val.lower() == 'nan':
            return None

        # Detectar última coma o punto
        last_dot = val.rfind('.')
        last_comma = val.rfind(',')
        last_sep = max(last_dot, last_comma)

        if last_sep != -1:
            digits_after = len(val) - last_sep - 1

            # Determinar cuál es el separador decimal
            if digits_after <= 2:  # Ej. 1.23 o 1,2 → separador decimal
                # Eliminar los separadores de miles
                val = re.sub(r'[.,](?=\d{3}(?:[.,]|$))', '', val)
                # Convertir el último separador a '.'
                val = val.replace(',', '.')
            else:
                # Probablemente separador de miles → eliminar todos
                val = val.replace('.', '').replace(',', '')
        return val

    normalized = cleaned.map(normalize)
    return pd.to_numeric(normalized, errors='coerce').astype(float)

for df in [df_productos_all, df_variables_all]:
    # columnas numéricas que pueden contener símbolos, comas o espacios
    # for col in ['cantidad', 'valor']:
    #     if col in df.columns:
    #         df[col] = (
    #             df[col]
    #             .astype(str)
    #             .str.replace(r'[^\d.,-]', '', regex=True)  # elimina cualquier símbolo no numérico
    #             .str.replace(',', '.', regex=False)         # convierte comas decimales a puntos
    #         )
    #         df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)  # convierte a float real

    for col in ['precio', 'cantidad', 'valor']:
        if col in df.columns:
            df[col] = smart_cast_numeric(df[col])

productos_string = df_productos.to_string(index=False)
variables_string = df_variables.to_string(index=False)

# if 'cantidad' in df_productos_all.columns:
#     print("Ejemplo de valores limpios en 'cantidad':")
#     print(df_productos_all['cantidad'].head(60))


model = ChatGroq(
    api_key = GROQ_API_KEY,
    model="openai/gpt-oss-120b",
    # model="deepseek-r1-distill-llama-70b",
    # model="groq/compound",
    # model="llama-3.3-70b-versatile",
    # model="moonshotai/kimi-k2-instruct-0905",
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You're a data visualization expert and use your favourite graphing library Plotly only. "
            "Suppose the data is provided from a PostgreSQL database. "
            "Here are the first 5 rows of the 'productos_clean_fields' table:\n{productos_string}\n\n"
            "And here are the first 5 rows of the 'variables_clean_fields' table:\n{variables_string}\n\n"
            "You also have access to the full dataframes in memory as `df_productos_all` and `df_variables_all`. "
            "Always use these full dataframes when building visualizations, not just the 5-row sample. "
            "Use only pandas, plotly.express (px), and plotly.graph_objects (go) libraries. "
            "Don't use external libraries like statsmodels or sklearn. "
            "Always assign the resulting plotly figure to a variable named `fig`. "
            "Do not create multiple figures. Only one figure should be created and stored in `fig`. "
            "Be careful to use the exact column names from the dataframe when filtering or aggregating."
            "For density maps, don't use aspect argument. "
            "Take into account that you should return just the code to create the graph with plotly, without any import and in one single code block. "
            "For objects, use attributes according to the official documentation. "
            "Take in mind i use pandas version 2.3.2 and plotly version 5.19.0. "
            "The code should be inside one single code block ```python ... ```, don't use multiple code blocks. "
            "This means that you still can write explanations outside the code block, but the code itself should be in one single block. "
            "Follow the user's indications when creating the graph."
        ),
        MessagesPlaceholder(variable_name="messages")
    ]
)

chain = prompt | model

def _sanitize_code(code: str) -> str:
    """
    Limpia restos de markdown, comillas tipográficas, zero-width chars y elimina llamadas .show().
    """
    s = code.replace("\r\n", "\n").replace("\r", "\n")
    s = s.lstrip("\ufeff").replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
    lines = s.split("\n")
    cleaned_lines = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("```") or stripped.lower() in ("<code>", "</code>"):
            continue
        if i == 0 and stripped.lower() == "python":
            continue
        cleaned_lines.append(line)
    s = "\n".join(cleaned_lines)
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    s = re.sub(r"(?m)^\s*\w+\.show\(\)\s*$", "", s)
    return s.strip()

def _parse_viz_prompt(viz_prompt: str) -> dict:
    """
    Extrae un 'especificación' simple desde un VIZ_PROMPT en texto natural.
    Busca:
      - Título entre comillas después de "titulada '...'/\"...\"" (opcional)
      - Campos de ejes "Eje X: '...'" y "Eje Y: '...'" (opcionales; por defecto label/value)
      - El bloque Datos (JSON): [...]
    Devuelve {'title', 'x_key', 'y_key', 'data'} o lanza ValueError si no encuentra datos.
    """
    s = viz_prompt or ""
    # Título
    title = None
    m = re.search(r"titulada\s+['\"]([^'\"]+)['\"]", s, re.IGNORECASE)
    if m:
        title = m.group(1).strip()
    # Ejes
    x_key, y_key = "label", "value"
    mx = re.search(r"Eje\s+X\s*:\s*['\"]([^'\"]+)['\"]", s, re.IGNORECASE)
    my = re.search(r"Eje\s*Y\s*:\s*['\"]([^'\"]+)['\"]", s, re.IGNORECASE)
    if mx: x_key = mx.group(1).strip()
    if my: y_key = my.group(1).strip()
    # Datos JSON
    dm = re.search(r"Datos\s*\(JSON\)\s*:\s*(\[.*\])\s*$", s, re.IGNORECASE | re.DOTALL)
    if not dm:
        # También intentar capturar hasta el fin del paréntesis si viene texto después
        dm = re.search(r"Datos\s*\(JSON\)\s*:\s*(\[.*?\])", s, re.IGNORECASE | re.DOTALL)
    if not dm:
        raise ValueError("No se encontró bloque de datos JSON en el VIZ_PROMPT")
    data_str = dm.group(1)
    try:
        data = json.loads(data_str)
        if not isinstance(data, list) or not data:
            raise ValueError("Datos JSON vacíos o inválidos")
    except Exception as e:
        raise ValueError(f"JSON inválido en VIZ_PROMPT: {e}")
    return {"title": title or "Gráfico", "x_key": x_key, "y_key": y_key, "data": data}

def _code_from_viz_prompt(viz_prompt: str) -> str:
    """
    Genera un código Plotly determinista a partir del VIZ_PROMPT (sin imports).
    El código define una única figura `fig` (px.bar) usando los datos del prompt.
    """
    spec = _parse_viz_prompt(viz_prompt)
    # Embebe los datos como literal Python seguro (via json.dumps y luego eval por python será válido)
    data_literal = json.dumps(spec["data"], ensure_ascii=False)
    title_literal = json.dumps(spec["title"], ensure_ascii=False)
    x_key = spec["x_key"]
    y_key = spec["y_key"]
    # Construimos un hovertemplate que muestre algunos campos si existen
    hover_cols = ["currency", "store", "brand", "country", "min", "max"]
    hover_list_literal = json.dumps(hover_cols)
    code = f"""
import pandas as _pd_tmp
_df = _pd_tmp.DataFrame({data_literal})
_x = {json.dumps(x_key)} if {json.dumps(x_key)} in _df.columns else 'label'
_y = {json.dumps(y_key)} if {json.dumps(y_key)} in _df.columns else 'value'
fig = px.bar(_df, x=_x, y=_y, title={title_literal})
# Hover enriquecido (si existen columnas)
_cols = {hover_list_literal}
_present = [c for c in _cols if c in _df.columns]
if _present:
    _df['_hover'] = _df[_present].astype(str).agg(' | '.join, axis=1)
    # Evitamos definir hovertemplate explícito para no interferir; dejamos customdata disponible
    fig.update_traces(customdata=_df['_hover'])
fig.update_layout(template='plotly_white')
"""
    # Quitar imports temporales si se cuelan en sanitize; el entorno ya tiene pd/px/go
    code = re.sub(r"(?m)^\s*import\s+pandas\s+as\s+_pd_tmp\s*$", "import pandas as _pd_tmp", code)
    return _sanitize_code(code)

def get_fig_from_code(code):
    local_variables = {
        "df_productos_all": df_productos_all,
        "df_variables_all": df_variables_all,
        "px": px,
        "pd": pd,
        "go": go
    }
    try:
        cleaned = _sanitize_code(code)
        compiled = compile(cleaned, "<string>", "exec")
        exec(compiled, local_variables, local_variables)
    except SyntaxError as e:
        bad_line = ""
        try:
            bad_line = cleaned.splitlines()[e.lineno - 1] if e.lineno else ""
        except Exception:
            pass
        return go.Figure().add_annotation(
            text=f"⚠️ SyntaxError: {e.msg} en línea {e.lineno}. {bad_line}",
            showarrow=False
        )
    except Exception as e:
        return go.Figure().add_annotation(
            text=f"⚠️ Error en el código generado: {e}",
            showarrow=False
        )

    if "fig" in local_variables and isinstance(local_variables["fig"], go.Figure):
        return local_variables["fig"]

    for v in local_variables.values():
        if isinstance(v, go.Figure):
            return v

    # if "fig" in local_variables:
    #     return local_variables["fig"]

    return go.Figure().add_annotation(
        text="⚠️ No se encontró ninguna figura en el código generado.",
        showarrow=False
    )



app = Dash()
server = app.server
# CORS configurable para desarrollo
CORS(server, resources={r"/*": {"origins": ALLOWED_ORIGINS}})
# Socket.IO para WebSocket bidireccional
socketio = SocketIO(
    server,
    cors_allowed_origins=ALLOWED_ORIGINS,
    async_mode="threading",
)
app.layout = [
    html.H1("Visualizaciones:"),
    # dag.AgGrid(
    #     rowData=df_productos_all.to_dict('records'),
    #     columnDefs=[{"field": i} for i in df_productos_all.columns],
    #     defaultColDef={"filter": True, "sortable": True, "floatingFilter": True}
    # ),
    dcc.Textarea(id='user-input', style={'width': '50%', 'height': 50, 'margin-top': 20}), #requiere peticiones exactas de usuario
    html.Br(),
    html.Button('Submit', id='submit-button'),
    dcc.Loading(
        [
            # html.Div(id='my-figure', children=''),
            html.Div(id='graph-container'),
            # dcc.Markdown(id='content', children=''),
            html.Div(id='content')
        ],
        type='cube',
    )
]

@callback(
    Output('graph-container', 'children'),
    # Output('my-figure', 'children'),
    Output('content', 'children'),
    Input('submit-button', 'n_clicks'),
    State('user-input', 'value'),
    prevent_initial_call=True
)

def create_graph(_, user_input):
    response = chain.invoke(
        {
            "messages": [HumanMessage(content=user_input)],
            "productos_string": productos_string,
            "variables_string": variables_string
        }
    )
    result_output = response.content
    print("Full response from model:\n", result_output)

    # revisar que la respuesta contiene codigo Dash entre ``` python ... ```
    code_block_match = re.search(r'```(?:[Pp]ython)?(.*?)```', result_output, re.DOTALL)
    print (code_block_match)

    if code_block_match:
        code_block = code_block_match.group(1).strip()
        # obtener el codigo sin llamada de xxx.show(), ya que visualizamos con funcion get_fig_from_code
        cleaned_code = re.sub(r'(?m)^\s*\w+\.show\(\)\s*$', '', code_block)
        fig = get_fig_from_code(cleaned_code)
        return dcc.Graph(figure=fig), result_output
    else:
        return go.Figure().add_annotation(text="⚠️ No se generó gráfico válido."), result_output

@server.route("/generate-graph", methods=["POST"])
def generate_graph():
    data = request.get_json()
    user_input = data.get("query", "")

    response = chain.invoke(
        {
            "messages": [HumanMessage(content=user_input)],
            "productos_string": productos_string,
            "variables_string": variables_string
        }
    )
    result_output = response.content

    # Extraer bloque de código plotly
    code_block_match = re.search(r'```(?:[Pp]ython)?(.*?)```', result_output, re.DOTALL)
    if not code_block_match:
        return jsonify({
            "error": "No se generó código válido",
            "raw": result_output
        })

    cleaned_code = re.sub(r'(?m)^\s*\w+\.show\(\)\s*$', '', code_block_match.group(1).strip())
    fig = get_fig_from_code(cleaned_code)

    fig_json = json.dumps(fig, cls=PlotlyJSONEncoder)
    return fig_json, 200, {"Content-Type": "application/json"}


# ==== Persistencia en Supabase (PostgreSQL) y API de figuras ====

def _extract_code_block(possible_code: str) -> str:
    """
    Dado un texto que puede contener un bloque ```python ...```,
    devuelve solo el código, limpiando llamadas *.show().
    Si no hay bloque, asume que todo el string es código y lo limpia.
    """
    # Capturar todos los bloques y elegir el más probable
    matches = re.findall(r"```(?:[Pp]ython|py)?\s*(.*?)```", possible_code, re.DOTALL)
    if matches:
        candidates = sorted(matches, key=lambda s: (("fig" in s) or ("go.Figure" in s)), reverse=True)
        code = candidates[0].strip()
    else:
        code = possible_code.strip()
    return _sanitize_code(code)


def _insert_message_visualizacion(visualizacion_code: str, text_content: Optional[str] = None) -> int:
    """
    Inserta un registro en la tabla de mensajes con el código de visualización.
    Devuelve el id generado (figure_id/message_id).
    Requiere que exista la columna 'visualizacion'. Otras columnas son opcionales.
    """
    # Si hay cliente supabase configurado, usarlo para insertar
    if supabase is not None:
        try:
            payload = {"visualizacion": _sanitize_code(visualizacion_code)}
            if text_content is not None:
                payload.update({"texto": text_content})
            # Supabase-py v2: usar returning="representation" en insert, sin .select()
            insert_res = supabase.table(DB_MESSAGES).insert(
                payload,
                returning="representation",
            ).execute()
            if insert_res.data and isinstance(insert_res.data, list) and "id" in insert_res.data[0]:
                return int(insert_res.data[0]["id"])  # type: ignore
            raise RuntimeError("Supabase no devolvió id en inserción")
        except Exception as e:
            raise RuntimeError(f"Error insertando en Supabase: {e}")

    # Fallback: usar conexión directa a Postgres (VM/local)
    full_table = f"{DB_SCHEMA}.{DB_MESSAGES}"
    with engine.begin() as conn:
        # Intento mínimo: solo la columna 'visualizacion'
        try:
            res = conn.execute(
                text(f"""
                    INSERT INTO {full_table} (visualizacion)
                    VALUES (:visualizacion)
                    RETURNING id
                """),
                {"visualizacion": _sanitize_code(visualizacion_code)},
            )
            new_id = res.scalar_one()
            return int(new_id)
        except Exception as e:
            # Intento alterno: incluir una posible columna de texto si existe en el esquema
            fallback_text_cols = ["texto", "text", "content", "mensaje"]
            for col in fallback_text_cols:
                try:
                    res = conn.execute(
                        text(f"""
                            INSERT INTO {full_table} (visualizacion, {col})
                            VALUES (:visualizacion, :contenido)
                            RETURNING id
                        """),
                        {"visualizacion": _sanitize_code(visualizacion_code), "contenido": text_content or ""},
                    )
                    new_id = res.scalar_one()
                    return int(new_id)
                except Exception:
                    continue
            # Si nada funciona, relanzar el error original para visibilidad
            raise e


def _get_visualizacion_code_by_id(message_id: int) -> Optional[str]:
    # Lectura vía Supabase si está configurado
    if supabase is not None:
        try:
            res = supabase.table(DB_MESSAGES).select("visualizacion").eq("id", message_id).single().execute()
            if res.data and "visualizacion" in res.data:
                return res.data["visualizacion"]  # type: ignore
            return None
        except Exception as e:
            # Fallback a Postgres directo si falla
            print(f"[supabase] Error leyendo visualizacion: {e}")

    full_table = f"{DB_SCHEMA}.{DB_MESSAGES}"
    with engine.connect() as conn:
        res = conn.execute(
            text(f"SELECT visualizacion FROM {full_table} WHERE id = :id"),
            {"id": message_id},
        ).scalar()
        return res


@server.route("/api/messages", methods=["POST"])
def create_message():
    """
    Crea un mensaje con una visualización asociada.
    Entrada JSON soportada:
    - { "visualizacion_code": "<codigo plotly>", "text": "opcional" }
    - { "query": "<instrucción>" }  -> genera el código con el LLM
    Respuesta: { "msg_id": <int>, "figure_id": <int>, "text": <string|null> }
    """
    payload = request.get_json(silent=True) or {}
    visualizacion_code = payload.get("visualizacion_code")
    viz_prompt = payload.get("viz_prompt") or payload.get("VIZ_PROMPT") or None
    user_text = payload.get("text")
    query = payload.get("query")

    # Si no viene el código, generarlo con la cadena (compatibilidad con /generate-graph)
    if not visualizacion_code and viz_prompt:
        try:
            visualizacion_code = _code_from_viz_prompt(viz_prompt)
        except Exception as e:
            return jsonify({"error": f"VIZ_PROMPT inválido: {e}"}), 400
    elif not visualizacion_code and query:
        response = chain.invoke(
            {
                "messages": [HumanMessage(content=query)],
                "productos_string": productos_string,
                "variables_string": variables_string,
            }
        )
        result_output = response.content
        visualizacion_code = _extract_code_block(result_output)
    elif visualizacion_code:
        visualizacion_code = _extract_code_block(visualizacion_code)
    else:
        return jsonify({"error": "Falta 'visualizacion_code' o 'query'"}), 400

    try:
        # Persistir código de visualización
        new_id = _insert_message_visualizacion(visualizacion_code, text_content=user_text)
    except Exception as e:
        return jsonify({"error": f"No se pudo insertar en {DB_MESSAGES}: {e}"}), 500

    payload = {"msg_id": new_id, "figure_id": new_id, "text": user_text}
    # Notificar a clientes WebSocket que hay un nuevo mensaje listo para renderizar
    try:
        socketio.emit("new_message", payload, broadcast=True)
    except Exception as e:
        print(f"[socketio] No se pudo emitir new_message: {e}")

    # Responder con IDs para que el frontend cargue luego la figura por GET /api/figures/<id>
    return jsonify(payload), 201


@server.route("/api/viz-prompt", methods=["POST"])
def api_viz_prompt():
    """
    Convierte un VIZ_PROMPT (texto) en una figura Plotly JSON.
    Entrada JSON: { "viz_prompt": "...", "persist": true|false, "text": "opcional" }
    Si persist=true (por defecto), guarda el código generado y devuelve { msg_id, figure_id } además del fig.
    """
    payload = request.get_json(silent=True) or {}
    viz_prompt = payload.get("viz_prompt") or payload.get("VIZ_PROMPT")
    if not viz_prompt:
        return jsonify({"error": "Falta 'viz_prompt'"}), 400
    try:
        code = _code_from_viz_prompt(viz_prompt)
        fig = get_fig_from_code(code)
        fig_json = json.dumps(fig, cls=PlotlyJSONEncoder)
    except Exception as e:
        return jsonify({"error": f"No se pudo generar figura desde VIZ_PROMPT: {e}"}), 400

    if str(payload.get("persist", "true")).lower() == "true":
        try:
            new_id = _insert_message_visualizacion(code, text_content=payload.get("text"))
        except Exception as e:
            return jsonify({"error": f"No se pudo insertar en {DB_MESSAGES}: {e}"}), 500
        return fig_json, 201, {"Content-Type": "application/json", "X-Figure-Id": str(new_id)}

    return fig_json, 200, {"Content-Type": "application/json"}


@server.route("/api/figures/<int:message_id>", methods=["GET"])
def serve_figure(message_id: int):
    """
    Devuelve el JSON Plotly de la figura asociada al message_id.
    Lee el código desde la columna 'visualizacion' y lo evalúa de forma controlada.
    """
    raw = _get_visualizacion_code_by_id(message_id)
    if not raw:
        return jsonify({"error": "Figura no encontrada"}), 404

    # Si lo almacenado es un VIZ_PROMPT, conviértelo a código primero
    s = (raw or "").strip()
    is_viz_prompt = bool(re.search(r"Datos\s*\(JSON\)\s*:\s*\[", s, re.IGNORECASE))
    code = _code_from_viz_prompt(s) if is_viz_prompt else s

    cleaned = _extract_code_block(code)
    fig = get_fig_from_code(cleaned)
    fig_json = json.dumps(fig, cls=PlotlyJSONEncoder)
    return fig_json, 200, {"Content-Type": "application/json"}


@server.route("/api/figures/<int:message_id>/code", methods=["GET"])
def serve_figure_code(message_id: int):
    """
    Devuelve el código limpio que se evalúa para generar la figura (para depuración).
    """
    raw = _get_visualizacion_code_by_id(message_id)
    if not raw:
        return jsonify({"error": "Figura no encontrada"}), 404
    s = (raw or "").strip()
    is_viz_prompt = bool(re.search(r"Datos\s*\(JSON\)\s*:\s*\[", s, re.IGNORECASE))
    code = _code_from_viz_prompt(s) if is_viz_prompt else s
    cleaned = _extract_code_block(code)
    return jsonify({"id": message_id, "code": cleaned, "length": len(cleaned)})

@server.route("/api/messages/recent", methods=["GET"])
def list_recent_messages():
    """
    Lista los últimos mensajes guardados con su id y algunos metadatos.
    Query param: ?limit=5
    """
    try:
        limit = int(request.args.get("limit", 5))
        limit = max(1, min(limit, 50))
    except Exception:
        limit = 5

    rows = []
    if supabase is not None:
        try:
            res = (
                supabase
                .table(DB_MESSAGES)
                .select("id, created_at, visualizacion")
                .order("id", desc=True)
                .limit(limit)
                .execute()
            )
            data = res.data or []
            for r in data:
                rows.append({
                    "id": r.get("id"),
                    "created_at": r.get("created_at"),
                    "code_len": len(r.get("visualizacion") or ""),
                })
            return jsonify({"items": rows, "count": len(rows)}), 200
        except Exception as e:
            print(f"[supabase] Error listando recientes: {e}")

    # Fallback Postgres directo
    full_table = f"{DB_SCHEMA}.{DB_MESSAGES}"
    try:
        with engine.connect() as conn:
            res = conn.execute(text(f"""
                SELECT id, created_at, visualizacion
                FROM {full_table}
                ORDER BY id DESC
                LIMIT :limit
            """), {"limit": limit})
            for r in res.mappings():
                rows.append({
                    "id": r.get("id"),
                    "created_at": r.get("created_at"),
                    "code_len": len(r.get("visualizacion") or ""),
                })
        return jsonify({"items": rows, "count": len(rows)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@server.route("/health/db", methods=["GET"])
def health_db():
    """Chequeo simple de conexión a la base de datos."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        health = {"ok": True, "database_url": _base_url, "sslmode": DB_SSLMODE}
        if supabase is not None:
            health.update({"supabase": True, "supabase_url": SUPABASE_URL})
        else:
            health.update({"supabase": False})
        return jsonify(health), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


def _maybe_create_messages_table():
    """
    Crea la tabla messages si no existe, sólo si CREATE_MESSAGES_TABLE_ON_START=true.
    Útil en entornos de dev. En Supabase puede requerir permisos.
    """
    if not CREATE_MESSAGES_TABLE_ON_START:
        return
    # Si estamos usando Supabase (REST), no creamos tablas desde aquí.
    if supabase is not None:
        print("[setup] Supabase activo: no se gestiona CREATE TABLE desde la API.")
        return
    full_table = f"{DB_SCHEMA}.{DB_MESSAGES}"
    try:
        with engine.begin() as conn:
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {full_table} (
                    id BIGSERIAL PRIMARY KEY,
                    visualizacion text NOT NULL,
                    texto text NULL,
                    created_at timestamptz NOT NULL DEFAULT now()
                )
            """))
    except Exception as e:
        # No interrumpir el arranque si falla
        print(f"[setup] No se pudo crear {full_table}: {e}")


def run_memory_agent():
    _maybe_create_messages_table()
    # Ejecutar con Socket.IO para habilitar WebSockets
    try:
        port = int(os.getenv("PORT", "8007"))
    except Exception:
        port = 8007
    socketio.run(server, debug=False, port=port)

if __name__ == "__main__":
    run_memory_agent()


# ============ Eventos WebSocket (Socket.IO) ============

@socketio.on("connect")
def ws_connect():
    emit("connected", {"ok": True})


@socketio.on("user_message")
def ws_user_message(data):
    """
    Permite mandar mensajes por WebSocket en lugar de REST.
    data admite:
      - {"visualizacion_code": "...", "text": "..."}
      - {"query": "..."}
    Responderá emitiendo "new_message" con { msg_id, figure_id, text } y el frontend luego hará fetch a /api/figures/<id>.
    """
    try:
        visualizacion_code = data.get("visualizacion_code") if isinstance(data, dict) else None
        viz_prompt = data.get("viz_prompt") if isinstance(data, dict) else None
        user_text = data.get("text") if isinstance(data, dict) else None
        query = data.get("query") if isinstance(data, dict) else None

        if not visualizacion_code and viz_prompt:
            try:
                visualizacion_code = _code_from_viz_prompt(viz_prompt)
            except Exception as e:
                emit("error", {"error": f"VIZ_PROMPT inválido: {e}"})
                return
        elif not visualizacion_code and query:
            response = chain.invoke(
                {
                    "messages": [HumanMessage(content=query)],
                    "productos_string": productos_string,
                    "variables_string": variables_string,
                }
            )
            result_output = response.content
            visualizacion_code = _extract_code_block(result_output)
        elif visualizacion_code:
            visualizacion_code = _extract_code_block(visualizacion_code)
        else:
            emit("error", {"error": "Falta 'visualizacion_code' o 'query'"})
            return

        new_id = _insert_message_visualizacion(visualizacion_code, text_content=user_text)
        payload = {"msg_id": new_id, "figure_id": new_id, "text": user_text}
        emit("new_message", payload, broadcast=True)
    except Exception as e:
        emit("error", {"error": str(e)})