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

from flask import request, jsonify
from flask_cors import CORS
import json
from plotly.utils import PlotlyJSONEncoder

from sqlalchemy import create_engine, text

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

productos_string = ""
variables_string = ""

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)

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

def get_fig_from_code(code):
    local_variables = {
        "df_productos_all": df_productos_all,
        "df_variables_all": df_variables_all,
        "px": px,
        "pd": pd,
        "go": go
    }
    try:
        # exec(code, {}, local_variables)
        exec(code, local_variables, local_variables)
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
CORS(server, resources={r"/*": {"origins": ["http://localhost:4200", "http://localhost:8080", "http://127.0.0.1:8080"]}})
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


def run_memory_agent():
    app.run(debug=False, port=8007)

if __name__ == "__main__":
    run_memory_agent()