# BackendVisual-service

Backend ligero (Flask/Dash) que persiste y sirve figuras dinámicas de Plotly generadas por el LLM. Permite:
- Guardar mensajes/indicaciones junto con el código de visualización.
- Recuperar el JSON de la figura para renderizarlo en el Frontend (React con `react-plotly.js`).

## Contexto dentro del proyecto
Forma parte de Pricing Inteligente para habilitar dashboards dinámicos solicitados por el usuario. Se alimenta de datos ya preparados/migrados por Extraction-service y Embedding-service (vía consultas del LLM), y expone figuras para el Frontend.

## Endpoints

- POST `/api/messages`
  - Propósito: Persistir un mensaje de chat con código Plotly asociado en la tabla `messages` (columna `visualizacion`). Retorna `msg_id` (también `figure_id`).
  - Body (alguna de las siguientes opciones):
    - `{ "visualizacion_code": "<plotly code>", "text": "texto opcional" }`
    - `{ "query": "<instruccion>" }` (el código se genera con dataframes en memoria)
  - Respuesta: `{ "msg_id": <int>, "figure_id": <int>, "text": <string|null> }`

- GET `/api/figures/<figure_id>`
  - Propósito: Retornar el JSON de Plotly asociado a `messages.visualizacion`.
  - content-type: `application/json`
  - Payload: JSON estándar de Plotly `{ data: [...], layout: { ... } }`.

- POST `/generate-graph`
  - Propósito: Generar una figura directamente desde una `query` (no persiste).
  - Body: `{ "query": "..." }`

## Variables de entorno
Conexión a base de datos (ej. Postgres de Supabase):
- `DB_NAME`
- `DB_USER`
- `DB_PASSWORD`
- `DB_HOST`
- `DB_PORT`
- `DB_PRODS` (tabla sample de productos)
- `DB_VARS` (tabla sample de variables)
- `DB_MESSAGES` (opcional; default `messages`)

## Integración con el Frontend
- Tras persistir un mensaje, el servicio de chat envía al UI un JSON con identificadores:

```json
{ "msg_id": 42, "text": "Aquí tienes la evolución de ventas.", "figure_id": 42 }
```

- El UI renderiza la figura consultando `/api/figures/42` y pasando el JSON a Plotly.

## Ejecución local
Servidor por defecto en puerto 8007:
```bash
python3 main.py
```
Asegúrate de instalar dependencias de `requirements.txt` y definir tu `.env`.

## Dockerización
No se incluye un `Dockerfile` en este servicio dentro del repo. Recomendación:
- Crear una imagen basada en `python:3.11-slim`, instalar requirements y exponer `8007`.
- Usar `gunicorn` o el servidor embebido de Flask para entornos de desarrollo.

## Resolución de problemas
- Respuesta vacía al obtener figuras: verifica que `messages.visualizacion` contenga código válido de Plotly.
- Errores de conexión a DB: valida credenciales y host/puerto en variables de entorno.
