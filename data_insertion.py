import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv
import os
from datetime import date

load_dotenv()

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

def data_insertion():
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )

    cur = conn.cursor()

    # --- Create products table ---
    cur.execute("""
    CREATE TABLE IF NOT EXISTS productos (
        id SERIAL PRIMARY KEY,
        producto VARCHAR(100),
        marca VARCHAR(100),
        peso FLOAT,
        precio FLOAT,
        pais VARCHAR(100),
        retail VARCHAR(100),
        cantidad VARCHAR(50)
    );
    """)

    # Insert sample data into productos only if empty
    cur.execute("SELECT COUNT(*) FROM productos;")
    count = cur.fetchone()[0]

    if count == 0:
        datos = [
            ("Leche", "Alpina", 1.0, 4500, "Colombia", "Jumbo", "Litro"),
            ("Arroz", "Diana", 5.0, 18000, "Colombia", "Éxito", "Bolsa"),
            ("Café", "Juan Valdez", 0.5, 25000, "Colombia", "Carulla", "Paquete"),
            ("Aceite", "Girasol", 1.0, 12000, "Argentina", "Jumbo", "Botella"),
            ("Harina", "Doña Arepa", 1.0, 5000, "Venezuela", "Éxito", "Bolsa"),
            ("Galletas", "Oreo", 0.36, 7000, "EEUU", "Jumbo", "Paquete"),
            ("Chocolate", "Nestlé", 0.25, 8500, "Suiza", "Carulla", "Tableta"),
            ("Queso", "Colanta", 0.5, 15000, "Colombia", "Jumbo", "Bloque"),
            ("Atún", "Van Camp’s", 0.17, 6500, "Ecuador", "Éxito", "Lata"),
            ("Cereal", "Kellogg’s", 0.5, 14000, "EEUU", "Carulla", "Caja"),
        ]

        insert_query = sql.SQL("""
            INSERT INTO productos (producto, marca, peso, precio, pais, retail, cantidad)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """)
        cur.executemany(insert_query, datos)
        print("✅ Productos inserted.")

    else:
        print("ℹ️ Productos already populated, skipping insert.")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS variables (
        id SERIAL PRIMARY KEY,
        name VARCHAR(100),
        value FLOAT,
        country VARCHAR(100),
        date DATE
    );
    """)

    cur.execute("SELECT COUNT(*) FROM variables;")
    count_v = cur.fetchone()[0]

    # Insert test data into variables
    if count_v == 0:
        variables_data = [
            ("PIB Brazil", 30000, "Brazil", date(2023, 12, 31)),
            ("PIB Colombia", 15000, "Colombia", date(2023, 12, 31)),
            ("Inflation USA", 3.5, "USA", date(2024, 1, 15)),
            ("Unemployment Argentina", 7.2, "Argentina", date(2024, 1, 15)),
        ("PIB Mexico", 25000, "Mexico", date(2023, 12, 31)),
        ("Inflation Brazil", 4.2, "Brazil", date(2024, 1, 15)),
        ("Unemployment Colombia", 9.8, "Colombia", date(2024, 1, 15)),
    ]
        insert_variables = sql.SQL("""
            INSERT INTO variables (name, value, country, date)
            VALUES (%s, %s, %s, %s)
        """)

        cur.executemany(insert_variables, variables_data)
        print("✅ Variables inserted.")
    else:
        print("ℹ️ Variables already populated, skipping insert.")

    # Commit and close
    conn.commit()
    cur.close()
    conn.close()
    print("🎉 Data insertion completed successfully.")

