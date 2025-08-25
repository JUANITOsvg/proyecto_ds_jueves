# Proyecto de Pipelines con Airflow y dbt

**!!IMPORTANTE: EL .env LO SUBI PARA QUE SEA MAS CONVENIENTE PARA TODOS, PERO NUNCA SUBAN UN .env!!**

## 🛠 Arrancar el Proyecto

1. **Cloná el repo:**

   ```bash
   git clone git@github.com:JUANITOsvg/proyecto_ds_jueves.git
   cd nose que, pero pone ls en bash
   ```

2. **Levantá todo con Docker Compose:**

   ```bash
   docker-compose up -d # para modo detached, pero se puede en la consola misma para ver logs
   ```

   Esto va a levantar:

   * **Airflow**: Para orquestar las pipelines.
   * **PostgreSQL**: Base de datos donde se guardan los datos.
   * **dbt**: Para transformar los datos.
   * **Superset**: Para visualizar los datos (si lo tenés activado).

3. **Accedé a las interfaces:**

   * Airflow: [http://localhost:8080](http://localhost:8080)
   * Superset: [http://localhost:8088](http://localhost:8088)

   Credenciales por defecto (en el .env que subí, lo subi pq no tiene nada muy loco):

   * Usuario: `admin`
   * Contraseña: `admin`

---

## 🧱 Crear una Pipeline

1. **Copiá una plantilla de DAG:**

   ```bash
   cp -r ejemplos/dag_template/ pipelines/dags/mi_pipeline/
   ```

2. **Editá el archivo `mi_pipeline.py`:**

   * Cambiá el nombre de la clase y el archivo.
   * Modificá las tareas según lo que necesites.
   * Si tu pipeline requiere dependencias adicionales, agregalas en un `requirements.txt` dentro de la carpeta de tu DAG.

3. **Airflow va a instalar las dependencias automáticamente** cuando levantes el contenedor.

---

## 📊 Consultar la Tabla Resultante

1. **Entrá a Apache Superset**: [http://localhost:8088](http://localhost:8088)

2. **Conectá la base de datos:**

   * Andá a **Data** > **Databases** > **+ Database**.
   * Usá la siguiente configuración:

     * **SQLAlchemy URI**: `postgresql+psycopg2://airflow:airflow@postgres:5432/warehouse`
     * **Nombre**: `Warehouse DB`

3. **Explorá los datos:**

   * Una vez conectada la base de datos, andá a **Data** > **Datasets**.
   * Buscá la tabla que generó tu pipeline (por ejemplo, `test_table_1`).
   * Hacela clic para ver los datos.

---

## 📁 Estructura del Proyecto

```
.
├── pipelines/
│   ├── dags/
│   │   └── mi_pipeline/
│   │       ├── mi_pipeline.py
│   │       └── requirements.txt
│   └── Dockerfile
├── docker-compose.yml
└── README.md
```

## Comandos utiles

   * Ver data en el whs: docker exec -it warehouse_postgres psql -U ${POSTGRES_USER:-admin} -d ${POSTGRES_DB:-warehouse}

