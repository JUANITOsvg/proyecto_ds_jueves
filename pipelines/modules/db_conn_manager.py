import psycopg2
import dotenv
import pandas as pd
from typing import List, Dict, Any, Optional, Union
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CONNECTION SETUP

def connect_to_db(dbname: Optional[str] = None, schema: Optional[str] = None) -> Optional[psycopg2.extensions.connection]:
    """
    Establish connection to PostgreSQL database using environment variables,
    with optional database and schema selection.
    """
    logger.info(f"Connecting to database '{dbname or 'warehouse'}' (schema={schema or 'default'})...")

    try:
        conn = psycopg2.connect(
            dbname=dbname or dotenv.get_key("/opt/airflow/.env", "POSTGRES_DB") or "warehouse",
            user=dotenv.get_key("/opt/airflow/.env", "POSTGRES_USER") or "admin", 
            password=dotenv.get_key("/opt/airflow/.env", "POSTGRES_PASSWORD") or "admin",
            port=dotenv.get_key("/opt/airflow/.env", "POSTGRES_PORT") or "5432",
            host=dotenv.get_key("/opt/airflow/.env", "DB_HOST") or "postgres",
        )

        # Optionally set schema search path
        if schema:
            with conn.cursor() as cur:
                cur.execute(f"SET search_path TO {schema}, public;")
                conn.commit()
                logger.info(f"Schema search_path set to '{schema}'")

        conn.autocommit = False
        logger.info("Connection successful.")
        return conn

    except psycopg2.Error as e:
        logger.error(f"Error connecting to the database: {e}")
        return None

def close_connection(conn: psycopg2.extensions.connection) -> None:
    """
    Safely close database connection.
    
    Args:
        conn: Database connection object
    """
    try:
        if conn:
            conn.close()
            logger.info("Database connection closed.")
    except psycopg2.Error as e:
        logger.error(f"Error closing connection: {e}")


# CRUD OPERATIONS

def create_table(conn: psycopg2.extensions.connection, table_name: str, columns: dict) -> bool:
    """
    Create a table with specified columns.
    
    Args:
        conn: Database connection object
        table_name: Name of the table to create
        columns: Dictionary mapping column names to their data types
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info(f"Creating table: {table_name}")
    
    try:
        cursor = conn.cursor()
        
        # Build column definitions
        column_defs = ', '.join(f"{col} {dtype}" for col, dtype in columns.items())
        
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                {column_defs}
            );
        """)
        
        conn.commit()
        logger.info(f"Table '{table_name}' created successfully.")
        return True
        
    except psycopg2.Error as e:
        logger.error(f"Error creating table {table_name}: {e}")
        conn.rollback()
        return False
    finally:
        cursor.close()


def insert_data(conn: psycopg2.extensions.connection, table_name: str, data: Dict[str, Any]) -> bool:
    """
    Insert a single record into a table.
    
    Args:
        conn: Database connection object
        table_name: Name of the target table
        data: Dictionary mapping column names to values
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        cursor = conn.cursor()
        
        columns = list(data.keys())
        values = list(data.values())
        placeholders = ', '.join(['%s'] * len(values))
        
        query = f"""
            INSERT INTO {table_name} ({', '.join(columns)})
            VALUES ({placeholders})
        """
        
        cursor.execute(query, values)
        conn.commit()
        logger.info(f"Data inserted successfully into {table_name}")
        return True
        
    except psycopg2.Error as e:
        logger.error(f"Error inserting data into {table_name}: {e}")
        conn.rollback()
        return False
    finally:
        cursor.close()


def insert_bulk_data(conn: psycopg2.extensions.connection, table_name: str, 
                    data_list: List[Dict[str, Any]]) -> bool:
    """
    Insert multiple records into a table efficiently.
    
    Args:
        conn: Database connection object
        table_name: Name of the target table
        data_list: List of dictionaries, each representing a record
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not data_list:
        logger.warning("No data provided for bulk insert")
        return False
        
    try:
        cursor = conn.cursor()
        
        # Use the first record to determine column structure
        columns = list(data_list[0].keys())
        placeholders = ', '.join(['%s'] * len(columns))
        
        query = f"""
            INSERT INTO {table_name} ({', '.join(columns)})
            VALUES ({placeholders})
        """
        
        # Prepare values for executemany
        values_list = [[record[col] for col in columns] for record in data_list]
        
        cursor.executemany(query, values_list)
        conn.commit()
        logger.info(f"Bulk insert completed: {len(data_list)} records into {table_name}")
        return True
        
    except psycopg2.Error as e:
        logger.error(f"Error in bulk insert to {table_name}: {e}")
        conn.rollback()
        return False
    finally:
        cursor.close()


def insert_dataframe(conn: psycopg2.extensions.connection, table_name: str, 
                    df: pd.DataFrame, if_exists: str = 'append') -> bool:
    """
    Insert pandas DataFrame into PostgreSQL table using SQLAlchemy.
    
    Args:
        conn: Database connection object
        table_name: Name of the target table
        df: Pandas DataFrame to insert
        if_exists: {'fail', 'replace', 'append'}, default 'append'
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        from sqlalchemy import create_engine
        import os
        
        # Create SQLAlchemy engine from environment variables
        db_url = f"postgresql://{dotenv.get_key('/opt/airflow/.env', 'POSTGRES_USER') or 'admin'}:" \
                f"{dotenv.get_key('/opt/airflow/.env', 'POSTGRES_PASSWORD') or 'admin'}@" \
                f"{dotenv.get_key('/opt/airflow/.env', 'DB_HOST') or 'postgres'}:" \
                f"{dotenv.get_key('/opt/airflow/.env', 'POSTGRES_PORT') or '5432'}/" \
                f"{dotenv.get_key('/opt/airflow/.env', 'POSTGRES_DB') or 'warehouse'}"
        
        engine = create_engine(db_url)
        
        df.to_sql(table_name, engine, if_exists=if_exists, index=False, method='multi')
        logger.info(f"DataFrame inserted successfully into {table_name}: {len(df)} rows")
        return True
        
    except Exception as e:
        logger.error(f"Error inserting DataFrame into {table_name}: {e}")
        return False


def select_to_dataframe(conn: psycopg2.extensions.connection, query: str) -> Optional[pd.DataFrame]:
    """
    Execute a query and return results as pandas DataFrame using SQLAlchemy.
    
    Args:
        conn: Database connection object
        query: SQL query to execute
        
    Returns:
        pd.DataFrame: Query results or None if error
    """
    try:
        from sqlalchemy import create_engine
        
        # Create SQLAlchemy engine from environment variables
        db_url = f"postgresql://{dotenv.get_key('/opt/airflow/.env', 'POSTGRES_USER') or 'admin'}:" \
                f"{dotenv.get_key('/opt/airflow/.env', 'POSTGRES_PASSWORD') or 'admin'}@" \
                f"{dotenv.get_key('/opt/airflow/.env', 'DB_HOST') or 'postgres'}:" \
                f"{dotenv.get_key('/opt/airflow/.env', 'POSTGRES_PORT') or '5432'}/" \
                f"{dotenv.get_key('/opt/airflow/.env', 'POSTGRES_DB') or 'warehouse'}"
        
        engine = create_engine(db_url)
        
        df = pd.read_sql_query(query, engine)
        logger.info(f"Query executed successfully: {len(df)} rows returned")
        return df
        
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        return None


def select_data(conn: psycopg2.extensions.connection, table_name: str, 
               columns: str = "*", where_clause: str = "", 
               order_by: str = "", limit: int = None) -> Optional[List[tuple]]:
    """
    Select data from a table with optional filtering and ordering.
    
    Args:
        conn: Database connection object
        table_name: Name of the table to query
        columns: Columns to select (default: "*")
        where_clause: WHERE clause (without "WHERE" keyword)
        order_by: ORDER BY clause (without "ORDER BY" keyword)
        limit: Maximum number of rows to return
        
    Returns:
        List[tuple]: Query results or None if error
    """
    try:
        cursor = conn.cursor()
        
        query = f"SELECT {columns} FROM {table_name}"
        
        if where_clause:
            query += f" WHERE {where_clause}"
        if order_by:
            query += f" ORDER BY {order_by}"
        if limit:
            query += f" LIMIT {limit}"
            
        cursor.execute(query)
        results = cursor.fetchall()
        
        logger.info(f"Selected {len(results)} rows from {table_name}")
        return results
        
    except psycopg2.Error as e:
        logger.error(f"Error selecting data from {table_name}: {e}")
        return None
    finally:
        cursor.close()


def select_to_dataframe(conn: psycopg2.extensions.connection, query: str) -> Optional[pd.DataFrame]:
    """
    Execute a query and return results as pandas DataFrame.
    
    Args:
        conn: Database connection object
        query: SQL query to execute
        
    Returns:
        pd.DataFrame: Query results or None if error
    """
    try:
        df = pd.read_sql_query(query, conn)
        logger.info(f"Query executed successfully: {len(df)} rows returned")
        return df
        
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        return None


def update_data(conn: psycopg2.extensions.connection, table_name: str, 
               updates: Dict[str, Any], where_clause: str) -> bool:
    """
    Update records in a table.
    
    Args:
        conn: Database connection object
        table_name: Name of the target table
        updates: Dictionary mapping column names to new values
        where_clause: WHERE clause (without "WHERE" keyword)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        cursor = conn.cursor()
        
        set_clause = ', '.join([f"{col} = %s" for col in updates.keys()])
        values = list(updates.values())
        
        query = f"""
            UPDATE {table_name} 
            SET {set_clause}
            WHERE {where_clause}
        """
        
        cursor.execute(query, values)
        rows_affected = cursor.rowcount
        conn.commit()
        
        logger.info(f"Updated {rows_affected} rows in {table_name}")
        return True
        
    except psycopg2.Error as e:
        logger.error(f"Error updating data in {table_name}: {e}")
        conn.rollback()
        return False
    finally:
        cursor.close()


def delete_data(conn: psycopg2.extensions.connection, table_name: str, 
               where_clause: str) -> bool:
    """
    Delete records from a table.
    
    Args:
        conn: Database connection object
        table_name: Name of the target table
        where_clause: WHERE clause (without "WHERE" keyword)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        cursor = conn.cursor()
        
        query = f"DELETE FROM {table_name} WHERE {where_clause}"
        cursor.execute(query)
        
        rows_affected = cursor.rowcount
        conn.commit()
        
        logger.info(f"Deleted {rows_affected} rows from {table_name}")
        return True
        
    except psycopg2.Error as e:
        logger.error(f"Error deleting data from {table_name}: {e}")
        conn.rollback()
        return False
    finally:
        cursor.close()


def drop_table(conn: psycopg2.extensions.connection, table_name: str) -> bool:
    """
    Drop a table from the database.
    
    Args:
        conn: Database connection object
        table_name: Name of the table to drop
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        cursor = conn.cursor()
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.commit()
        
        logger.info(f"Table '{table_name}' dropped successfully")
        return True
        
    except psycopg2.Error as e:
        logger.error(f"Error dropping table {table_name}: {e}")
        conn.rollback()
        return False
    finally:
        cursor.close()


# TRANSACTION MANAGEMENT

def execute_transaction(conn: psycopg2.extensions.connection, 
                       operations: List[tuple]) -> bool:
    """
    Execute multiple operations as a single transaction.
    
    Args:
        conn: Database connection object
        operations: List of tuples (query, params) for each operation
        
    Returns:
        bool: True if all operations successful, False otherwise
    """
    try:
        cursor = conn.cursor()
        
        for query, params in operations:
            cursor.execute(query, params)
            
        conn.commit()
        logger.info(f"Transaction completed successfully: {len(operations)} operations")
        return True
        
    except psycopg2.Error as e:
        logger.error(f"Transaction failed: {e}")
        conn.rollback()
        return False
    finally:
        cursor.close()


# UTILITY FUNCTIONS

def table_exists(conn: psycopg2.extensions.connection, table_name: str) -> bool:
    """
    Check if a table exists in the database.
    
    Args:
        conn: Database connection object
        table_name: Name of the table to check
        
    Returns:
        bool: True if table exists, False otherwise
    """
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables 
                WHERE table_name = %s
            )
        """, (table_name,))
        
        exists = cursor.fetchone()[0]
        return exists
        
    except psycopg2.Error as e:
        logger.error(f"Error checking if table {table_name} exists: {e}")
        return False
    finally:
        cursor.close()


def get_table_info(conn: psycopg2.extensions.connection, table_name: str) -> Optional[List[tuple]]:
    """
    Get information about table columns.
    
    Args:
        conn: Database connection object
        table_name: Name of the table
        
    Returns:
        List[tuple]: Column information or None if error
    """
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_name = %s
            ORDER BY ordinal_position
        """, (table_name,))
        
        columns = cursor.fetchall()
        return columns
        
    except psycopg2.Error as e:
        logger.error(f"Error getting table info for {table_name}: {e}")
        return None
    finally:
        cursor.close()


def get_table_count(conn: psycopg2.extensions.connection, table_name: str) -> Optional[int]:
    """
    Get the number of rows in a table.
    
    Args:
        conn: Database connection object
        table_name: Name of the table
        
    Returns:
        int: Number of rows or None if error
    """
    try:
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        return count
        
    except psycopg2.Error as e:
        logger.error(f"Error getting count for {table_name}: {e}")
        return None
    finally:
        cursor.close()


# CONTEXT MANAGER FOR SAFE DATABASE OPERATIONS

class DatabaseManager:
    """
    Context manager for safe database operations with automatic connection management.
    """

    def __init__(self, dbname: Optional[str] = None, schema: Optional[str] = None):
        """
        Initialize DatabaseManager with optional schema.

        Args:
            dbname (str, optional): Database name (default from .env or 'warehouse')
            schema (str, optional): Schema name (e.g., 'dev', 'warehouse')
        """
        self.conn = None
        self.dbname = dbname
        self.schema = schema

    def __enter__(self):
        self.conn = connect_to_db(self.dbname, self.schema)
        if self.conn is None:
            raise ConnectionError(f"Failed to connect to database '{self.dbname or 'warehouse'}'")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            if exc_type is not None:
                self.conn.rollback()
                logger.error("Transaction rolled back due to exception")
            close_connection(self.conn)
    
    # --- Delegate methods to underlying connection ---
    
    def create_table(self, table_name: str, columns: dict) -> bool:
        return create_table(self.conn, table_name, columns)
    
    def insert_data(self, table_name: str, data: Dict[str, Any]) -> bool:
        return insert_data(self.conn, table_name, data)
    
    def insert_bulk_data(self, table_name: str, data_list: List[Dict[str, Any]]) -> bool:
        return insert_bulk_data(self.conn, table_name, data_list)
    
    def insert_dataframe(self, table_name: str, df: pd.DataFrame, if_exists: str = 'append') -> bool:
        return insert_dataframe(self.conn, table_name, df, if_exists)
    
    def select_data(self, table_name: str, columns: str = "*", 
                   where_clause: str = "", order_by: str = "", limit: int = None) -> Optional[List[tuple]]:
        return select_data(self.conn, table_name, columns, where_clause, order_by, limit)
    
    def select_to_dataframe(self, query: str) -> Optional[pd.DataFrame]:
        return select_to_dataframe(self.conn, query)
    
    def update_data(self, table_name: str, updates: Dict[str, Any], where_clause: str) -> bool:
        return update_data(self.conn, table_name, updates, where_clause)
    
    def delete_data(self, table_name: str, where_clause: str) -> bool:
        return delete_data(self.conn, table_name, where_clause)
    
    def drop_table(self, table_name: str) -> bool:
        return drop_table(self.conn, table_name)
    
    def execute_transaction(self, operations: List[tuple]) -> bool:
        return execute_transaction(self.conn, operations)
    
    def table_exists(self, table_name: str) -> bool:
        return table_exists(self.conn, table_name)
    
    def get_table_info(self, table_name: str) -> Optional[List[tuple]]:
        return get_table_info(self.conn, table_name)
    
    def get_table_count(self, table_name: str) -> Optional[int]:
        return get_table_count(self.conn, table_name)
