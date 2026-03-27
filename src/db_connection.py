# src/db_connection.py

import os
import mysql.connector
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_db_credentials():
    """
    Returns database credentials from environment variables or
    Streamlit Secrets (for Streamlit Cloud deployment).
    """
    # Try Streamlit secrets first (for deployed app)
    try:
        import streamlit as st
        return {
            "host":     st.secrets.get("MYSQL_HOST",     os.getenv("MYSQL_HOST", "localhost")),
            "port":     int(st.secrets.get("MYSQL_PORT", os.getenv("MYSQL_PORT", 3306))),
            "user":     st.secrets.get("MYSQL_USER",     os.getenv("MYSQL_USER", "root")),
            "password": st.secrets.get("MYSQL_PASSWORD", os.getenv("MYSQL_PASSWORD", "")),
            "database": st.secrets.get("MYSQL_DATABASE", os.getenv("MYSQL_DATABASE", "f1_analysis")),
        }
    except Exception:
        # Fall back to environment variables (.env file)
        return {
            "host":     os.getenv("MYSQL_HOST", "localhost"),
            "port":     int(os.getenv("MYSQL_PORT", 3306)),
            "user":     os.getenv("MYSQL_USER", "root"),
            "password": os.getenv("MYSQL_PASSWORD", ""),
            "database": os.getenv("MYSQL_DATABASE", "f1_analysis"),
        }

def get_mysql_connection():
    """
    Returns a raw mysql-connector connection.
    Use this for simple INSERT and SELECT operations.
    """
    connection = mysql.connector.connect(
        host=os.getenv("MYSQL_HOST", "localhost"),
        port=int(os.getenv("MYSQL_PORT", 3306)),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DATABASE", "f1_analysis")
    )
    return connection


def get_sqlalchemy_engine():
    """
    Returns a SQLAlchemy engine.
    Use this for pandas read_sql and to_sql operations.
    """
    user = os.getenv("MYSQL_USER", "root")
    password = os.getenv("MYSQL_PASSWORD")
    host = os.getenv("MYSQL_HOST", "localhost")
    port = os.getenv("MYSQL_PORT", 3306)
    database = os.getenv("MYSQL_DATABASE", "f1_analysis")

    connection_string = (
        f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
    )

    engine = create_engine(connection_string, echo=False)
    return engine


def test_connection():
    """
    Quick test to verify database connectivity.
    Run this after setup to confirm everything works.
    """
    try:
        conn = get_mysql_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT VERSION()")
        version = cursor.fetchone()
        print(f"Connected successfully. MySQL version: {version[0]}")
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Connection failed: {e}")


if __name__ == "__main__":
    test_connection()