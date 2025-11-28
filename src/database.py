from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase

def create_sql_database_and_toolkit(connection_string: str):
    """
    Create engine + SQLDatabase wrapper + return engine and SQLDatabase object.
    """
    engine = create_engine(connection_string)
    # SQLDatabase can accept an engine or a connection URI depending on version.
    sql_db = SQLDatabase(engine)
    return engine, sql_db
