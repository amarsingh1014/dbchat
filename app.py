import os
import asyncio
from dotenv import load_dotenv
import streamlit as st

from typing import Optional

# Custom DB generator
from dbgen import create_sample_database

# Modularized imports
from src.database import create_sql_database_and_toolkit
from src.agent import build_langgraph_agent
from src.chat import render_chat_history, stream_graph_events

load_dotenv()

# ---------------------------
# UI / Page config
# ---------------------------
st.set_page_config(page_title="DBChat — LangGraph + SQL", page_icon="🤖", layout="wide")
st.title("🤖 DBChat: LangGraph ReAct SQL Agent")
st.markdown(
    """
    Ask natural language questions about your database.
    * Choose DB type in the sidebar (SQLite/Postgres/MySQL).
    * Connect -> the app creates an agent (LangGraph + Groq) and an SQL toolkit.
    * Conversations are stored in session state; clear chat to reset.
    """
)

# ---------------------------
# Session-state defaults
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []        # list of {"role": "...", "content": "..."}
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "default_thread"
# We will store connection & agent in session_state after connect:
# session_state.db_engine, session_state.sql_db, session_state.agent_graph, session_state.db_connected

# ---------------------------
# Sidebar: DB Config & Connect
# ---------------------------
with st.sidebar:
    st.header("Database Connection")

    db_settings = {
        "SQLite": {"driver": "sqlite", "default_port": None},
        "PostgreSQL": {"driver": "postgresql+psycopg2", "default_port": 5432},
        "MySQL": {"driver": "mysql+pymysql", "default_port": 3306},
    }

    db_type = st.radio("Database Type", options=list(db_settings.keys()))

    # SQLite UI
    if db_type == "SQLite":
        db_path = st.text_input("SQLite DB Path", value="example.db")
        connection_string = f"sqlite:///{db_path}"
        # create sample DB button
        if st.button("Create Sample Database (SQLite)"):
            try:
                create_sample_database(db_path)
                st.success(f"Sample DB created at {db_path}")
            except Exception as e:
                st.error(f"Failed to create sample DB: {e}")
    else:
        host = st.text_input("Host", value="localhost")
        port = st.number_input("Port", value=db_settings[db_type]["default_port"])
        database = st.text_input("Database", value="")
        username = st.text_input("Username", value="")
        password = st.text_input("Password", type="password")
        driver = db_settings[db_type]["driver"]
        if db_type == "PostgreSQL":
            connection_string = f"{driver}://{username}:{password}@{host}:{port}/{database}?sslmode=require"
        else:
            connection_string = f"{driver}://{username}:{password}@{host}:{port}/{database}"

    # LLM / GROQ config
    groq_api_key_env = os.getenv("GROQ_API_KEY", "")
    groq_api_key = st.text_input("GROQ API Key (or set env GROQ_API_KEY)", value=groq_api_key_env, type="password")
    groq_model = st.selectbox("Groq Model", options=[
        "llama-3.3-70b-versatile",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "gemma2-9b-it"
    ])

    # Connect button
    connect_button = st.button("Connect to Database")

    # Show masked connection (do not leak password)
    if "password" in locals() and password:
        try:
            masked = connection_string.replace(password, "********")
        except Exception:
            masked = connection_string
    else:
        masked = connection_string
    st.code(masked, language="bash")

    st.markdown("---")
    # Clear chat safely
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# ---------------------------
# Helper: safe render chat history
# ---------------------------
# render_chat_history() is now imported from src.chat
render_chat_history()

# ---------------------------
# Connection flow: create engine, sql_db, and agent_graph on Connect
# ---------------------------
if connect_button:
    try:
        # Create engine and SQLDatabase wrapper
        engine, sql_db = create_sql_database_and_toolkit(connection_string)

        # Build agent graph (LangGraph)
        agent_graph = build_langgraph_agent(sql_db=sql_db, groq_model=groq_model, groq_api_key=groq_api_key)

        # persist into session_state
        st.session_state.db_engine = engine
        st.session_state.sql_db = sql_db
        st.session_state.agent_graph = agent_graph
        st.session_state.db_connected = True

        st.success("Connected and agent initialized!")

    except Exception as e:
        st.error(f"Connection / agent creation failed: {e}")
        # cleanup any partial state
        for k in ("db_engine", "sql_db", "agent_graph", "db_connected"):
            if k in st.session_state:
                del st.session_state[k]

# If already connected (from previous run), expose DB info & agent
if "db_connected" in st.session_state and st.session_state.db_connected:
    st.sidebar.success("Connected ✅")
    try:
        tables = st.session_state.sql_db.get_usable_table_names()
        st.sidebar.subheader("Available Tables")
        st.sidebar.write(tables)
    except Exception:
        st.sidebar.write("Could not list tables (driver mismatch?)")

# ---------------------------
# Main chat input area (only if agent is connected)
# ---------------------------
if "db_connected" in st.session_state and st.session_state.db_connected:
    user_prompt = st.chat_input("Ask a question about your database...")

    if user_prompt:
        # append user message
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # stream agent output into assistant message
        with st.chat_message("assistant"):
            # Streamlit supports passing an async generator into st.write_stream
            # We provide st.empty container to allow status expander inside the generator
            final_text = st.write_stream(stream_graph_events(user_prompt, st.empty()))

            # persist assistant text when available
            if final_text:
                st.session_state.messages.append({"role": "assistant", "content": final_text})

else:
    st.info("Configure DB in the sidebar and press Connect to initialize the agent.")

# ---------------------------
# Footer: small debug & export
# ---------------------------
with st.expander("Session Debug"):
    st.write("session_state keys:", list(st.session_state.keys()))
    st.write("messages length:", len(st.session_state.messages))
    st.write("thread_id:", st.session_state.thread_id)

    if st.button("Export chat as text"):
        txt = "\n\n".join(f'{m["role"]}: {m["content"]}' for m in st.session_state.messages if isinstance(m, dict))
        st.download_button("Download .txt", txt, file_name="dbchat_conversation.txt", mime="text/plain")
