import os
import asyncio
from dotenv import load_dotenv
import streamlit as st

from typing import Optional

# SQL / DB
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit

# LangGraph + Groq (agentic)
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

# Custom DB generator
from dbgen import create_sample_database

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
def render_chat_history():
    for msg in st.session_state.messages or []:
        if not isinstance(msg, dict):
            continue
        if "role" not in msg or "content" not in msg:
            continue
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

render_chat_history()

# ---------------------------
# Utilities creating agent/toolkit
# ---------------------------
def create_sql_database_and_toolkit(connection_string: str):
    """
    Create engine + SQLDatabase wrapper + return engine and SQLDatabase object.
    """
    engine = create_engine(connection_string)
    # SQLDatabase can accept an engine or a connection URI depending on version.
    sql_db = SQLDatabase(engine)
    return engine, sql_db

def build_langgraph_agent(sql_db: SQLDatabase, groq_model: str, groq_api_key: Optional[str]):
    """
    Build the LangGraph ReAct agent graph with SQLDatabaseToolkit.
    We return the graph object (agent_graph).
    """
    # LLM initialization - adjust args to your ChatGroq version if needed
    # Use deterministic behavior for SQL: temperature=0
    try:
        llm = ChatGroq(
            model=groq_model,
            temperature=0,
            max_retries=2,
            groq_api_key=groq_api_key if groq_api_key else None
        )
    except TypeError:
        # fallback if ChatGroq expects different arg names in your installed version
        llm = ChatGroq(
            model=groq_model,
            temperature=0
        )

    toolkit = SQLDatabaseToolkit(db=sql_db, llm=llm)
    tools = toolkit.get_tools()

    system_prompt = """You are an expert SQL Data Analyst interacting with a SQL database.
Rules:
1) Begin by listing tables to understand available data.
2) Inspect schema only for relevant tables.
3) Use SELECT only (no INSERT/UPDATE/DELETE/DROP).
4) Limit results to 5 rows unless user asks for more.
5) If a SQL query errors, analyze and correct it, then retry once.
"""

    memory = MemorySaver()

    # create_react_agent from langgraph; use 'prompt' parameter (preferred) to inject system prompt
    graph = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        checkpointer=memory
    )
    return graph

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
# Async stream handler for agent_graph (same pattern as your working code)
# ---------------------------
async def stream_graph_events(user_input: str, chat_container):
    """
    Async generator that runs the LangGraph agent graph and yields streamed chunks.
    It visualizes tool start/end events in a status expander.
    """
    status = chat_container.status("Thinking & Querying...", expanded=True)
    full_response = ""

    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    inputs = {"messages": [HumanMessage(content=user_input)]}

    try:
        agent_graph = st.session_state.agent_graph
        async for event in agent_graph.astream_events(inputs, config=config, version="v2"):
            kind = event.get("event")
            name = event.get("name")

            if kind == "on_tool_start":
                tool_name = name
                tool_inputs = event["data"].get("input")
                status.write(f"🛠️ **Tool:** `{tool_name}`")
                if tool_name == "sql_db_query":
                    if isinstance(tool_inputs, dict):
                        q = tool_inputs.get("query", str(tool_inputs))
                    else:
                        q = str(tool_inputs)
                    status.code(q, language="sql")

            elif kind == "on_tool_end":
                output = event["data"].get("output")
                outstr = str(output)
                if len(outstr) > 300:
                    outstr = outstr[:300] + "... (truncated)"
                status.write(f"✅ **Result:** {outstr}")

            elif kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"].content
                if chunk:
                    full_response += chunk
                    yield chunk

        status.update(label="Reasoning Complete", state="complete", expanded=False)

    except Exception as e:
        status.update(label="Execution Error", state="error")
        st.error(f"Agent execution error: {e}")
        yield f"\n\nSystem Error: {e}"

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
