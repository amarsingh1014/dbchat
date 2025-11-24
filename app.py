import streamlit as st
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

# LangChain / Groq imports (new API)
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_groq import ChatGroq

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate

from dbgen import create_sample_database  # custom DB generator

load_dotenv()

# -----------------------------------------
# Streamlit UI Setup
# -----------------------------------------
st.set_page_config(page_title="DBChat", page_icon="🤖")
st.title("DBChat: Chat with your Database")

# Available databases
db_settings = {
    "SQLite": {"driver": "sqlite", "default_port": None},
    "PostgreSQL": {"driver": "postgresql+psycopg2", "default_port": 5432},
    "MySQL": {"driver": "mysql+pymysql", "default_port": 3306},
}

if "current_connection" not in st.session_state:
    st.session_state.current_connection = {
        "db_type": None,
        "connection_string": None
    }

# -----------------------------------------
# Sidebar UI
# -----------------------------------------
with st.sidebar:
    st.header("Database Connection")

    db_type = st.radio("Database Type", options=list(db_settings.keys()))

    # --- SQLite ---
    if db_type == "SQLite":
        db_path = st.text_input("Database Path", "sample.db")
        connection_string = f"sqlite:///{db_path}"
        password = None

        if not os.path.exists(db_path):
            st.warning(f"Database '{db_path}' does not exist.")
            if st.button("Create Sample Database"):
                try:
                    created = create_sample_database(db_path)
                    if created:
                        st.success(f"Created sample DB '{db_path}'.")
                    else:
                        st.info("Sample DB already exists.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    # --- PostgreSQL / MySQL ---
    else:
        host = st.text_input("Host", "localhost")
        port = st.number_input("Port", value=db_settings[db_type]["default_port"])
        database = st.text_input("Database")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if db_type == "PostgreSQL":
            connection_string = f"{db_settings[db_type]['driver']}://{username}:{password}@{host}:{port}/{database}?sslmode=require"
        else:
            connection_string = f"{db_settings[db_type]['driver']}://{username}:{password}@{host}:{port}/{database}"

    groq_api_key = os.getenv("GROQ_API_KEY") or st.text_input("GROQ API Key", type="password")
    groq_model = st.selectbox("GROQ Model", [
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "gemma2-9b-it"
    ])

    # detect changes and reset connection state
    if (st.session_state.current_connection["db_type"] != db_type or
        st.session_state.current_connection["connection_string"] != connection_string):
        if "db_connected" in st.session_state:
            del st.session_state.db_connected
            st.warning("Connection changed. Reconnect.")
        st.session_state.current_connection = {
            "db_type": db_type,
            "connection_string": connection_string
        }

    # Connect button
    connect_button = st.button("Connect to Database")

    # Show masked connection string (avoid leaking password)
    if connection_string and (password is not None):
        masked = connection_string.replace(password, "********") if password else connection_string
        st.code(masked, language="bash")

    # Clear chat button
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# -----------------------------------------
# Chat History Initialization
# -----------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -----------------------------------------
# CONNECT + CREATE SQL AGENT (new API)
# -----------------------------------------
if connect_button or "db_connected" in st.session_state:
    try:
        if "db_connected" not in st.session_state:
            # Auto-create SQLite DB if missing
            if db_type == "SQLite" and not os.path.exists(db_path):
                created = create_sample_database(db_path)
                if created:
                    st.success(f"Auto-created sample DB '{db_path}'.")
                else:
                    st.info("Sample DB already exists.")

            # Connect engine + SQLDatabase wrapper
            engine = create_engine(connection_string)
            db = SQLDatabase(engine)

            # LLM: Groq
            llm = ChatGroq(
                temperature=0,
                model_name=groq_model,
                groq_api_key=groq_api_key
            )

            # Toolkit + Tools
            toolkit = SQLDatabaseToolkit(db=db, llm=llm)
            tools = toolkit.get_tools()

            # --- Agent Prompt ---
            system_prompt = """
            You are an expert SQL agent. Convert natural language questions into SQL queries, 
            run them on the database, and return clear results.
            Only use the provided tables and schema. 
            If you cannot answer using SQL, explain why.
            """
            prompt = PromptTemplate.from_template(system_prompt + "\n{input}")

            # --- Modern REAct Agent ---
            agent = create_react_agent(
                llm=llm,
                tools=tools,
                prompt=prompt
            )

            # --- Modern Agent Executor ---
            agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=True,
            )

            st.session_state.agent = agent_executor
            st.session_state.db_connected = True
            st.success("Connected to the database!")

        # -----------------------------------------
        # CHAT INPUT
        # -----------------------------------------
        if prompt := st.chat_input("Ask a question about your database"):
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.write(prompt)

            with st.chat_message("assistant"):
                st_callback = StreamlitCallbackHandler(st.container())
                response = st.session_state.agent.invoke(
                    {"input": prompt},
                    callbacks=[st_callback]
                )
                answer = response["output"]
                st.write(answer)

            st.session_state.messages.append({"role": "assistant", "content": answer})

    except Exception as e:
        st.error(f"Error: {str(e)}")
        if "db_connected" in st.session_state:
            del st.session_state.db_connected

else:
    st.info("Configure DB & click 'Connect to Database' in the sidebar.")
