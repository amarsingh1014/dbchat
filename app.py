import streamlit as st
import os
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from sqlalchemy import create_engine
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from dotenv import load_dotenv
from dbgen import create_sample_database  # Import the function from dbgen.py

load_dotenv()

st.set_page_config(
    page_title="DBChat",
    page_icon="🤖",
)

st.title("DBChat: Chat with your Database")

# Database connection settings
db_settings = {
    "SQLite": {
        "driver": "sqlite",
        "default_port": None
    },
    "PostgreSQL": {
        "driver": "postgresql+psycopg2",
        "default_port": 5432
    },
    "MySQL": {
        "driver": "mysql+pymysql",
        "default_port": 3306
    }
}

# Store the current connection settings in session state to detect changes
if "current_connection" not in st.session_state:
    st.session_state.current_connection = {
        "db_type": None,
        "connection_string": None
    }

# Sidebar for configuration
with st.sidebar:
    st.header("Database Connection")
    db_type = st.radio("Database Type", options=list(db_settings.keys()))
    
    if db_type == "SQLite":
        db_path = st.text_input("Database Path", "sample.db")
        connection_string = f"{db_settings[db_type]['driver']}:///{db_path}"
        password = None  # Set password to None for SQLite
        
        # Check if database exists
        if not os.path.exists(db_path):
            st.warning(f"Database file '{db_path}' does not exist.")
            create_sample = st.button("Create Sample Database")
            if create_sample:
                try:
                    # Call the function from dbgen.py directly
                    created = create_sample_database(db_path)
                    if created:
                        st.success(f"Sample database '{db_path}' created successfully!")
                    else:
                        st.info(f"Sample database '{db_path}' structure ensured.")
                except Exception as e:
                    st.error(f"Error creating sample database: {str(e)}")
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
    groq_model = st.selectbox("GROQ Model", ["meta-llama/llama-4-maverick-17b-128e-instruct", "gemma2-9b-it"], index=0)
    
    # Check if connection settings have changed
    if (st.session_state.current_connection["db_type"] != db_type or 
        st.session_state.current_connection["connection_string"] != connection_string):
        # Reset connection state if connection settings changed
        if "db_connected" in st.session_state:
            del st.session_state.db_connected
            st.warning("Connection settings changed. Please connect to the new database.")
        
        # Update stored connection settings
        st.session_state.current_connection = {
            "db_type": db_type,
            "connection_string": connection_string
        }
    
    connect_button = st.button("Connect to Database")
    
    # Only show connection string if it exists and password is set for non-SQLite databases
    if connection_string and (password is not None):
        masked_connection = connection_string.replace(password, "********") if password else connection_string
        st.code(f"Connection string: {masked_connection}", language="bash")

    # Add a button to clear chat history
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Main chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Display connection status and initialize agent
if connect_button or "db_connected" in st.session_state:
    try:
        if "db_connected" not in st.session_state:
            # Check if SQLite database exists before connecting
            if db_type == "SQLite" and not os.path.exists(db_path):
                # Auto-create the database if it doesn't exist
                created = create_sample_database(db_path)
                if created:
                    st.success(f"Sample database '{db_path}' was automatically created.")
                else:
                    st.info(f"Sample database '{db_path}' structure ensured.")
                
            engine = create_engine(connection_string)
            db = SQLDatabase(engine)
            
            llm = ChatGroq(
                temperature=0,
                groq_api_key=groq_api_key,
                model_name=groq_model
            )
            
            # Create toolkit and agent
            toolkit = SQLDatabaseToolkit(db=db, llm=llm)
            
            st.session_state.agent = create_sql_agent(
                llm=llm,
                toolkit=toolkit,
                verbose=True,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            )
            
            st.session_state.db_connected = True
            st.success("Successfully connected to the database!")
        
        # Chat input and processing
        if prompt := st.chat_input("Ask a question about your database"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
                
            with st.chat_message("assistant"):
                st_callback = StreamlitCallbackHandler(st.container())
                response = st.session_state.agent.run(prompt, callbacks=[st_callback])
                st.markdown(response)
                
            st.session_state.messages.append({"role": "assistant", "content": response})
            
    except Exception as e:
        st.error(f"Error connecting to database: {str(e)}")
        if "db_connected" in st.session_state:
            del st.session_state.db_connected
else:
    st.info("Please configure your database connection in the sidebar and click 'Connect to Database'")