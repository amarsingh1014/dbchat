from typing import Optional
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

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
