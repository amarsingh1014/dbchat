import streamlit as st
from langchain_core.messages import HumanMessage

def render_chat_history():
    for msg in st.session_state.messages or []:
        if not isinstance(msg, dict):
            continue
        if "role" not in msg or "content" not in msg:
            continue
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

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
