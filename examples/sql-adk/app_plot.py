"""Gradio dashboard backed by a Google ADK agent team.

The app mirrors the original Gemini workflow but swaps in a coordinated team of
ADK agents:

1. A root ``analytics_team_lead`` agent receives structured requests that start
   with ``Task:``.
2. Based on that header it delegates to one of three specialists using ADK’s
   agent-team (auto-routing) pattern:
      * ``sql_writer_agent`` – emits raw SQL Server statements.
      * ``plot_selector_agent`` – picks a chart type from ``bar|line|scatter|pie``.
      * ``insight_writer_agent`` – writes short business narratives.
3. The root agent streams the specialist’s reply back without modification,
   keeping coordination logic inside ADK instead of Python.

Each Gradio callback now talks to the team lead, which keeps the UI identical
while showcasing an agentic design that delegates responsibly rather than
instantiating separate clients per task.
"""

import asyncio
import os
import sys
import threading
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

src_path = Path(__file__).resolve().parents[2] / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import gradio as gr
import altair as alt
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.apps.app import App
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types as genai_types
from quarto_mssql_gemini import load_schema_description, run_query

# Load environment variables
load_dotenv()

# --- Google ADK configuration -------------------------------------------------
# A single agent team handles analytics tasks; we keep the runner cached so that
# Gradio callbacks pay only for the actual model invocation, not team setup.
ADK_MODEL = os.getenv("GOOGLE_ADK_MODEL", "gemini-2.5-flash")
TEAM_APP_NAME = "sql-adk-analytics-team"
TEAM_USER_ID = "sql-adk-user"

SQL_AGENT_NAME = "sql_writer_agent"
PLOT_AGENT_NAME = "plot_selector_agent"
INSIGHT_AGENT_NAME = "insight_writer_agent"
ROOT_AGENT_NAME = "analytics_team_lead"

TeamContext = Tuple[Runner, InMemorySessionService]
_team_lock = threading.Lock()
_team_context: Optional[TeamContext] = None


def _run(coro):
    """Run an async coroutine from synchronous code."""
    return asyncio.run(coro)


def _ensure_session(session_service: InMemorySessionService, session_id: str) -> None:
    """Create the ADK session if it does not already exist."""
    existing = _run(
        session_service.get_session(
            app_name=TEAM_APP_NAME,
            user_id=TEAM_USER_ID,
            session_id=session_id,
        )
    )
    if existing:
        return
    _run(
        session_service.create_session(
            app_name=TEAM_APP_NAME,
            user_id=TEAM_USER_ID,
            session_id=session_id,
        )
    )


def _content_to_text(content: genai_types.Content) -> str:
    """Convert an ADK content payload into plain text."""
    if not content or not content.parts:
        return ""
    chunks = []
    for part in content.parts:
        text_value = getattr(part, "text", None)
        if text_value:
            chunks.append(text_value)
    return "".join(chunks)


def _ensure_agent_team() -> TeamContext:
    """Initialise the analytics agent team exactly once."""
    global _team_context
    if _team_context:
        return _team_context

    with _team_lock:
        if _team_context:
            return _team_context

        sql_agent = Agent(
            name=SQL_AGENT_NAME,
            model=ADK_MODEL,
            instruction=(
                "You generate SQL Server (T-SQL) queries. Output ONLY the SQL "
                "statement with no explanation, markdown, or surrounding text. "
                "Prefer TOP instead of LIMIT and avoid temp tables unless "
                "explicitly requested."
            ),
            description="Produces executable SQL Server queries for analytics questions.",
        )

        plot_agent = Agent(
            name=PLOT_AGENT_NAME,
            model=ADK_MODEL,
            instruction=(
                "You select the best chart type for a dataset. Respond with one "
                "word chosen from: bar, line, scatter, pie. Return nothing else."
            ),
            description="Chooses the most appropriate visualization type.",
        )

        insight_agent = Agent(
            name=INSIGHT_AGENT_NAME,
            model=ADK_MODEL,
            instruction=(
                "You summarise tabular analytics results for business readers. "
                "Reply with at most four concise sentences highlighting trends, "
                "notable values, and implications."
            ),
            description="Writes short narratives explaining analytical findings.",
        )

        root_agent = Agent(
            name=ROOT_AGENT_NAME,
            model=ADK_MODEL,
            instruction=(
                "You coordinate analytics specialists. Every user message starts "
                "with 'Task: <TASK_NAME>' followed by the task details. Delegate "
                "as follows:\n"
                "- Task: SQL_QUERY -> sql_writer_agent\n"
                "- Task: PLOT_TYPE -> plot_selector_agent\n"
                "- Task: EXPLANATION -> insight_writer_agent\n"
                "Pass the remainder of the message to the selected agent and "
                "return their final response verbatim without extra commentary. "
                "If the task is unknown, reply with 'Unsupported task'."
            ),
            description="Routes analytics tasks to the appropriate specialist agent.",
            sub_agents=[sql_agent, plot_agent, insight_agent],
        )

        session_service = InMemorySessionService()
        app = App(name=TEAM_APP_NAME, root_agent=root_agent)
        runner = Runner(
            app=app,
            session_service=session_service,
        )

        _team_context = (runner, session_service)
        return _team_context


def _invoke_team(task: str, payload: str, session_id: str) -> str:
    """Send a structured request to the agent team and capture the response."""
    runner, session_service = _ensure_agent_team()
    _ensure_session(session_service, session_id)

    user_message = genai_types.Content(
        role="user",
        parts=[
            genai_types.Part.from_text(
                text=f"Task: {task}\n\n{payload}".strip()
            )
        ],
    )

    async def _collect():
        final = ""
        async for event in runner.run_async(
            user_id=TEAM_USER_ID,
            session_id=session_id,
            new_message=user_message,
        ):
            if event.is_final_response():
                final = _content_to_text(event.content)
        return final.strip()

    return _run(_collect())


# Load schema information
def load_schema():
    return load_schema_description()


# Generate SQL query using Google ADK
def generate_sql_query(question, schema, session_id: str, history_text: str):
    prompt = f"""
Conversation so far:
{history_text or "No prior conversation."}

Database schema:
{schema}

Question: {question}

Return only the SQL Server query in plain text with no markdown or explanation.
""".strip()

    sql_text = _invoke_team("SQL_QUERY", prompt, session_id)

    cleaned_sql = sql_text.replace("```sql", "").replace("```", "").strip()
    if not cleaned_sql:
        raise RuntimeError("Google ADK returned an empty SQL statement.")
    return cleaned_sql

# Execute SQL query and return DataFrame
def execute_query(query):
    try:
        df = run_query(query)
        return df if not df.empty else None
    except Exception as e:
        print(f"Database error: {str(e)}")
        return None

def determine_plot_type(df, question, session_id: str, history_text: str):
    prompt = f"""
Conversation so far:
{history_text or "No prior conversation."}

Question: {question}
Columns: {', '.join(df.columns)}

Choose the best visualization for these data using a single word: bar, line, scatter, or pie.
""".strip()

    plot_label = _invoke_team("PLOT_TYPE", prompt, session_id)

    plot_label = plot_label.lower()
    for option in ("bar", "line", "scatter", "pie"):
        if option in plot_label:
            return option
    return "bar"

def create_plot(df, question, session_id: str, history_text: str):
    if df is None or df.empty:
        return None

    plot_type = determine_plot_type(df, question, session_id, history_text)
    
    # Basic chart configuration
    width = 600
    height = 400
    
    try:
        if plot_type == 'bar':
            # Assume first column is category and second is value
            chart = alt.Chart(df).mark_bar().encode(
                x=alt.X(df.columns[0], type='nominal'),
                y=alt.Y(df.columns[1], type='quantitative'),
                tooltip=list(df.columns)
            ).properties(width=width, height=height)
            
        elif plot_type == 'line':
            # Assume first column is temporal and second is value
            chart = alt.Chart(df).mark_line().encode(
                x=alt.X(df.columns[0], type='temporal'),
                y=alt.Y(df.columns[1], type='quantitative'),
                tooltip=list(df.columns)
            ).properties(width=width, height=height)
            
        elif plot_type == 'scatter':
            # Assume first two columns are numerical
            chart = alt.Chart(df).mark_circle().encode(
                x=alt.X(df.columns[0], type='quantitative'),
                y=alt.Y(df.columns[1], type='quantitative'),
                tooltip=list(df.columns)
            ).properties(width=width, height=height)
            
        elif plot_type == 'pie':
            # Create pie chart using theta encoding
            chart = alt.Chart(df).mark_arc().encode(
                theta=alt.Theta(df.columns[1], type='quantitative'),
                color=alt.Color(df.columns[0], type='nominal'),
                tooltip=list(df.columns)
            ).properties(width=width, height=height)
            
        else:
            # Default to bar chart
            chart = alt.Chart(df).mark_bar().encode(
                x=alt.X(df.columns[0], type='nominal'),
                y=alt.Y(df.columns[1], type='quantitative'),
                tooltip=list(df.columns)
            ).properties(width=width, height=height)
            
        return chart
    except Exception as e:
        print(f"Error creating plot: {str(e)}")
        return None

def generate_explanation(df, question, history_text: str, session_id: str):
    if df is None or df.empty:
        return "No data to explain."

    data_summary = df.to_string()
    prompt = f"""
Conversation so far:
{history_text or "No prior conversation."}

Question: {question}
Data:
{data_summary}

Write a concise narrative (<= 4 sentences) covering key trends, notable values, and any business implication.
""".strip()

    try:
        return _invoke_team("EXPLANATION", prompt, session_id)
    except Exception as exc:
        return f"Error generating explanation: {exc}"

def _format_history(history: List[dict], max_turns: int = 6) -> str:
    """Render recent chat history for prompt conditioning."""
    if not history:
        return ""
    filtered = [msg for msg in history if msg.get("role") in {"user", "assistant"}]
    truncated = filtered[-max_turns * 2 :]
    lines = []
    for message in truncated:
        role = message.get("role", "")
        speaker = "User" if role == "user" else "Analyst"
        lines.append(f"{speaker}: {message.get('content', '')}")
    return "\n".join(lines)


def _handle_conversation(
    user_message: str,
    history: List[dict],
    session_id: Optional[str],
):
    """Process a conversational turn and return updated artefacts."""
    user_message = (user_message or "").strip()
    if not user_message:
        return (
            history,
            history,
            session_id or "",
            "",
            None,
            None,
            "Please enter a question to explore the data.",
            "",
        )

    if history is None:
        history = []

    if session_id is None or not session_id.strip():
        session_id = str(uuid.uuid4())

    history.append({"role": "user", "content": user_message})
    history_text = _format_history(history)
    schema = load_schema()

    try:
        sql_query = generate_sql_query(
            question=user_message,
            schema=schema,
            session_id=session_id,
            history_text=history_text,
        )
        df = execute_query(sql_query)

        if df is None:
            bot_message = (
                "I couldn't retrieve any rows for that request. Try refining the filters "
                "or asking for a broader slice of the data."
            )
            history.append({"role": "assistant", "content": bot_message})
            return (
                history,
                history,
                session_id,
                sql_query,
                None,
                None,
                bot_message,
                "",
            )

        plot = create_plot(df, user_message, session_id, history_text)
        explanation = generate_explanation(df, user_message, history_text, session_id)
        history.append({"role": "assistant", "content": explanation})
        return (
            history,
            history,
            session_id,
            sql_query,
            df,
            plot,
            explanation,
            "",
        )

    except Exception as exc:
        error_message = f"Something went wrong while processing that request: {exc}"
        history.append({"role": "assistant", "content": error_message})
        return (
            history,
            history,
            session_id,
            "",
            None,
            None,
            error_message,
            "",
        )

# Custom CSS for modern look
custom_css = """
.container {
    max-width: 800px !important;
    margin: auto;
    padding-top: 1.5rem;
}

.text-input {
    border-radius: 12px !important;
    border: 1px solid #e5e7eb !important;
    background-color: #f9fafb !important;
}

.output-display {
    border-radius: 12px !important;
    border: 1px solid #e5e7eb !important;
    background-color: #ffffff !important;
    padding: 1.5rem !important;
    font-family: 'SF Pro Text', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif !important;
}

.examples-table {
    gap: 0.5rem !important;
}

.example-button {
    border-radius: 8px !important;
    background-color: #f3f4f6 !important;
    border: none !important;
    padding: 0.5rem 1rem !important;
}
"""

def save_plot_to_html(plot_data):
    if plot_data is not None:
        try:
            # Create a temporary file name
            output_path = "visualization.html"
            
            chart = None

            # Gradio may return an Altair Chart instance directly
            if isinstance(plot_data, alt.Chart):
                chart = plot_data
            # Or a dict that already contains a Vega-Lite spec
            elif isinstance(plot_data, dict):
                # When the Gradio plot component returns a dict, the spec can be nested
                if "spec" in plot_data:
                    chart = alt.Chart.from_dict(plot_data["spec"])
                elif "__type__" in plot_data and "spec" in plot_data.get("value", {}):
                    chart = alt.Chart.from_dict(plot_data["value"]["spec"])
            elif hasattr(plot_data, "plot"):
                try:
                    chart = alt.Chart.from_dict(json.loads(plot_data.plot))
                except (TypeError, json.JSONDecodeError):
                    chart = None

            if chart is not None:
                html = chart.to_html()
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(html)
                return output_path
        except Exception as e:
            print(f"Error saving plot: {str(e)}")
    return None

# Create the main interface
with gr.Blocks() as combined_interface:
    # Add the title and description
    gr.Markdown("# SQL Query Assistant")
    gr.Markdown("Ask questions about your data in natural language")
    
    chat_history = gr.Chatbot(label="Conversation", height=320, type="messages")
    user_input = gr.Textbox(
        label="Message",
        placeholder="Ask about your metrics or refine a previous answer...",
        lines=2,
        elem_classes="text-input"
    )

    sql_output = gr.Textbox(
        label="Last SQL Query",
        lines=4,
        elem_classes="output-display"
    )

    df_output = gr.DataFrame(label="Results", interactive=False)
    plot_output = gr.Plot(label="Visualization")
    explanation_output = gr.Textbox(
        label="Latest Narrative",
        lines=4,
        elem_classes="output-display"
    )

    gr.Examples(
        examples=[
            ["What were the total sales by country in 2022?"],
            ["Zoom in on the top 5 products by revenue."],
            ["Can you compare online vs. retail channel performance?"],
            ["Drill into monthly revenue for 2023 only."]
        ],
        inputs=user_input
    )

    with gr.Row():
        send_btn = gr.Button("Send", variant="primary")
        reset_btn = gr.Button("Reset Conversation")
        download_data_btn = gr.Button("Download Results as Excel")
        download_plot_btn = gr.Button("Download Plot as HTML")

    history_state = gr.State([])
    session_state = gr.State("")

    def save_df_to_excel(df):
        if df is not None:
            output_path = "query_results.xlsx"
            df.to_excel(output_path, index=False)
            return output_path
        return None

    def clear_conversation():
        return [], [], "", "", None, None, "", ""

    send_btn.click(
        fn=_handle_conversation,
        inputs=[user_input, history_state, session_state],
        outputs=[
            chat_history,
            history_state,
            session_state,
            sql_output,
            df_output,
            plot_output,
            explanation_output,
            user_input,
        ],
    )

    user_input.submit(
        fn=_handle_conversation,
        inputs=[user_input, history_state, session_state],
        outputs=[
            chat_history,
            history_state,
            session_state,
            sql_output,
            df_output,
            plot_output,
            explanation_output,
            user_input,
        ],
    )

    reset_btn.click(
        fn=clear_conversation,
        inputs=[],
        outputs=[
            chat_history,
            history_state,
            session_state,
            sql_output,
            df_output,
            plot_output,
            explanation_output,
            user_input,
        ],
    )

    download_data_btn.click(
        fn=save_df_to_excel,
        inputs=df_output,
        outputs=gr.File(label="Download Excel")
    )

    download_plot_btn.click(
        fn=save_plot_to_html,
        inputs=plot_output,
        outputs=gr.File(label="Download Plot")
    )

# Update interface configuration
if __name__ == "__main__":
    combined_interface.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False
    )
