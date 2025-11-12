"""Gradio + ADK single-agent SQL assistant.

This module wires a lightweight Google ADK ``LlmAgent`` into the original
Gemini-powered dataframe app. We keep a single agent/runner pair alive inside
process-wide globals so every user question follows the same pattern:

1. Load MSSQL schema metadata from the shared helper package.
2. Build a deterministic instruction and pass it to ADK via ``Runner.run``.
3. Capture the agent's final text response, clean any residual markdown, and
   forward the SQL through the existing query/execution helpers.

The approach follows ADK's single-agent request/response design pattern—
no sub-agent orchestration is required because SQL generation is the only
delegated task. The cached runner keeps ADK initialisation costs low while
Gradio handles UI state.
"""

import os
import sys
import threading
import uuid
from pathlib import Path
from typing import Optional

src_path = Path(__file__).resolve().parents[2] / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import gradio as gr
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types as genai_types
from quarto_mssql_gemini import load_schema_description, run_query

# Load environment variables
load_dotenv()

# --- Google ADK configuration -------------------------------------------------
# We keep a single ADK runner around and reuse it for every prompt so that the
# agent keeps its cached system instruction and we avoid re-initialising the SDK.
ADK_MODEL = os.getenv("GOOGLE_ADK_MODEL", "gemini-2.5-flash")
SQL_AGENT_NAME = "SqlQueryAgent"
SQL_APP_NAME = "sql-adk-sql-agent"
SQL_USER_ID = "sql-adk-user"

_sql_runner_lock = threading.Lock()
_sql_runner: Optional[Runner] = None
_sql_session_service: Optional[InMemorySessionService] = None


def _content_to_text(content: genai_types.Content) -> str:
    """Extract plain text from a response payload."""
    if not content or not content.parts:
        return ""
    fragments = []
    for part in content.parts:
        text_value = getattr(part, "text", None)
        if text_value:
            fragments.append(text_value)
    return "".join(fragments)


def _ensure_sql_runner() -> tuple[Runner, InMemorySessionService]:
    """Initialise the ADK runner exactly once in a threadsafe manner."""
    global _sql_runner, _sql_session_service
    if _sql_runner and _sql_session_service:
        return _sql_runner, _sql_session_service

    with _sql_runner_lock:
        if _sql_runner and _sql_session_service:
            return _sql_runner, _sql_session_service

        sql_agent = LlmAgent(
            name=SQL_AGENT_NAME,
            model=ADK_MODEL,
            instruction=(
                "You are a SQL Server (T-SQL) expert. Only return executable SQL "
                "without commentary, markdown, code fences, or analysis. Prefer "
                "TOP instead of LIMIT and avoid temporary tables unless asked."
            ),
            description="Produces clean SQL Server queries for business questions.",
        )
        session_service = InMemorySessionService()
        runner = Runner(
            app_name=SQL_APP_NAME,
            agent=sql_agent,
            session_service=session_service,
        )

        _sql_runner = runner
        _sql_session_service = session_service
        return runner, session_service


def _invoke_sql_agent(prompt: str) -> str:
    """Send a prompt to the ADK agent and capture the final text response."""
    runner, session_service = _ensure_sql_runner()
    session_id = str(uuid.uuid4())

    # The in-memory session service requires us to create a session before use.
    session_service.create_session_sync(
        app_name=runner.app_name,
        user_id=SQL_USER_ID,
        session_id=session_id,
    )

    user_message = genai_types.Content(
        role="user",
        parts=[genai_types.Part.from_text(text=prompt)],
    )

    final_response = ""
    for event in runner.run(
        user_id=SQL_USER_ID,
        session_id=session_id,
        new_message=user_message,
    ):
        if event.author == SQL_AGENT_NAME and event.is_final_response():
            final_response = _content_to_text(event.content)

    return final_response.strip()


# Load schema information
def load_schema():
    return load_schema_description()


# Generate SQL query using Google ADK
def generate_sql_query(question, schema):
    prompt = f"""
Database schema:
{schema}

Question: {question}

Return only the SQL Server query using SELECT, WITH, or EXEC syntax with no markdown.
""".strip()

    raw_sql = _invoke_sql_agent(prompt)
    if not raw_sql:
        raise RuntimeError("Google ADK returned an empty SQL response.")

    cleaned_sql = raw_sql.replace("```sql", "").replace("```", "").strip()
    return cleaned_sql

# Execute SQL query and return DataFrame
def execute_query(query):
    try:
        df = run_query(query)
        return df if not df.empty else None
    except Exception as e:
        print(f"Database error: {str(e)}")
        return None

# Main chatbot function
def chatbot(question):
    try:
        schema = load_schema()
        sql_query = generate_sql_query(question, schema)
        df = execute_query(sql_query)
        
        if df is not None:
            text_response = f"""Question:
{question}

SQL Query:
{sql_query}"""
            
            return text_response, df  # Return only two values
        else:
            return "No results found for this query.", None
            
    except Exception as e:
        return f"Error processing request: {str(e)}", None

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

# Create the main interface
with gr.Blocks() as combined_interface:
    # Add the title and description
    gr.Markdown("# SQL Query Assistant")
    gr.Markdown("Ask questions about your data in natural language")
    
    # Create input and output components
    question_input = gr.Textbox(
        label="Ask a question",
        placeholder="e.g., What were the total sales by country in 2022?",
        lines=2,
        elem_classes="text-input"
    )
    
    response_output = gr.Textbox(
        label="Query Information",
        lines=4,
        elem_classes="output-display"
    )
    
    # Show DataFrame output that can also be used for download
    df_output = gr.DataFrame(
        label="Results",
        interactive=False
    )
    
    # Add examples
    gr.Examples(
        examples=[
            ["What were the total sales by country in 2022?"],
            ["Show me the top 5 products by revenue in descending order"],
            ["What is the average discount by sales channel?"],
            ["Calculate the monthly revenue for each product category in 2023"]
        ],
        inputs=question_input
    )
    
    # Add submit, download, and refresh buttons
    with gr.Row():
        submit_btn = gr.Button("Submit")
        download_btn = gr.Button("Download Results as Excel")
        refresh_btn = gr.Button("Refresh")
    
    def save_df_to_excel(df):
        if df is not None:
            output_path = "query_results.xlsx"
            df.to_excel(output_path, index=False)
            return output_path
        return None
    
    def clear_inputs():
        return "", None, None  # Return three values for three components
    
    # Set up event handlers
    submit_btn.click(
        fn=chatbot,
        inputs=question_input,
        outputs=[response_output, df_output]
    )
    
    download_btn.click(
        fn=save_df_to_excel,
        inputs=df_output,
        outputs=gr.File(label="Download Excel")
    )
    
    refresh_btn.click(
        fn=clear_inputs,
        inputs=[],
        outputs=[question_input, response_output, df_output]  # Match the three components
    )

# Update interface configuration
if __name__ == "__main__":
    combined_interface.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False
    )
