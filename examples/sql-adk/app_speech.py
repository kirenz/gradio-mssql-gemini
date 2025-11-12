import json
import os
import sys
import threading
import uuid
from pathlib import Path
from typing import Dict, Tuple

src_path = Path(__file__).resolve().parents[2] / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import altair as alt
import gradio as gr
import numpy as np
import soundfile as sf  # Add this import at the top
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types as genai_types
from quarto_mssql_gemini import load_schema_description, run_query

# Load environment variables
load_dotenv()

# --- Google ADK configuration -------------------------------------------------
# We maintain a tiny agent cache so each ADK sub-agent (SQL, plotting, narration,
# transcription) only needs to be initialised once.
ADK_MODEL = os.getenv("GOOGLE_ADK_MODEL", "gemini-2.5-flash")
ADK_USER_ID = "sql-adk-user"

AgentContext = Tuple[Runner, InMemorySessionService]
_agent_lock = threading.Lock()
_agent_cache: Dict[str, AgentContext] = {}


def _content_to_text(content: genai_types.Content) -> str:
    """Extract plain text from an ADK response payload."""
    if not content or not content.parts:
        return ""
    fragments = []
    for part in content.parts:
        text_value = getattr(part, "text", None)
        if text_value:
            fragments.append(text_value)
    return "".join(fragments)


def _get_agent(agent_key: str, instruction: str, description: str) -> AgentContext:
    """Create or reuse a named ADK agent."""
    with _agent_lock:
        if agent_key in _agent_cache:
            return _agent_cache[agent_key]

        agent = LlmAgent(
            name=agent_key,
            model=ADK_MODEL,
            instruction=instruction,
            description=description,
        )
        session_service = InMemorySessionService()
        runner = Runner(
            app_name=f"sql-adk-{agent_key.lower()}",
            agent=agent,
            session_service=session_service,
        )

        context = (runner, session_service)
        _agent_cache[agent_key] = context
        return context


def _invoke_agent(
    agent_key: str,
    instruction: str,
    parts: list[genai_types.Part],
    description: str,
) -> str:
    """Run a list of content parts through an ADK agent."""
    runner, session_service = _get_agent(agent_key, instruction, description)
    session_id = str(uuid.uuid4())

    session_service.create_session_sync(
        app_name=runner.app_name,
        user_id=ADK_USER_ID,
        session_id=session_id,
    )

    user_message = genai_types.Content(role="user", parts=parts)

    final_text = ""
    for event in runner.run(
        user_id=ADK_USER_ID,
        session_id=session_id,
        new_message=user_message,
    ):
        if event.author == runner.agent.name and event.is_final_response():
            final_text = _content_to_text(event.content)

    return final_text.strip()

# Load schema information
def load_schema():
    return load_schema_description()

# Generate SQL query using Google ADK
def generate_sql_query(question, schema):
    prompt = f"""
Database schema:
{schema}

Question: {question}

Return only the SQL Server query with no commentary or markdown.
""".strip()

    sql_text = _invoke_agent(
        agent_key="SqlQueryAgent",
        instruction=(
            "You are a SQL Server (T-SQL) specialist. Answer with executable SQL "
            "only, use TOP instead of LIMIT, and avoid markdown fences."
        ),
        parts=[genai_types.Part.from_text(text=prompt)],
        description="Writes SQL queries for speech-enabled analytics requests.",
    )

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

def determine_plot_type(df, question):
    prompt = f"""
Question: {question}
Columns: {', '.join(df.columns)}

Choose exactly one plot label: bar, line, scatter, or pie.
""".strip()

    plot_text = _invoke_agent(
        agent_key="PlotTypeAgent",
        instruction=(
            "You pick chart types for analytics stories. Answer with a single word: "
            "bar, line, scatter, or pie."
        ),
        parts=[genai_types.Part.from_text(text=prompt)],
        description="Selects the best chart for voice workflows.",
    )

    plot_text = plot_text.lower()
    for option in ("bar", "line", "scatter", "pie"):
        if option in plot_text:
            return option
    return "bar"

def create_plot(df, question):
    if df is None or df.empty:
        return None
        
    plot_type = determine_plot_type(df, question)
    
    # Basic chart configuration
    width = 600
    height = 400
    
    try:
        if len(df.columns) < 2:
            print("Error creating plot: insufficient columns for visualization")
            return None

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

def generate_explanation(df, question):
    if df is None or df.empty:
        return "No data to explain."
    
    # Convert DataFrame to string representation
    data_summary = df.to_string()
    
    prompt = f"""
Question: {question}
Data:
{data_summary}

Summarise the key trends, notable values, and implications in <= 4 sentences.
""".strip()

    try:
        return _invoke_agent(
            agent_key="InsightWriterAgent",
            instruction=(
                "You write short narratives about analysis results. Keep responses "
                "under four sentences and highlight trends, outliers, plus implications."
            ),
            parts=[genai_types.Part.from_text(text=prompt)],
            description="Narrates the data for the voice assistant.",
        )
    except Exception as e:
        return f"Error generating explanation: {str(e)}"

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
            
            plot = create_plot(df, question)
            explanation = generate_explanation(df, question)
            return text_response, df, plot, explanation
        else:
            return "No results found for this query.", None, None, "No data to explain."
            
    except Exception as e:
        return f"Error processing request: {str(e)}", None, None, "Error occurred."

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

            # Gradio may return an Altair Chart directly
            if isinstance(plot_data, alt.Chart):
                chart = plot_data
            # Or a dict containing a Vega-Lite spec
            elif isinstance(plot_data, dict):
                if "spec" in plot_data:
                    chart = alt.Chart.from_dict(plot_data["spec"])
                elif "__type__" in plot_data and "spec" in plot_data.get("value", {}):
                    chart = alt.Chart.from_dict(plot_data["value"]["spec"])
            # Or a PlotData object with a serialized spec
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

def process_speech(audio):
    if audio is None:
        print("Debug: Received None audio input")
        return "No speech detected. Please try again."
    
    try:
        print(f"Debug: Received audio type: {type(audio)}")
        
        # Handle different input formats
        if isinstance(audio, tuple):
            sample_rate, audio_data = audio
            print(f"Debug: Sample rate: {sample_rate}")
        elif isinstance(audio, np.ndarray):
            audio_data = audio
            sample_rate = 16000  # Default sample rate
        else:
            print(f"Debug: Unexpected audio format: {type(audio)}")
            return "Error: Unsupported audio format"

        # Create temporary audio file
        temp_path = "temp_audio.wav"
        sf.write(temp_path, audio_data, sample_rate)
        print(f"Debug: Saved temporary file at {temp_path}")
        
        # Transcribe audio using Gemini
        with open(temp_path, "rb") as audio_file:
            audio_bytes = audio_file.read()

        transcript_text = _invoke_agent(
            agent_key="TranscriberAgent",
            instruction=(
                "You convert spoken business questions into plain text. "
                "Return only the transcription with no preamble or trailing words."
            ),
            parts=[
                genai_types.Part.from_text(
                    text="Transcribe the following audio. Respond with the raw question only."
                ),
                genai_types.Part.from_bytes(
                    data=audio_bytes,
                    mime_type="audio/wav",
                ),
            ],
            description="Turns microphone input into a SQL-ready question.",
        )
            
        # Clean up temporary file
        os.remove(temp_path)
        print(f"Debug: Transcription successful: {transcript_text}")
        
        return transcript_text.strip()
    except Exception as e:
        print(f"Debug: Error in process_speech: {str(e)}")
        return f"Error: {str(e)}"

def process_voice_query(text):
    print(f"Debug: Starting process_voice_query with text: '{text}'")
    if not text:
        print("Debug: No transcribed text received")
        return "No question transcribed. Please record your question first.", None, None, "No data to explain."

    try:
        # Call chatbot with the transcribed text
        print(f"Debug: Calling chatbot with question: '{text}'")
        result = chatbot(text)
        
        if isinstance(result, tuple):
            # Ensure the question appears in the response
            original_response = result[0]
            modified_response = f"""Question:
{text}

SQL Query:{original_response.split('SQL Query:', 1)[1] if 'SQL Query:' in original_response else original_response}"""
            
            print(f"Debug: Modified response: {modified_response}")
            return modified_response, result[1], result[2], result[3]
        
        return result
    except Exception as e:
        print(f"Debug: Error in process_voice_query: {str(e)}")
        return f"Error processing voice query: {str(e)}", None, None, "Error occurred."

# Create the main interface
with gr.Blocks() as combined_interface:
    # Add the title and description
    gr.Markdown("# SQL Query Assistant")
    gr.Markdown("Ask questions about your data using text or voice")
    
    with gr.Tab("Text Input"):
        # Create text input components
        question_input = gr.Textbox(
            label="Ask a question",
            placeholder="e.g., What were the total sales by country in 2022?",
            lines=2,
            elem_classes="text-input"
        )
    
    with gr.Tab("Voice Input"):
        # Simplified audio input
        audio_input = gr.Audio(
            sources="microphone",
            label="Record your question"
        )
        transcribed_text = gr.Textbox(
            label="Transcribed Question",
            interactive=False
        )
    
    # Shared output components
    response_output = gr.Textbox(
        label="Query Information",
        lines=4,
        elem_classes="output-display"
    )
    
    df_output = gr.DataFrame(
        label="Results",
        interactive=False
    )
    
    plot_output = gr.Plot(label="Visualization")
    
    explanation_output = gr.Textbox(
        label="Data Explanation",
        lines=3,
        elem_classes="output-display"
    )
    
    # Add examples (only in text input tab)
    with gr.Tab("Text Input"):
        gr.Examples(
            examples=[
                ["What were the total sales by country in 2022?"],
                ["Show me the top 5 products by revenue in descending order"],
                ["What is the average discount by sales channel?"],
                ["Calculate the monthly revenue for each product category in 2023"]
            ],
            inputs=question_input
        )
    
    # Add buttons
    with gr.Row():
        submit_btn = gr.Button("Submit Text Question")
        process_audio_btn = gr.Button("Process Voice Question")
        download_data_btn = gr.Button("Download Results as Excel")
        download_plot_btn = gr.Button("Download Plot as HTML")
        refresh_btn = gr.Button("Refresh")

    def save_df_to_excel(df):
        if df is not None:
            output_path = "query_results.xlsx"
            df.to_excel(output_path, index=False)
            return output_path
        return None
    
    def clear_inputs():
        return ["", None, None, None, None]  # Added None for explanation
    
    # Set up event handlers
    submit_btn.click(
        fn=chatbot,
        inputs=question_input,
        outputs=[response_output, df_output, plot_output, explanation_output]
    )
    
    # Add speech processing handlers
    audio_input.change(
        fn=process_speech,
        inputs=audio_input,
        outputs=transcribed_text
    ).then(  # Add automatic processing after transcription
        fn=process_voice_query,
        inputs=transcribed_text,
        outputs=[response_output, df_output, plot_output, explanation_output]
    )
    
    process_audio_btn.click(
        fn=process_voice_query,
        inputs=transcribed_text,
        outputs=[response_output, df_output, plot_output, explanation_output]
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
    
    refresh_btn.click(
        fn=clear_inputs,
        inputs=[],
        outputs=[question_input, response_output, df_output, plot_output, explanation_output]
    )

# Update interface configuration
if __name__ == "__main__":
    print("Starting application...")
    server_name = os.getenv("GRADIO_SERVER_NAME", "localhost")
    port_env = os.getenv("GRADIO_SERVER_PORT")
    server_port = int(port_env) if port_env else None
    combined_interface.launch(
        share=False,
        server_name=server_name,
        server_port=server_port,
        debug=True
    )
