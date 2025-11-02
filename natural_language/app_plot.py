import json
import gradio as gr
import pandas as pd
import altair as alt
from dotenv import load_dotenv
from google import genai
from database.connection import engine

# Load environment variables
load_dotenv()

# Initialize Gemini client
client = genai.Client()

# Load schema information
def load_schema():
    schema_df = pd.read_csv('../database/schema.csv')
    schema_text = schema_df.to_string()
    return schema_text

# Generate SQL query using Gemini
def generate_sql_query(question, schema):
    prompt = f"""
You are a SQL Server (T-SQL) expert. Generate only SQL Server compatible queries without any formatting or explanation. Use TOP instead of LIMIT.

Database schema:
{schema}

Question: {question}
"""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )

    generated_text = response.text or ""
    query = generated_text.strip()
    query = query.replace('```sql', '').replace('```', '').strip()
    
    return query

# Execute SQL query and return DataFrame
def execute_query(query):
    try:
        with engine.connect() as conn:
            df = pd.read_sql_query(query, conn)
            return df if not df.empty else None
    except Exception as e:
        print(f"Database error: {str(e)}")
        return None

def determine_plot_type(df, question):
    prompt = f"""
    Given this question: "{question}"
    And a DataFrame with these columns: {', '.join(df.columns)}
    
    Determine the most appropriate plot type and return only one of these options:
    - bar: for categorical comparisons
    - line: for time series or trends
    - scatter: for relationship between two numerical variables
    - pie: for part-to-whole relationships (only if 5 or fewer categories)
    
    Return only the plot type without any explanation.
    """
    
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )
    
    generated_text = response.text or ""
    return generated_text.strip()

def create_plot(df, question):
    if df is None or df.empty:
        return None
        
    plot_type = determine_plot_type(df, question)
    
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

def generate_explanation(df, question):
    if df is None or df.empty:
        return "No data to explain."
    
    # Convert DataFrame to string representation
    data_summary = df.to_string()
    
    prompt = f"""
    Given this question: "{question}"
    And this data:
    {data_summary}
    
    Provide a brief, clear explanation of what the data shows. Focus on:
    1. Key insights or patterns
    2. Notable numbers or trends
    3. Business implications if relevant
    
    Keep the explanation under 4 sentences.
    """
    
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        generated_text = response.text or ""
        return generated_text.strip()
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
    
    # Add plot output
    plot_output = gr.Plot(label="Visualization")
    
    # Add explanation output
    explanation_output = gr.Textbox(
        label="Data Explanation",
        lines=3,
        elem_classes="output-display"
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
    combined_interface.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False
    )
