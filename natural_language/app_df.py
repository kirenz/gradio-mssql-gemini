import gradio as gr
import pandas as pd
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
