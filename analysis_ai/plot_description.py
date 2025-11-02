from dotenv import load_dotenv
from google import genai

# Load environment variables if not already loaded
load_dotenv()
client = genai.Client()

def get_plot_description(data_description, metrics):
    """Generate a description of plot data using Gemini."""
    prompt = f"""Analyze the following data and provide a brief, insightful business description:
    Data Context: {data_description}
    Metrics: {metrics}
    
    Please provide a 2-3 sentence analysis focusing on key trends and business implications."""
    
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"""You are a business analyst providing insights from data.

{prompt}""",
        )
        generated_text = response.text or ""
        return generated_text.strip()
    except Exception as e:
        return f"Error generating description: {str(e)}"
