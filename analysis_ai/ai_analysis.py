from dotenv import load_dotenv
from google import genai

# Load environment variables and initialize Gemini client
load_dotenv()
client = genai.Client()

def get_plot_description(data_description, metrics):
    """
    Generate an insightful description of sales data using Gemini.
    The function takes the data context and key metrics, then returns
    a business-focused analysis.
    """
    prompt = f"""As an experienced business analyst, analyze this sales performance data:

    Context: {data_description}

    Key Performance Metrics:
    {metrics}

    Provide a 2-4 sentence analysis that:
    1. Identifies the most significant insights
    2. Suggests potential business implications or recommendations"""
    
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"""You are an experienced business analyst providing clear, data-driven insights with specific numbers and actionable recommendations.

{prompt}""",
        )
        generated_text = response.text or ""
        return generated_text.strip()
    except Exception as e:
        return f"Error generating analysis: {str(e)}"
