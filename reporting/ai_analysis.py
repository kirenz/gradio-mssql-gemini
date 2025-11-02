import os
from dotenv import load_dotenv

# Load environment variables and initialize Gemini client
load_dotenv()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

_generate = None
_error_msg = None

try:
    from google import genai as google_genai  # type: ignore
except ImportError:
    google_genai = None

if google_genai:
    try:
        client_kwargs = {}
        if GEMINI_API_KEY:
            client_kwargs["api_key"] = GEMINI_API_KEY
        client = google_genai.Client(**client_kwargs)

        def _generate(prompt: str) -> str:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            return getattr(response, "text", "") or ""
    except Exception as exc:
        _error_msg = f"Could not initialize Gemini client: {exc}"

if _generate is None:
    try:
        import google.generativeai as google_generativeai  # type: ignore
    except ImportError:
        google_generativeai = None

    if google_generativeai:
        try:
            if GEMINI_API_KEY:
                google_generativeai.configure(api_key=GEMINI_API_KEY)
            model = google_generativeai.GenerativeModel(GEMINI_MODEL)

            def _generate(prompt: str) -> str:
                response = model.generate_content(prompt)
                return getattr(response, "text", "") or ""

            _error_msg = None
        except Exception as exc:
            _error_msg = f"Could not initialize Gemini client: {exc}"
    elif _error_msg is None:
        _error_msg = (
            "google-genai package is not installed. Install it to enable automated analysis."
        )

def get_plot_description(data_description, metrics, context=""):
    """
    Generate an insightful description of sales data using Gemini.
    
    Args:
        data_description: Description of what the data represents
        metrics: Key performance metrics
        context: Additional context like filters applied
    """
    prompt = f"""As an experienced business analyst, analyze this sales performance data:

    Context: {data_description}
    
    Applied Filters:
    {context}

    Key Performance Metrics:
    {metrics}

    Provide a 2-4 sentence analysis that:
    1. Identifies the most significant insights considering the applied filters
    2. Suggests potential business implications or recommendations for this specific segment"""
    
    try:
        full_prompt = (
            "You are an experienced business analyst providing clear, data-driven insights "
            "with specific numbers and actionable recommendations.\n\n"
            f"{prompt}"
        )
        if _generate is None:
            raise RuntimeError(_error_msg or "Gemini client is not configured.")

        generated_text = _generate(full_prompt)
        return generated_text.strip()
    except Exception as e:
        return f"Error generating analysis: {str(e)}"
