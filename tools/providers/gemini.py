import os
import logging

CLIENT = None
def lazy_get_client():
    global CLIENT
    
    if CLIENT is None:
        from google import genai
        
        assert "GEMINI_API_KEY" in os.environ, "Please set the GEMINI_API_KEY environment variable"

        CLIENT = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return CLIENT

def process_with_gemini_2_5_pro(request):
    return translate_with_gemini(request, "gemini-2.5-pro", max_tokens=8192)

def process_with_gemma_3_12b(request):
    return translate_with_gemini(request, "gemma-3-12b-it", max_tokens=8192)

def process_with_gemma_3_27b(request):
    return translate_with_gemini(request, "gemma-3-27b-it", max_tokens=8192)


def translate_with_gemini(request, model, max_tokens):
    client = lazy_get_client()
    from google.genai import types

    config = types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=max_tokens,
        response_mime_type="text/plain",
        safety_settings=[
            types.SafetySetting(category=category, threshold=types.HarmBlockThreshold.BLOCK_NONE)
            for category in [
                types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            ]
        ],
    )

    try:
        response = client.models.generate_content(
            model=model,
            contents=request["prompt"],
            config=config
        )
    except Exception as e:
        logging.warning(f"Skipping: {e}")
        return None

    # finish_reason 1 means STOP
    if response.candidates[0].finish_reason != "STOP":
        logging.warning(f"Finish reason: {response.candidates[0].finish_reason}; {response.text}")
        return None

    assert response.candidates[0].finish_reason == "STOP", f"Finish reason: {response.choices[0].finish_reason}"


    thinking_tokens = 0
    # check if it thoughts_token_count is in response.usage_metadata
    if hasattr(response.usage_metadata, 'thoughts_token_count'):
        thinking_tokens = response.usage_metadata.thoughts_token_count

    return response.text, {"input_tokens": response.usage_metadata.input_token_count,
                           "output_tokens": response.usage_metadata.output_token_count + thinking_tokens,
                           "thinking_tokens": thinking_tokens}
