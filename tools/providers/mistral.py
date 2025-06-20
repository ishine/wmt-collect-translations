import os
import logging

CLIENT = None
def lazy_get_client():
    global CLIENT
    
    if CLIENT is None:
        from mistralai import Mistral, UserMessage

        assert "MISTRAL_API_KEY" in os.environ, "Please set the MISTRAL_API_KEY environment variable"

        CLIENT = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    return CLIENT

def process_with_mistral_medium(request):
    return process_with_mistral(request, "mistral-medium-latest", max_tokens=8192)

def process_with_magistral_medium(request):
    return process_with_mistral(request, "magistral-medium-latest", max_tokens=8192)

def process_with_mistral(request, model, max_tokens):
    client = lazy_get_client()

    messages = [{"role": "user", "content": request['prompt']}]

    try:
        response = client.chat.complete(
            model=model,
            messages=messages,
            temperature=0,        
        )
    except Exception as e:
        logging.error(f"Error: {e}")
        return None

    if response.choices[0].finish_reason != "stop":
        logging.warning(f"Finish reason: {response.choices[0].finish_reason}; {response.choices[0].message.content}")
        return None

    assert response.choices[0].finish_reason == "stop", f"Finish reason: {response.choices[0].finish_reason}"

    return response.choices[0].message.content, {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "thinking_tokens": 0
    }

