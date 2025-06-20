import os
import copy
import logging


CLIENT = None
def lazy_get_client():
    global CLIENT
    
    if CLIENT is None:
        from cohere import ClientV2
        assert "COHERE_API_KEY" in os.environ, "Please set the COHERE_API_KEY environment variable"
        CLIENT = ClientV2(api_key=os.environ.get("COHERE_API_KEY"))
    return CLIENT


def process_with_command_A(request):
    return process_with_cohere(request, "command-a-03-2025", max_tokens=8192)

def process_with_command_R7B(request):
    return process_with_cohere(request, "command-r7b-12-2024", max_tokens=4096)

def process_with_aya_expanse_32B(request):
    return process_with_cohere(request, "c4ai-aya-expanse-32b", max_tokens=8192)

def process_with_aya_expanse_8B(request):
    return process_with_cohere(request, "c4ai-aya-expanse-8b", max_tokens=4096)


def process_with_cohere(request, model, max_tokens=8192):
    # to avoid overwriting the original request
    request = copy.deepcopy(request)
    co = lazy_get_client()

    messages=[{
			"role": "user",
			"content": [{"type": "text", "text": request['prompt']}]
		}]

    response = co.chat(
        model=model,
        temperature=0,
        messages=messages,
        max_tokens=max_tokens,
    )

    if response.finish_reason != "COMPLETE":
        logging.warning(f"Finish reason: {response.finish_reason}; {response.text}")
        return None

    assert response.finish_reason == "COMPLETE", f"Finish reason: {response.finish_reason}"
    
    return response.message.content[0].text, {
        "input_tokens": response.usage.billed_units.input_tokens,
        "output_tokens": response.usage.billed_units.output_tokens,
        "thinking_tokens": 0
    }