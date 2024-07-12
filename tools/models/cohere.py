import os
import copy
import logging


CLIENT = None
def lazy_get_client():
    global CLIENT
    
    if CLIENT is None:
        import cohere 
        assert "COHERE_API_KEY" in os.environ, "Please set the COHERE_API_KEY environment variable"
        CLIENT = cohere.Client(api_key=os.environ.get("COHERE_API_KEY"))
    return CLIENT


def translate_with_command_R_plus(prompt):
    return translate_with_cohere(prompt, "command-r-plus")

def translate_with_aya(prompt):
    return translate_with_cohere(prompt, "c4ai-aya-23")


def translate_with_cohere(prompt, model):
    co = lazy_get_client()

    prompt = copy.deepcopy(prompt)
    
    for message in prompt:
        if message["role"] == "assistant":
            message["role"] = "CHATBOT"
        if message["role"] == "user":
            message["role"] = "USER"
        message["message"] = message["content"]
        del message["content"]

    last_message = prompt[-1]["message"]
    prompt = prompt[:-1]
    
    response = co.chat(
        model=model,
        temperature=0,
        k=0,
        p=0.99,
        message=last_message,
        chat_history=prompt,
        max_tokens=2048,
    )

    if response.finish_reason != "COMPLETE":
        logging.warning(f"Finish reason: {response.finish_reason}; {response.text}")
        return None

    assert response.finish_reason == "COMPLETE", f"Finish reason: {response.finish_reason}"
    
    return response.text, (response.meta.billed_units.input_tokens, response.meta.billed_units.output_tokens)