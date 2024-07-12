import os
import logging

CLIENT = None
def lazy_get_client():
    global CLIENT
    
    if CLIENT is None:
        from mistralai.client import MistralClient
        assert "MISTRAL_KEY" in os.environ, "Please set the MISTRAL_KEY environment variable"
        CLIENT = MistralClient(api_key=os.environ.get("MISTRAL_KEY"))
    return CLIENT


def translate_with_mistral_large(prompt):
    client = lazy_get_client()

    from mistralai.models.chat_completion import ChatMessage

    # convert prompt to ChatMessage
    messages = []
    for message in prompt:
        messages.append(ChatMessage(role=message["role"], content=message['content']))

    try:
        response = client.chat(
            model="mistral-large-latest",
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

    return response.choices[0].message.content, (response.usage.prompt_tokens, response.usage.completion_tokens)

