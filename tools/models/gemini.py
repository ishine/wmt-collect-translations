import os
import logging

CLIENT = None
def lazy_get_client():
    global CLIENT
    
    if CLIENT is None:
        import google.generativeai as genai
        from google.generativeai.types import HarmCategory, HarmBlockThreshold

        assert "GEMINI_API_KEY" in os.environ, "Please set the GEMINI_API_KEY environment variable"
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])

        generation_config = {
            "temperature": 0,
            "top_p": 0.95,
            "top_k": 1,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }

        CLIENT = genai.GenerativeModel(
            model_name="gemini-1.5-pro",
            generation_config=generation_config,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
    return CLIENT


def translate_with_gemini(prompt):
    client = lazy_get_client()
    from google.generativeai.types import generation_types

    history = []
    for message in prompt:
        role = message["role"]
        if role == "assistant":
            role = "model"
        history.append({
            "role": role,
            "parts": [message["content"]],
        })
    last_message = history[-1]
    history = history[:-1]

    chat_session = client.start_chat(history=history)
    try:
        response = chat_session.send_message(last_message['parts'])
    except (generation_types.StopCandidateException, generation_types.BlockedPromptException) as e:
        logging.warning(f"Skipping: {e}")
        return None

    # finish_reason 1 means STOP
    if response.candidates[0].finish_reason != 1:
        logging.warning(f"Finish reason: {response.candidates[0].finish_reason}; {response.text}")
        return None

    assert response.candidates[0].finish_reason == 1, f"Finish reason: {response.choices[0].finish_reason}"

    return response.text, (response.usage_metadata.prompt_token_count, response.usage_metadata.candidates_token_count)

