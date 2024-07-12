import os
import logging

CLIENT = None
def lazy_get_client():
    global CLIENT
    
    if CLIENT is None:
        from together import Together
        assert "TOGETHER_API_KEY" in os.environ, "Please set the TOGETHER_API_KEY environment variable"
        CLIENT = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
    return CLIENT


def together_ai_llama3_70b(prompt):
    client = lazy_get_client()
    import together

    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-3-70b-chat-hf",
            messages=prompt,
            max_tokens=4096,
            temperature=0,
            top_p=0.99,
            top_k=0,
        )
    except together.error.APIError as e:
        print(e)
        return None

    if response.choices[0].finish_reason != "stop":
        logging.warning(f"Finish reason: {response.choices[0].finish_reason}; {response.choices[0].message.content}")
        return None

    assert response.choices[0].finish_reason == "stop", f"Finish reason: {response.choices[0].finish_reason}"

    return response.choices[0].message.content, (response.usage.prompt_tokens, response.usage.completion_tokens)

