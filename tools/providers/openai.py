import os


CLIENT = None
def lazy_get_client():
    global CLIENT
        
    if CLIENT is None:
        from openai import OpenAI

        assert "OPENAI_API_KEY" in os.environ, "Please set the OPENAI_API_KEY environment variable"
        CLIENT = OpenAI()
    return CLIENT


def process_with_openai_gpt4_1(request, temperature=0.0):  
    return openai_call(request, "gpt-4.1", temperature=temperature)



def openai_call(request, model, temperature=0.0):
    client = lazy_get_client()
    import openai

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": request['prompt']}
            ],
            temperature=temperature,
        )    
    except (openai.BadRequestError, openai.APITimeoutError) as e:
        return None
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(e)
        raise e

    if response.choices[0].finish_reason != "stop":
        return None

    assert response.choices[0].finish_reason == "stop", f"Finish reason: {response.choices[0].finish_reason}"

    return response.choices[0].message.content, {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "thinking_tokens": 0
    }

