import os


CLIENT = None
def lazy_get_client():
    global CLIENT
        
    if CLIENT is None:
        from openai import AzureOpenAI
        assert "OPENAI_AZURE_ENDPOINT" in os.environ, "Please set the OPENAI_AZURE_ENDPOINT environment variable"
        assert "OPENAI_AZURE_KEY" in os.environ, "Please set the OPENAI_AZURE_KEY environment variable"
        CLIENT = AzureOpenAI(
            api_version="2023-07-01-preview",
            azure_endpoint=os.environ.get("OPENAI_AZURE_ENDPOINT"),
            api_key=os.environ.get("OPENAI_AZURE_KEY"),
        )
    return CLIENT


def openai_gpt4(prompt):  
    return openai_call(prompt, "gpt-4")

def openai_gpt4o(prompt):  
    return openai_call(prompt, "gpt-4o")

def openai_call(prompt, model):
    client = lazy_get_client()
    import openai

    try:
        response = client.chat.completions.create(
            model=model,
            messages=prompt,
            temperature=0,
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

    return response.choices[0].message.content, (response.usage.prompt_tokens, response.usage.completion_tokens)

