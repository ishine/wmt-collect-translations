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

def process_with_deepseek_v3(request):
    return process_with_together_ai(request, "deepseek-ai/DeepSeek-V3", max_tokens=8192)

def process_qwen3_235b(request):
    return process_with_together_ai(request, "Qwen/Qwen3-235B-A22B-fp8-tput", max_tokens=8192)

def process_with_llama_4_maverick(request):
    return process_with_together_ai(request, "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", max_tokens=8192)

def process_with_llama_4_scout(request):
    return process_with_together_ai(request, "meta-llama/Llama-4-Scout-17B-16E-Instruct", max_tokens=8192)

def process_with_mistral_7b(request):
    return process_with_together_ai(request, "mistralai/Mistral-7B-Instruct-v0.3", max_tokens=8192)


def process_with_together_ai(request, model, max_tokens=8192):
    client = lazy_get_client()
    import together
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": request['prompt']}],
            max_tokens=max_tokens,
            temperature=0.0,
            chat_template_kwargs={
                "enable_thinking": False, # turns off QWEN thinking
                "temperature": 0.0,  
                "max_tokens": max_tokens
            },
        )
    except together.error.APIError as e:
        print(e)
        return None

    if response.choices[0].finish_reason != "stop":
        logging.warning(f"Finish reason: {response.choices[0].finish_reason}; {response.choices[0].message.content}")
        return None

    assert response.choices[0].finish_reason == "stop", f"Finish reason: {response.choices[0].finish_reason}"

    return response.choices[0].message.content, {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "thinking_tokens": 0,  # Together AI does not provide thinking tokens
    }

