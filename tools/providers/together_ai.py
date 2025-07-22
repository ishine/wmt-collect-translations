import os
import logging
from tools.errors import ERROR_MAX_TOKENS

CLIENT = None
def lazy_get_client():
    global CLIENT
    
    if CLIENT is None:
        from together import Together
        assert "TOGETHER_API_KEY" in os.environ, "Please set the TOGETHER_API_KEY environment variable"
        CLIENT = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
    return CLIENT

def process_with_deepseek_v3(request, temperature=0.0):
    return process_with_together_ai(request, "deepseek-ai/DeepSeek-V3", max_tokens=8192, temperature=temperature)

def process_qwen3_235b(request, temperature=0.0):
    return process_with_together_ai(request, "Qwen/Qwen3-235B-A22B-fp8-tput", max_tokens=8192, temperature=temperature)

def process_qwen25_7b(request, temperature=0.0):
    return process_with_together_ai(request, "Qwen/Qwen2.5-7B-Instruct-Turbo", max_tokens=8192, temperature=temperature)

def process_with_llama_4_maverick(request, temperature=0.0):
    return process_with_together_ai(request, "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", max_tokens=8192, temperature=temperature)

def process_with_llama_4_scout(request, temperature=0.0):
    return process_with_together_ai(request, "meta-llama/Llama-4-Scout-17B-16E-Instruct", max_tokens=8192, temperature=temperature)

def process_with_llama_3_1_8b(request, temperature=0.0):
    return process_with_together_ai(request, "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", max_tokens=8192, temperature=temperature)

def process_with_mistral_7b(request, temperature=0.0):
    return process_with_together_ai(request, "mistralai/Mistral-7B-Instruct-v0.3", max_tokens=8192, temperature=temperature)


def process_with_together_ai(request, model, max_tokens=8192, temperature=0.0):
    client = lazy_get_client()
    import together
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": request['prompt']}],
            max_tokens=max_tokens,
            temperature=temperature,
            chat_template_kwargs={
                "enable_thinking": False, # turns off QWEN thinking
                "temperature": temperature,
                "max_tokens": max_tokens
            },
        )
    except together.error.APIError as e:
        print(f"APIError: {e}")
        return None

    if response.choices[0].finish_reason == "length":
        return ERROR_MAX_TOKENS
    if response.choices[0].finish_reason != "stop":
        logging.warning(f"Finish reason: {response.choices[0].finish_reason}; {response.choices[0].message.content}")
        return None

    return response.choices[0].message.content, {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "thinking_tokens": 0,  # Together AI does not provide thinking tokens
    }

