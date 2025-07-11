import os
import logging
from tools.errors import FINISH_STOP, FINISH_LENGTH

CLIENT = None
def lazy_get_client():
    global CLIENT
    
    if CLIENT is None:
        import anthropic
        assert "ANTHROPIC_API_KEY" in os.environ, "Please set the ANTHROPIC_API_KEY environment variable"
        CLIENT = anthropic.Anthropic()
    return CLIENT


def process_with_claude_3_7(request, temperature=0.0):
    return process_with_anthropic(request, "claude-3-7-sonnet-20250219", max_tokens=16384, temperature=temperature)


def process_with_claude_4(request, temperature=0.0):
    return process_with_anthropic(request, "claude-sonnet-4-20250514", max_tokens=16384, temperature=temperature)

def process_with_anthropic(request, model, max_tokens, temperature=0.0):
    client = lazy_get_client()

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": request['prompt']}]
    )

    if response.stop_reason == "max_tokens":
        finish_reason = FINISH_LENGTH
    elif response.stop_reason == "end_turn":
        finish_reason = FINISH_STOP
    else:
        logging.warning(f"Finish reason: {response.stop_reason}; {response.content[0].text}")
        return None


    return response.content[0].text, {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "thinking_tokens": 0,
        "finish_reason": finish_reason
    }

