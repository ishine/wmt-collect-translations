import os
import logging

CLIENT = None
def lazy_get_client():
    global CLIENT
    
    if CLIENT is None:
        import anthropic
        assert "ANTHROPIC_API_KEY" in os.environ, "Please set the ANTHROPIC_API_KEY environment variable"
        CLIENT = anthropic.Anthropic()
    return CLIENT


def anthropic_claude_35(prompt):
    client = lazy_get_client()

    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=2000,
        temperature=0,
        messages=prompt
    )

    if response.stop_reason != 'end_turn':
        logging.warning(f"Finish reason: {response.stop_reason}; {response.content[0].text}")
        return None

    assert response.stop_reason == 'end_turn', f"Finish reason: {response.stop_reason}"

    return response.content[0].text, (response.usage.input_tokens, response.usage.output_tokens)

