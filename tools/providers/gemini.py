import os
import logging

CLIENT = None
def lazy_get_client():
    global CLIENT
    
    if CLIENT is None:
        from google import genai
        
        assert "GEMINI_API_KEY" in os.environ, "Please set the GEMINI_API_KEY environment variable"

        CLIENT = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return CLIENT

def process_with_gemini_2_5_pro(request, temperature=0.0):
    return translate_with_gemini(request, "gemini-2.5-pro", max_tokens=65536, temperature=temperature)

def process_with_gemma_3_12b(request, temperature=0.0):
    return translate_with_gemini(request, "gemma-3-12b-it", max_tokens=32768, temperature=temperature)

def process_with_gemma_3_27b(request, temperature=0.0):
    return translate_with_gemini(request, "gemma-3-27b-it", max_tokens=32768, temperature=temperature)


# for WMT25 to speed up, we used a separate script which didn't have backup mechanisms. Thus after collecting the translations, we fill the gaps with original script
import json
gemini_cache = {}
with open("gemini_cache.jsonl", "r") as f:
    for line in f:
        entry = json.loads(line)
        if "response" in entry and "candidates" in entry['response']:
            if entry['response']['candidates'][0]['finishReason'] == "STOP":
                gemini_cache[entry["key"]] = entry['response']


def translate_with_gemini(request, model, max_tokens, temperature=0.0):
    client = lazy_get_client()
    from google.genai import types


    import hashlib
    if "doc_id" in request:
        hashid = f"{request['doc_id']}_{request['source_language']}_{request['target_language']}_{request['segment']}_{request['prompt_instruction']}"

    else:
        hashid = f"{request['taskid']}_{request['prompt']}"
    hashid = hashlib.md5(hashid.encode("utf-8")).hexdigest()
    if hashid in gemini_cache:
        result = gemini_cache[hashid]
        input_tokens = result['usageMetadata']['promptTokenCount']
        candidate_tokens = result['usageMetadata']['candidatesTokenCount']
        thinking_tokens = result['usageMetadata']['thoughtsTokenCount']
        return result['candidates'][0]['content']['parts'][0]['text'], {"input_tokens": input_tokens,
                           "output_tokens": candidate_tokens + thinking_tokens,
                           "thinking_tokens": thinking_tokens}

    
    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        response_mime_type="text/plain",
        safety_settings=[
            types.SafetySetting(category=category, threshold=types.HarmBlockThreshold.BLOCK_NONE)
            for category in [
                types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            ]
        ],
    )

    try:
        response = client.models.generate_content(
            model=model,
            contents=request["prompt"],
            config=config
        )
    except Exception as e:
        logging.warning(f"Skipping: {e}")
        return None

    if response.candidates[0].finish_reason != "STOP":
        logging.warning(f"Finish reason: {response.candidates[0].finish_reason}; {response.text}")
        return None

    assert response.candidates[0].finish_reason == "STOP", f"Finish reason: {response.choices[0].finish_reason}"


    input_tokens = response.usage_metadata.prompt_token_count
    candidate_tokens = response.usage_metadata.candidates_token_count
    thinking_tokens = response.usage_metadata.thoughts_token_count

    if candidate_tokens is None:
        # gemma has only total prompt token count which equals to input tokens
        thinking_tokens = 0
        candidate_tokens = 0
    
    return response.text, {"input_tokens": input_tokens,
                           "output_tokens": candidate_tokens + thinking_tokens,
                           "thinking_tokens": thinking_tokens}
