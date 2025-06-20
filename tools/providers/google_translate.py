import os


CLIENT = None
def lazy_get_client():
    global CLIENT
    
    if CLIENT is None:
        from google.cloud import translate_v2 as translate
        assert "OPENAI_API_KEY" in os.environ, "Please set the OPENAI_API_KEY environment variable"
        CLIENT = translate.Client()
    return CLIENT

def translate_with_google_api(request):    
    goog_translate_client = lazy_get_client()

    target_language = request['target_language'].split("_")[0]
    
    result = goog_translate_client.translate(
                    request['segment'],
                    source_language=request['source_language'],
                    target_language=target_language,
                )

    return result.get('translatedText'), None
