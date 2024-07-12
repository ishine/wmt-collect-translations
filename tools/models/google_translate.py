import os


CLIENT = None
def lazy_get_client():
    global CLIENT
    
    if CLIENT is None:
        from google.cloud import translate_v2 as translate
        assert "GOOGLE_APPLICATION_CREDENTIALS" in os.environ, "Please set the GOOGLE_APPLICATION_CREDENTIALS environment variable"
        CLIENT = translate.Client()
    return CLIENT

def translate_with_google_api(segment: str, source_language=None, target_language="de"):    
    goog_translate_client = lazy_get_client()
    
    result = goog_translate_client.translate(
                    segment,
                    source_language=source_language,
                    target_language=target_language,
                )
    
    return result.get('translatedText')
