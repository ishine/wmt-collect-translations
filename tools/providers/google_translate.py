import os
import pandas as pd
import ipdb

CLIENT = None
SUPPORTED_LANGUAGES = None
def lazy_get_client():
    global CLIENT, SUPPORTED_LANGUAGES
    
    if CLIENT is None:
        from google.cloud import translate_v2 as translate
        assert "OPENAI_API_KEY" in os.environ, "Please set the OPENAI_API_KEY environment variable"
        CLIENT = translate.Client()
    
        SUPPORTED_LANGUAGES = pd.DataFrame(CLIENT.get_languages())

    return CLIENT

def get_supported_languages(lang):
    lazy_get_client()
    if lang in SUPPORTED_LANGUAGES['language'].values:
        return lang
    elif lang.split('_')[0] in SUPPORTED_LANGUAGES['language'].values:
        return lang.split('_')[0]
    else:
        print(f"Language '{lang}' is not supported by Google Translate.")
        return None
        

def translate_with_google_api(request):    
    goog_translate_client = lazy_get_client()

    target_language = get_supported_languages(request['target_language'])
    if target_language is None:
        return None

    result = goog_translate_client.translate(
                    request['segment'],
                    source_language=request['source_language'],
                    target_language=target_language,
                )

    return result.get('translatedText'), None
