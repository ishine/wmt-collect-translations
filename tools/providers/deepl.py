import os
from tools.errors import ERROR_UNSUPPORTED_LANGUAGE, FINISH_STOP


CLIENT = None
def lazy_get_client():
    global CLIENT
    
    if CLIENT is None:
        import deepl
        assert "DEEPL_PRO_AUTH_KEY" in os.environ, "Please set the DEEPL_PRO_AUTH_KEY environment variable"
        CLIENT = deepl.Translator(os.environ['DEEPL_PRO_AUTH_KEY'])
    return CLIENT

supported_languages_deepl = [
    "ar", "bg", "cs", "da", "de", "el", "en", "en-gb", "en-us", "es", 
    "et", "fi", "fr", "hu", "id", "it", "ja", "ko", "lt", "lv", 
    "nb", "nl", "pl", "pt", "pt-br", "pt-pt", "ro", "ru", "sk", 
    "sl", "sv", "tr", "uk", "zh"
]

def translate_with_deepl(request):    
    client = lazy_get_client()

    target_language = request['target_language'].split('_')[0]
    if target_language not in supported_languages_deepl:
        return ERROR_UNSUPPORTED_LANGUAGE

    result = client.translate_text(
                request['segment'],
                source_lang=request['source_language'],
                target_lang=target_language,
            )
    
    return result.text, {"finish_reason": FINISH_STOP}
