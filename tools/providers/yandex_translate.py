import os

CLIENT = None
def lazy_get_client():
    global CLIENT
    
    if CLIENT is None:
        from yandex_translate import YandexTranslate
        assert 'YANDEX_APPLICATION_CREDENTIALS' in os.environ, 'Please set the environment variable YANDEX_APPLICATION_CREDENTIALS'
        CLIENT = YandexTranslate(os.environ['YANDEX_APPLICATION_CREDENTIALS'])
    return CLIENT


def translate_with_yandex(request):
    client = lazy_get_client()
    
    source_language = request['source_language']
    target_language = request['target_language'].split("_")[0]  # Handle cases like 'en_US' to 'en'
    
    result = client.translate(request['segment'], f'{source_language}-{target_language}')
    assert result.get('code') == 200, f"Yandex Translate API error: {result.get('code')} - {result.get('text')}"

    return result.get('text')[0], None
