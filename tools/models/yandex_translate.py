import os

CLIENT = None
def lazy_get_client():
    global CLIENT
    
    if CLIENT is None:
        from yandex_translate import YandexTranslate
        assert 'YANDEX_APPLICATION_CREDENTIALS' in os.environ, 'Please set the environment variable YANDEX_APPLICATION_CREDENTIALS'
        CLIENT = YandexTranslate(os.environ['YANDEX_APPLICATION_CREDENTIALS'])
    return CLIENT


def translate_with_yandex(segment: str, source_language=None, target_language="de"):
    client = lazy_get_client()
    
    result = client.translate(segment, f'{source_language}-{target_language}')
    return result.get('text')[0]
