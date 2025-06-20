import os
import urllib.request
import json
import os
import ssl


def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

CLIENT = None
def lazy_get_client():
    global CLIENT
    allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.
    return CLIENT




def translate_with_phi3_medium(prompt):
    client = lazy_get_client()
    
    data =  {
        "messages": prompt,
        "max_tokens": 2048,
        "temperature": 0,
        "top_p": 1
    }

    body = str.encode(json.dumps(data))

    url = 'https://Phi-3-medium-4k-instruct-yxgtn-serverless.eastus2.inference.ai.azure.com/v1/chat/completions'
    api_key = os.environ.get("PHI_API_KEY")
    headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}
    req = urllib.request.Request(url, body, headers)

    try:
        response = urllib.request.urlopen(req, timeout=120)

        response = json.loads(response.read().decode("utf8", 'ignore'))
    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))
        print(error.info())
        print(error.read().decode("utf8", 'ignore'))
        return None
    # except TimeoutError as e:
    #     print("The request timed out")
    #     return None

    if response['choices'][0]['finish_reason'] != "stop":
        return None

    assert response['choices'][0]['finish_reason'] == "stop", f"Finish reason: {response['choices'][0]['finish_reason']}"

    return response['choices'][0]['message']['content'], {"input_tokens": response['usage']['prompt_tokens'],
                                                           "output_tokens": response['usage']['completion_tokens'],
                                                           "thinking_tokens": 0}

