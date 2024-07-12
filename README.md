# WMT Collecting Translations

This tool is used to collect translations from various providers and LLMs that may be used at WMT General MT.


# Usage

The tool is using 3-shot prompting when translating with LLMs.

## Setting up secrets

You need to set one or multiple following secrets for the full utilization:

```
export MTAPI_SUBSCRIPTION_KEY=          # Microsoft Azure API key
export GOOGLE_APPLICATION_CREDENTIALS=  # Google credentials in json file
export DEEPL_PRO_AUTH_KEY=              # DeepL credentials
export YANDEX_APPLICATION_CREDENTIALS=  # Yandex API key
export TOGETHER_API_KEY=                # Together API key for LLama 3
export COHERE_API_KEY=                  # Cohere key
export OPENAI_AZURE_ENDPOINT=           # OpenAI Azure endpoint URL
export OPENAI_AZURE_KEY=                # OpenAI Azure key
export MISTRAL_KEY=                     # Mistral API key
export GEMINI_API_KEY=                  # Gemini API key for Google AI Studio
export ANTHROPIC_API_KEY=               # Anthropic key for claude
export PHI_API_KEY=                     # API key for Phi model
```


## Download WMT testsets

Download latest blindset and rename to `wmt_testset`:
```
wget https://www2.statmt.org/wmt24/WMT24_GeneralMT.zip
unzip WMT24_GeneralMT.zip
mv WMT24_GeneralMT wmt_testset
```

Extract XML into txt, extract also without testsuites:

```
for file in `ls wmt_testset/*.xml --color=no`; do cat $file | wmt-unwrap -o $file.full; done

for file in `ls wmt_testset/*.xml --color=no`; do cat $file | wmt-unwrap -o $file.no-testsuites --no-testsuites; done
```


## Running translations

```
python main.py --system='SYSTEM'
```

Where SYSTEM is any of `MicrosoftTranslator|GoogleTranslate|DeepL|YandexTranslate|GPT-4|Llama3-70B|CommandR-plus|Aya23|Mistral-Large|GPT-4o|Claude-3.5|Gemini-1.5-Pro|Phi-3-Medium`. It should be easily extended to other models.

