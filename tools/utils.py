import re
import os
import time
import ipdb
import glob
import logging
import traceback
import hashlib
import pandas as pd
import diskcache as dc
from tqdm import tqdm
from collections import defaultdict

from tools.providers.cohere import process_with_command_A, process_with_command_R7B, process_with_aya_expanse_32B, process_with_aya_expanse_8B
from tools.providers.together_ai import process_with_deepseek_v3, process_qwen3_235b, process_with_llama_4_maverick, process_with_llama_4_scout, process_with_mistral_7b
from tools.providers.openai import process_with_openai_gpt4_1
from tools.providers.anthropic import process_with_claude_3_7, process_with_claude_4
from tools.providers.google_translate import translate_with_google_api
from tools.providers.yandex_translate import translate_with_yandex
from tools.providers.mistral import process_with_mistral_medium, process_with_magistral_medium
from tools.providers.gemini import process_with_gemini_2_5_pro, process_with_gemma_3_12b, process_with_gemma_3_27b
from tools.providers.microsoft_translator import translate_with_microsoft_api
from tools.providers.deepl import translate_with_deepl
from tools.providers.phi import translate_with_phi3_medium
from tools.errors import ERROR_MAX_TOKENS, ERROR_UNSUPPORTED_LANGUAGE


SYSTEMS = {
    'CommandA': process_with_command_A,
    'CommandR7B': process_with_command_R7B,
    'AyaExpanse-32B': process_with_aya_expanse_32B,
    'AyaExpanse-8B': process_with_aya_expanse_8B,
    'DeepSeek-V3': process_with_deepseek_v3,
    'Qwen3-235B': process_qwen3_235b,
    'Llama-4-Maverick': process_with_llama_4_maverick,
    'Llama-4-Scout': process_with_llama_4_scout,
    'Mistral-7B': process_with_mistral_7b,
    'GPT-4.1': process_with_openai_gpt4_1,
    'Claude-4': process_with_claude_4,
    "Mistral-Medium": process_with_mistral_medium,
    'YandexTranslate': translate_with_yandex,
    'GoogleTranslate': translate_with_google_api,
    'Gemini-2.5-Pro': process_with_gemini_2_5_pro,
    'Gemma-3-12B': process_with_gemma_3_12b,
    'Gemma-3-27B': process_with_gemma_3_27b,

    'DeepL': translate_with_deepl,

    "Magistral-Medium": process_with_magistral_medium, # model is not able to translate at all
    'MicrosoftTranslator': translate_with_microsoft_api,
    'Phi-3-Medium': translate_with_phi3_medium,
}

non_prompt_systems = ['YandexTranslate', 'GoogleTranslate', 'DeepL', 'MicrosoftTranslator']


def check_paragraph_alignment(source_text, translated_text):
    """
    Check if the number of paragraphs in the source and translated text are the same. This is requirement of GenMT 2025
    """
    source_paragraphs = source_text.split('\n\n')
    translated_paragraphs = translated_text.split('\n\n')

    if len(source_paragraphs) != len(translated_paragraphs):
        return False
    return True


def remove_tripple_quotes(text):
    # check if there are exactly two occurences of ``` in the text
    if text.count("```") == 2:
        # get only the text inbetween the tripple quotes
        text = text.split("```")[1]

    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]

    return text


def _process_document_level(system_name, request, translation_granularity):
    if translation_granularity == 'document-level':
        request['prompt'] = f"{request['prompt_instruction']}\n\n{request['segment']}"
    elif translation_granularity == 'document-level-wrapped':
        request['prompt'] = f"{request['prompt_instruction']}\n\n```{request['segment']}```"
    elif translation_granularity == 'document-level-html':
        segment = request['segment'].replace('\n\n', '\n<br>\n\n')
        if "Please translate the following" in request['prompt_instruction']:
            instruction = request['prompt_instruction'].replace('Please translate the following', 'Keep HTML tags in the answer. Please translate the following')
        elif "Translate the following" in request['prompt_instruction']:
            # used in Testsuites
            instruction = request['prompt_instruction'].replace('Translate the following', 'Keep HTML tags in the answer. Translate the following')
        else:
            raise ValueError("Prompt instruction should contain 'Please translate the following' or 'Translate the following'")
        request['prompt'] = f"{instruction}\n\n{segment}"

    answer = SYSTEMS[system_name](request)

    if answer is None or answer in [ERROR_MAX_TOKENS, ERROR_UNSUPPORTED_LANGUAGE]:
        return answer

    answer, tokens = answer

    if translation_granularity == 'document-level-wrapped':
        answer = remove_tripple_quotes(answer)
    elif translation_granularity == 'document-level-html':
        answer = re.sub(r'\n*\s*<br>\s*\n*', '<br>', answer)
        answer = re.sub(r'\n{2,}', '\n', answer)
        answer = answer.replace('<br>', '\n\n')
    
    return answer, tokens


def _process_line_level(system_name, request):
    answers = []
    tokens = {}
    seg_request = request.copy()
    for sentence in request['segment'].split('\n'):
        seg_request['prompt'] = f"{request['prompt_instruction']}\n\n{sentence}"
        seg_request['segment'] = sentence

        response = SYSTEMS[system_name](seg_request)
        if response is None or response == ERROR_MAX_TOKENS:
            return response

        translated_sentence, sentence_tokens = response
        translated_sentence = re.sub(r'\n{1,}', ' ', translated_sentence)
        answers.append(translated_sentence.strip('\n'))
        # add tokens to the dictionary
        if sentence_tokens is not None:
            for key, value in sentence_tokens.items():
                if key not in tokens:
                    tokens[key] = 0
                tokens[key] += value
    return '\n'.join(answers), tokens


def _process_paragraph_level(system_name, request, translation_granularity='paragraph-level'):
    answers = []
    tokens = {}
    seg_request = request.copy()
    for paragraph in request['segment'].split('\n\n'):
        seg_request['prompt'] = f"{request['prompt_instruction']}\n\n{paragraph}"
        seg_request['segment'] = paragraph

        response = SYSTEMS[system_name](seg_request)
        if response == ERROR_MAX_TOKENS:
            # there are few long paragraphs in testsuites, use sentence-level only for those
            response = _process_line_level(system_name, seg_request)
            translation_granularity = 'line-level'

        if response is None:
            return None
        
        translated_paragraph, paragraph_tokens = response
        translated_paragraph = re.sub(r'\n{2,}', '\n', translated_paragraph)
        answers.append(translated_paragraph.strip())
        # add tokens to the dictionary
        if paragraph_tokens is not None:
            for key, value in paragraph_tokens.items():
                if key not in tokens:
                    tokens[key] = 0
                tokens[key] += value
    return ('\n\n'.join(answers), tokens), translation_granularity


def _request_system(system_name, request):
    attempt_document_level = True
    for translation_granularity in ['document-level', 'document-level-wrapped', 'document-level-html', 'paragraph-level']:
        if not attempt_document_level and translation_granularity != 'paragraph-level':
            continue

        if "document-level" in translation_granularity:
            # non-LLM systems are not affected by document-level variants
            if system_name in non_prompt_systems and translation_granularity != 'document-level':
                continue
            answer = _process_document_level(system_name, request, translation_granularity)
        elif translation_granularity == 'paragraph-level':
            answer, translation_granularity = _process_paragraph_level(system_name, request, translation_granularity)

        if answer is None:
            print(f"System {system_name} returned None for doc_id {request['doc_id']} with translation granularity {translation_granularity}")
            continue
        if answer == ERROR_MAX_TOKENS:
            attempt_document_level = False
            continue
        if answer == ERROR_UNSUPPORTED_LANGUAGE:
            return None

        answer, tokens = answer

        answer = answer.strip()
        if check_paragraph_alignment(request['segment'], answer):
            return {
                'doc_id': request['doc_id'],
                'translation': answer,
                'translation_granularity': translation_granularity,
                "tokens": tokens
            }
        else:
            print(f"Paragraph alignment failed for {system_name} on doc_id {request['doc_id']} with {translation_granularity}.")

    return None


def collect_answers(blindset, system_name):
    cache = dc.Cache(f'cache/{system_name}', expire=None, size_limit=int(10e10), cull_limit=0, eviction_policy='none')

    answers = []
    for _, row in tqdm(blindset.iterrows(), total=len(blindset), desc=system_name):
        request = {
            'doc_id': row['doc_id'],
            'source_language': row['src_lang'],
            'target_language': row['tgt_lang'],
            'segment': row['src_text'],
            'prompt_instruction': row['prompt_instruction']
        }
        # create hash merging all the information in the request
        hashid = f"{request['doc_id']}_{request['source_language']}_{request['target_language']}_{request['segment']}_{request['prompt_instruction']}"
        hashid = hashlib.md5(hashid.encode('utf-8')).hexdigest()

        # None represent problem in the translation that was originally skipped
        if hashid not in cache or cache[hashid] is None:
            try:
                cache[hashid] = _request_system(system_name, request)
            except Exception as e:
                logging.error(f"Error processing {request['doc_id']} with {system_name}: {e}")
                logging.error(traceback.format_exc())
                cache[hashid] = None

        answers.append(cache[hashid])

        # # if most are None, the system doesn't support the language
        # if sum([1 for t in answers if t is None]) > len(answers) / 2:
        #     logging.info(f"Skipping {lp} as it is not supported by {system_name}")
        #     continue

        # input_token_count = 0
        # output_token_count = 0
        # with open(target_filename, "w") as f:
        #     for line in translated:
        #         translation = line
        #         # if line is tupple, then it contains also the token count information
        #         if isinstance(line, tuple):
        #             translation = line[0]
        #             input_token_count += line[1][0]
        #             output_token_count += line[1][1]
        #         elif line is None:
        #             translation = ""
        #             none_counter += 1

        #         if SYSTEMS[system_name]["prompt"] is not None:
        #             translation = remove_tripple_quotes(translation)
                
        #         f.write(translation + "\n")

        # # save token count information into file
        # with open(f"{target_filename}.tokens", "w") as f:
        #     f.write(f"Input tokens: {input_token_count}\n")
        #     f.write(f"Output tokens: {output_token_count}\n")

    return answers