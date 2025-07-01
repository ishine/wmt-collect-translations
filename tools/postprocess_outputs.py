"""
Title: Script to postprocess Gemini Translations
Author: Sweta Agrawal
"""

import pandas as pd
import hashlib


def get_hash(row):
    hashid = f"{row['doc_id']}_{row['src_lang']}_{row['tgt_lang']}_{row['src_text']}_{row['prompt_instruction']}"
    hashid = hashlib.md5(hashid.encode("utf-8")).hexdigest()
    return f"{hashid}"


blindset = pd.read_json("data/genmt_blindset_wmt25.jsonl", lines=True)
blindset["key"] = blindset.apply(get_hash, axis=1)

outputs = pd.read_json(
    "outputs_final/wmt-general_mt.jsonl",
    lines=True,
)
assert len(blindset) == len(outputs)

all_outs = []
count_none = 0
count_prohibit = 0
count_nan = 0
for x in outputs["response"].to_list():
    if x != x or "candidates" not in x:
        count_nan += 1
        all_outs.append(None)
        continue
    x = x["candidates"][0]
    if x["finishReason"] == "MAX_TOKENS":
        count_none += 1
        all_outs.append(None)
        continue
    elif x["finishReason"] == "PROHIBITED_CONTENT":
        count_prohibit += 1
        all_outs.append(None)
        continue
    else:
        if "content" in x and "parts" in x["content"]:
            all_outs.append(x["content"]["parts"][0]["text"])
        else:
            all_outs.append(None)

outputs["response_text"] = all_outs
all_keys_out = outputs["key"].to_list()
all_keys_blindset = blindset["key"].to_list()

merged_df = pd.merge(
    blindset, outputs, on="key", how="inner", suffixes=("_test", "_out")
)

assert len(merged_df) == len(outputs)
