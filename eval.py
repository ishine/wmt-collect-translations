"""
  Title: Machine Translation API command line tool for translating WMT testsets
  Purpose: To validate that translation pipeline is not broken with BLEU rather than checking quality of translations.
  Author: Tom Kocmi
"""

import os
import glob
import sacrebleu
import pandas as pd
from tqdm import tqdm
from absl import flags, app


FLAGS = flags.FLAGS


def main(args):
    print("Calculating BLEU over WMT testset to check for problems. Do not use BLEU for comparing quality of translations")

    sys_scores = {}
    reference_system = "GoogleTranslate"
    ref = pd.read_json("wmt_translations/" + reference_system + ".jsonl", lines=True)
    ref['lp'] = ref['doc_id'].apply(lambda x: x.split("_#_")[0])
    for file in tqdm(glob.glob('wmt_translations/*.jsonl'), "Verifying systems", position=0):
        system = os.path.basename(file).replace(".jsonl", "")
        df = pd.read_json(file, lines=True)

        # keep only rows in reference system over column doc_id
        df = df[df['doc_id'].isin(ref['doc_id'])]

        if len(df) != len(ref):
            print(f"Length mismatch for system {file}: {len(df)} vs {len(ref)}")
            continue

        # add a column ref into df with reference translations by aligning on doc_id column
        df = df.merge(ref[['doc_id', 'translation']], on='doc_id', how='left', suffixes=('', '_ref'))

        df['lp'] = ref['doc_id'].apply(lambda x: x.split("_#_")[0])
        for lp in df['lp'].unique():
            if lp not in sys_scores:
                sys_scores[lp] = {}

            score = sacrebleu.corpus_bleu(df['translation'].to_list(), [df['translation_ref'].to_list()]).score 
            sys_scores[lp][system] = score
        
    scores = pd.DataFrame(sys_scores)
    # sort by average over all columns
    scores['average'] = scores.mean(axis=1)
    scores = scores.sort_values(by='average', ascending=False)
    
    print(scores)

if __name__ == '__main__':
    app.run(main)
