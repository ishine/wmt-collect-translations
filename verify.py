"""
  Title: Machine Translation API command line tool for translating WMT testsets
  Purpose: To validate that translation pipeline is not broken with BLEU rather than checking quality of translations.
  Author: Tom Kocmi
"""

import os
import glob
import ipdb
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
    
    granularity = {}
    for file in tqdm(glob.glob('wmt_translations/*.jsonl'), "Verifying systems", position=0):
        print(f"Processing {file}...")
        system = os.path.basename(file).replace(".jsonl", "")
        df = pd.read_json(file, lines=True)
        df['lp'] = df['doc_id'].apply(lambda x: x.split("_#_")[0])
        # get count of individual translation granularities
        granularity[system] = df['translation_granularity'].value_counts().to_dict()


        # add a column ref into df with reference translations by aligning on doc_id column
        df = df.merge(ref[['doc_id', 'hypothesis']], on='doc_id', how='left', suffixes=('', '_ref'))

        try:
            for lp in df['lp'].unique():
                subdf = df[df['lp'] == lp]

                lensubdf = len(subdf)
                # drop na in translation and translation_ref
                subdf = subdf.dropna(subset=['hypothesis', 'hypothesis_ref'])
                if len(subdf) != lensubdf:
                    print(f"Warning: {lensubdf - len(subdf)} translations missing for {lp} in {system}")
                    continue

                if lp not in sys_scores:
                    sys_scores[lp] = {}

                score = sacrebleu.corpus_bleu(subdf['hypothesis'].to_list(), [subdf['hypothesis_ref'].to_list()]).score 

                sys_scores[lp][system] = score
        except Exception as e:
            print(f"Error processing {system} for language pair {lp}: {e}")
            continue
        
    scores = pd.DataFrame(sys_scores)
    scores['average'] = scores.mean(axis=1)
    scores = scores.sort_values(by='average', ascending=False)
    scores = scores[['average'] + [col for col in scores.columns if col != 'average']]
    scores = scores.round(1)
    print("\n\nBLEU with GoogleTranslate:")    
    print(scores)

    print("\n\nTranslation granularity:")
    gran = pd.DataFrame(granularity).T
    gran = gran.div(gran.sum(axis=1), axis=0) * 100
    gran = gran.round(1)
    print(gran)

    # for each row, find which columns it has NaN values and print them
    for system, row in scores.iterrows():
        missing = row[row.isna()].index.tolist()
        if missing:
            print(f"{system} is missing scores for:\n{', '.join(missing)}\n")

    ipdb.set_trace()

if __name__ == '__main__':
    app.run(main)
