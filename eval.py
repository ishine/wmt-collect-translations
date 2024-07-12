"""
  Title: Machine Translation API command line tool for translating WMT testsets
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
    folder = "test/*.xml"

    sys_scores = {}
    for file in tqdm(glob.glob(folder), "Calculating BLEU over WMT testset", position=0):
        lp = file.split(".")[-2]
        if "-" not in lp:
            continue
        sys_scores[lp] = {}
        
        source_language = lp.split("-")[0]
        target_language = lp.split("-")[1]

        sources = []
        references = []
        
        src_file = f"{file}.no-testsuites.{source_language}"
        ref_file = f"{file}.no-testsuites.{target_language}"

        with open(src_file) as f:
            sources = f.readlines()
            sources = [line.strip() for line in sources]

        with open(ref_file) as f:
            references = f.readlines()
            references = [line.strip() for line in references]

        for system in glob.glob(f"wmt_translations/*"):
            sys_file = system + "/" + src_file.split("/")[-1].replace("2024", "2024.src")
            hypothesis = []
            if not os.path.exists(sys_file):
                continue
            with open(sys_file) as f:
                hypothesis = f.readlines()
                hypothesis = [line.strip() for line in hypothesis]

            assert len(hypothesis) == len(references), f"Length mismatch: {system} {lp}"

            score = sacrebleu.corpus_bleu(hypothesis, [references]).score 
            sys_scores[lp][system] = score
        

    pd.DataFrame(sys_scores).to_excel("BLEU_scores.xlsx")

if __name__ == '__main__':
    app.run(main)
