"""
    Title: Machine Translation API command line tool for translating WMT testsets
    Author: Tom Kocmi
"""

import os
import ipdb
import pandas as pd
from absl import flags, app
from tools.utils import collect_answers, SYSTEMS
 

flags.DEFINE_enum('system', 'CommandA', SYSTEMS, 'Define the system to use for translation')
flags.DEFINE_bool('parallel', False, 'Run in parallel mode (default: False)')

FLAGS = flags.FLAGS

def main(args):
    assert os.path.exists("genmt_blindset_wmt25.jsonl"), "Download genmt_blindset_wmt25.jsonl file from WMT website"
    blindset = pd.read_json("genmt_blindset_wmt25.jsonl", lines=True)
    if FLAGS.parallel:
        # avoid clashes by shuffling samples
        blindset = blindset.sample(frac=1, random_state=42).reset_index(drop=True)

    answers = collect_answers(blindset, FLAGS.system)
    df = pd.DataFrame(answers)

    # for each tgt_lang, count how many "FAILED" there are and if more than 25% are FAILED, remove that tgt_lang
    for tgt_lang in df['tgt_lang'].unique():
        num_none = df[df['tgt_lang'] == tgt_lang]['hypothesis'].str.contains("FAILED", na=False).sum()
        if num_none > 0.25 * len(df[df['tgt_lang'] == tgt_lang]):
            df = df[df['tgt_lang'] != tgt_lang]

    if not FLAGS.parallel:
        os.makedirs("wmt_translations", exist_ok=True)
        df.to_json(f"wmt_translations/{FLAGS.system}.jsonl", orient='records', lines=True, force_ascii=False)
    else:
        print("Running in parallel mode, not saving results to disk as the data are shuffled.")

    mt_num_none = df[df['hypothesis'].str.contains("FAILED", na=False)]['hypothesis'].count()
    print(f"Number of untranslated answers in MT: {mt_num_none}")


if __name__ == '__main__':
    app.run(main)
