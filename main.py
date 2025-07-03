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

FLAGS = flags.FLAGS

def main(args):
    assert os.path.exists("genmt_blindset_wmt25.jsonl"), "Download genmt_blindset_wmt25.jsonl file from WMT website"
    blindset = pd.read_json("genmt_blindset_wmt25.jsonl", lines=True)

    answers = collect_answers(blindset, FLAGS.system)
    df = pd.DataFrame(answers)

    os.makedirs("wmt_translations", exist_ok=True)
    df.to_json(f"wmt_translations/{FLAGS.system}.jsonl", orient='records', lines=True, force_ascii=False)


    assert os.path.exists("blindset_mist_2025.json"), "Download blindset_mist_2025.json file from WMT website"
    blindset = pd.read_json("blindset_mist_2025.json")

    answers = collect_answers(blindset, FLAGS.system, "mist")
    if answers is not None:
        mistdf = pd.DataFrame(answers)

        os.makedirs("wmt_mist", exist_ok=True)
        mistdf.to_json(f"wmt_mist/{FLAGS.system}.json", orient='records', force_ascii=False)

        # print number of None answers
        mist_num_none = mistdf[mistdf['answer'] == "FAILED"]['answer'].count()
        print(f"Number of None answers in MIST: {mist_num_none}")

    mt_num_none = df[df['hypothesis'].str.contains("FAILED", na=False)]['hypothesis'].count()
    print(f"Number of untranslated answers in MT: {mt_num_none}")

if __name__ == '__main__':
    app.run(main)
