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

    # print("DEBUG: skipping testsuites")
    # blindset = blindset[blindset['collection_id']=='general']

    # print("DEBUG: skipping non Czech")
    # blindset = blindset[blindset['tgt_lang']=='cs_CZ']


    answers = collect_answers(blindset, FLAGS.system)

    # make dataframe but drop None items in the list
    df = pd.DataFrame([a for a in answers if a is not None])

    print("###"*10)
    print(f"Collected {len(df)} translations for {FLAGS.system} system")
    print(f"Missing translations: {len(blindset) - len(df)}")


    os.makedirs("wmt_translations", exist_ok=True)
    df.to_json(f"wmt_translations/{FLAGS.system}.jsonl", orient='records', lines=True, force_ascii=False)

if __name__ == '__main__':
    app.run(main)
