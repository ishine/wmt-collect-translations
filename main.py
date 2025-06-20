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

    # print("DEBUG: downsampling dataset")
    # blindset = blindset.head(3090).tail(10)

    # print("DEBUG: skipping testsuites")
    # blindset = blindset[blindset['collection_id']=='general']


    answers = collect_answers(blindset, FLAGS.system)

    # # make dataframe but drop None items in the list
    # df = pd.DataFrame([a for a in answers if a is not None])
    # ipdb.set_trace()

if __name__ == '__main__':
    app.run(main)
