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
flags.DEFINE_enum('subtask', 'all', ['mist', 'mist_oeg', 'mist_mtqe', 'mist_summ', 'all'], 'Select which subtask to run (mist, mist_oeg, mist_mtqe, all)')

FLAGS = flags.FLAGS

def main(args):
    if FLAGS.subtask in ['mist', 'all']:
        assert os.path.exists("blindset_mist_2025.json"), "Download blindset_mist_2025.json file from WMT website"
        blindset = pd.read_json("blindset_mist_2025.json")
        if FLAGS.parallel:
            blindset = blindset.sample(frac=1, random_state=42).reset_index(drop=True)

        answers = collect_answers(blindset, FLAGS.system, "mist")
        if answers is not None:
            mistdf = pd.DataFrame(answers)

            if not FLAGS.parallel:
                os.makedirs("wmt_mist", exist_ok=True)
                mistdf.to_json(f"wmt_mist/{FLAGS.system}.json", orient='records', force_ascii=False)
            else:
                print("Running in parallel mode, not saving results to disk as the data are shuffled.")

            mist_num_none = mistdf[mistdf['answer'] == "FAILED"]['answer'].count()
            print(f"Number of None answers in MIST: {mist_num_none}")


    if FLAGS.subtask in ['mist_oeg', 'all']:
        assert os.path.exists("llm_judge_oeg_2025_final.json"), "Download llm_judge_oeg_2025_final.json file from WMT website"
        blindset = pd.read_json("llm_judge_oeg_2025_final.json")
        if FLAGS.parallel:
            blindset = blindset.sample(frac=1, random_state=42).reset_index(drop=True)

        answers = collect_answers(blindset, FLAGS.system, "mist_oeg")
        if answers is not None:
            mistdf = pd.DataFrame(answers)

            if not FLAGS.parallel:
                os.makedirs("wmt_mist_oeg", exist_ok=True)
                mistdf.to_json(f"wmt_mist_oeg/{FLAGS.system}.json", orient='records', force_ascii=False)
            else:
                print("Running in parallel mode, not saving results to disk as the data are shuffled.")

            mist_num_none = mistdf[mistdf['answer'] == "FAILED"]['answer'].count()
            print(f"Number of None answers in MIST OEG: {mist_num_none}")


    if FLAGS.subtask in ['mist_summ', 'all']:
        assert os.path.exists("llm_judge_summ_2025_final.json"), "Download llm_judge_summ_2025_final.json file from WMT website"
        blindset = pd.read_json("llm_judge_summ_2025_final.json")
        if FLAGS.parallel:
            blindset = blindset.sample(frac=1, random_state=42).reset_index(drop=True)

        answers = collect_answers(blindset, FLAGS.system, "mist_summ")
        if answers is not None:
            mistdf = pd.DataFrame(answers)

            if not FLAGS.parallel:
                os.makedirs("wmt_mist_summ", exist_ok=True)
                mistdf.to_json(f"wmt_mist_summ/{FLAGS.system}.json", orient='records', force_ascii=False)
            else:
                print("Running in parallel mode, not saving results to disk as the data are shuffled.")

            mist_num_none = mistdf[mistdf['answer'] == "FAILED"]['answer'].count()
            print(f"Number of None answers in MIST SUMM: {mist_num_none}")


    if FLAGS.subtask in ['mist_mtqe', 'all']:
        assert os.path.exists("mteval-task1-test25.jsonl"), "Download mteval-task1-test25.jsonl file from WMT website"
        blindset = pd.read_json("mteval-task1-test25.jsonl", lines=True)
        if FLAGS.parallel:
            blindset = blindset.sample(frac=1, random_state=42).reset_index(drop=True)

        answers = collect_answers(blindset, FLAGS.system, "mist_mtqe")
        if answers is not None:
            mistdf = pd.DataFrame(answers)

            if not FLAGS.parallel:
                mistdf = mistdf.rename(columns={'answer': 'overall'})
                mistdf['doc_id'] = blindset['doc_id']
                mistdf['segment_id'] = blindset['segment_id']
                mistdf['source_lang'] = blindset['source_lang']
                mistdf['target_lang'] = blindset['target_lang']
                mistdf['set_id'] = blindset['set_id']
                mistdf['system_id'] = blindset['system_id']
                mistdf['domain_name'] = blindset['domain_name']
                mistdf['method'] = blindset['method']
                
                os.makedirs("wmt_mist_mtqe", exist_ok=True)
                mistdf.to_csv(f"wmt_mist_mtqe/{FLAGS.system}.tsv", sep='\t', index=False)
            else:
                print("Running in parallel mode, not saving results to disk as the data are shuffled.")

            mist_num_none = mistdf[mistdf['overall'] == "FAILED"]['overall'].count()
            print(f"Number of None answers in MIST MTQE: {mist_num_none}")


if __name__ == '__main__':
    app.run(main)
