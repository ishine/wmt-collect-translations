"""
Title: Script to run and collect Gemini Translations in batch mode
Author: Sweta Agrawal
"""

import tqdm
import hashlib
import json
import os
import ipdb
import pandas as pd
from google import genai
from google.genai import types
from absl import flags, app
from tools.utils import SYSTEMS
from datetime import datetime
import time
import logging

assert (
    "GEMINI_API_KEY" in os.environ
), "Please set the GEMINI_API_KEY environment variable"

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

flags.DEFINE_enum(
    "system",
    "gemini-2.5-pro",
    ["gemini-2.5-pro", "gemma-3-12b-it", "gemma-3-27b-it"],
    "Define the system to use for translation",
)
flags.DEFINE_enum(
    "task",
    "wmt_mist",
    ["wmt_mist", "wmt_translations"],
    "Define the system to use for translation",
)

FLAGS = flags.FLAGS


def create_requests_and_upload(blindset, system_name, task):
    requests = []
    unsupported_languages = []

    for _, row in tqdm.tqdm(blindset.iterrows(), total=len(blindset), desc=system_name):
        if task == "general_mt":
            # completely skip unsupported languages
            if (row["src_lang"], row["tgt_lang"]) in unsupported_languages:
                continue

            # create hash merging all the information in the request
            hashid = f"{row['doc_id']}_{row['src_lang']}_{row['tgt_lang']}_{row['src_text']}_{row['prompt_instruction']}"
            hashid = hashlib.md5(hashid.encode("utf-8")).hexdigest()

            request = {
                "key": hashid,
                "request": {
                    "contents": [
                        {
                            "parts": [
                                {
                                    "text": f"{row['prompt_instruction']}\n\n{row['src_text']}"
                                }
                            ]
                        }
                    ],
                    "generationConfig": {"temperature": 0.0, "maxOutputTokens": 32768},
                },
                "thinking_config": {
                    "includeThoughts": True,
                    "thinkingBudget": 14029,
                },
            }
            requests.append(request)

        elif task == "mist":
            hashid = f"{row['taskid']}_{row['prompt']}"
            hashid = hashlib.md5(hashid.encode("utf-8")).hexdigest()
            # create hash merging all the information in the request
            request = {
                "key": hashid,
                "request": {
                    "contents": [{"parts": [{"text": row["prompt"]}]}],
                    "generationConfig": {"temperature": 0.0, "maxOutputTokens": 32768},
                },
                "thinking_config": {
                    "includeThoughts": True,
                    "thinkingBudget": 14029,
                },
            }
            requests.append(request)

    # Upload the file to the File API
    logging.info(f"Created {len(requests)} requests")
    logging.info(f"First request: {requests[0]}")
    with open(f"batch-requests-{task}.json", "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")

    logging.info(f"Uploading to file")
    uploaded_file = client.files.upload(
        file=f"batch-requests-{task}.json",
        config=types.UploadFileConfig(display_name="batch-requests-{task}"),
    )
    logging.info(f"Uploaded to file: {uploaded_file.name}")

    return uploaded_file


def collect_answers_batch_gemini(blindset, system_name, task="general_mt"):
    uploaded_file = create_requests_and_upload(blindset, system_name, task)
    logging.info(f"Creating batches using file: {uploaded_file.name}")
    file_batch_job = client.batches.create(
        model=system_name,
        src=uploaded_file.name,
        config={
            "display_name": f"batch-job-{task}",
        },
    )

    job_name = file_batch_job.name
    batch_job = client.batches.get(name=job_name)

    completed_states = set(
        [
            "JOB_STATE_SUCCEEDED",
            "JOB_STATE_FAILED",
            "JOB_STATE_CANCELLED",
        ]
    )
    logging.info(f"Polling status for job: {job_name}")
    batch_job = client.batches.get(name=job_name)  # Initial get
    logging.info(f"Batch name: {batch_job.state.name}")

    while batch_job.state.name not in completed_states:
        print(f"Current state: {batch_job.state.name}")
        time.sleep(30)  # Wait for 30 seconds before polling again
        batch_job = client.batches.get(name=job_name)

    logging.info(f"Job finished with state: {batch_job.state.name}")

    if batch_job.state.name == "JOB_STATE_SUCCEEDED":
        # If batch job was created with a file
        if batch_job.dest and batch_job.dest.file_name:
            # Results are in a file
            result_file_name = batch_job.dest.file_name
            logging.info(f"Results are in file: {result_file_name}")

            logging.info("Downloading result file content...")
            file_content = client.files.download(file=result_file_name)
            # Process file_content (bytes) as needed
            logging.info(file_content.decode("utf-8"))
        else:
            logging.error("No results found")

    elif batch_job.state.name == "JOB_STATE_FAILED":
        logging.error(f"Error: {batch_job.error}")

    answers = file_content.decode("utf-8")
    return answers


def main(args):
    if FLAGS.task == "wmt_translations":
        assert os.path.exists(
            "genmt_blindset_wmt25.jsonl"
        ), "Download genmt_blindset_wmt25.jsonl file from WMT website"
        blindset = pd.read_json("genmt_blindset_wmt25.jsonl", lines=True)

        answers = collect_answers_batch_gemini(blindset, FLAGS.system)
        if answers is not None:
            os.makedirs("wmt_translations", exist_ok=True)
            with open(f"wmt_translations/{FLAGS.system}.jsonl", "w") as f:
                for answer in answers:
                    f.write(answer + "\n")
    elif FLAGS.task == "wmt_mist":
        assert os.path.exists(
            "blindset_mist_2025.json"
        ), "Download blindset_mist_2025.json file from WMT website"
        blindset = pd.read_json("blindset_mist_2025.json")
        logging.info(f"Read blindset with {len(blindset)} lines")
        answers = collect_answers_batch_gemini(blindset, FLAGS.system, "mist")
        if answers is not None:
            os.makedirs("wmt_mist", exist_ok=True)
            with open(f"wmt_mist/{FLAGS.system}.jsonl", "w") as f:
                for answer in answers:
                    f.write(answer + "\n")
    else:
        print(f"Task not found: {FLAGS.task}")


if __name__ == "__main__":
    app.run(main)
