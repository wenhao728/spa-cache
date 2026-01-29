import argparse
import json
import os
import sys
from pathlib import Path

import evaluate as hf_evaluate

project_dir = str(Path(__file__).resolve().parents[2])
if project_dir not in sys.path:
    sys.path.append(project_dir)

from diff_cache.utils.sanitize import sanitize

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
pass_at_k = hf_evaluate.load("code_eval")


def parse_args():
    parser = argparse.ArgumentParser(description="Post-process HumanEval results.")
    parser.add_argument("--doc_file_name", type=Path, required=True, help="File to save the docs.")
    parser.add_argument("--res_file_name", type=Path, required=True, help="File to save the results.")
    parser.add_argument("--pass_at_k", type=int, default=1, help="Value of k for Pass@k metric.")
    args = parser.parse_args()
    args.doc_file_name = args.doc_file_name.with_suffix(".jsonl")
    args.res_file_name = args.res_file_name.with_suffix(".jsonl")
    return args


def pass_at_k_metric(references, predictions, k):
    return pass_at_k.compute(
        references=references,
        predictions=predictions,
        k=[k],
    )[0][f"pass@{k}"]


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line))
    return data


def write_jsonl(data, file_path):
    with open(file_path, 'w', encoding="utf-8") as file:
        for item in data:
            file.write(json.dumps(item) + '\n')


def main():
    args = parse_args()
    print('=' * 20)
    print(f"Post-processing results in {args.res_file_name.name}...")
    doc = read_jsonl(args.doc_file_name)
    resps = read_jsonl(args.res_file_name)
    n_samples = len(doc)
    assert len(resps) == n_samples * args.pass_at_k, (
        f"Number of responses ({len(resps)}) does not match number of samples in doc ({n_samples}) times"
        f" times pass_at_k ({args.pass_at_k})."
    )

    references = [sample['target'] for sample in doc]
    predictions = [[] for _ in doc]
    
    for i, r in enumerate(resps):
        doc_idx = i % n_samples
        predictions[doc_idx].append(
            sanitize(
                doc[doc_idx]['doc']['prompt'] + "\n" + r.split('```python\n', 1)[-1].split('```')[0], 
                doc[doc_idx]['doc']["entry_point"]
            )
        )

    pass_at_ks = [
        pass_at_k_metric([reference], [prediction], args.pass_at_k) 
        for reference, prediction in zip(references, predictions)
    ]
    # breakpoint()
    pass_res = sum(pass_at_ks)/len(pass_at_ks)
    # pass_std = (sum((x - pass_res) ** 2 for x in pass_at_ks) / (len(pass_at_ks) - 1)) ** 0.5
    print(f"Pass@{args.pass_at_k}: {pass_res:.2%}")

    res = [
        {"task_id": d['doc']['task_id'], "completion": pred, f"pass_at_{args.pass_at_k}": res} 
        for d, pred, res  in zip(doc, predictions, pass_at_ks)
    ]
    res = [{f'pass_at_{args.pass_at_k}': pass_res}] + res
    write_jsonl(res, args.res_file_name.with_suffix(".cleaned.jsonl"))


if __name__ == "__main__":
    main()