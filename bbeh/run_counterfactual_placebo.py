"""
bbeh/run_counterfactual_placebo.py — run the Placebo counterfactual arm.
"""

import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from bbeh import api_client, config, data, official_eval, prompts
from bbeh.reasoner import SolveResult


def main():
    manifest_path = os.path.join(config.WORK_DIR, 'frozen_experiment_manifest', 'counterfactual_manifest.json')
    if not os.path.exists(manifest_path):
        raise SystemExit(f"Manifest not found: {manifest_path}")

    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest_list = json.load(f)
    manifest = {r['query_id']: r for r in manifest_list}

    items = data.select_items(data.load_split('test'), limit_per_task=30)
    out_dir = os.path.join(config.RUNS_DIR, f'memory_zpd_counterfactual_placebo_{config._slug(config.STUDENT_MODEL)}')
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, 'results.jsonl')
    inj_path = os.path.join(out_dir, 'memory_injections.jsonl')

    cache = data.read_jsonl_indexed(results_path, key='id') if os.path.exists(results_path) else {}
    todo = [it for it in items if it['id'] not in cache]

    print(f"Total Placebo Items: {len(items)}, Done: {len(cache)}, Todo: {len(todo)}")

    usage = api_client.TokenUsage()
    client = api_client.build_client(config.STUDENT_MODEL, max_tokens=config.SOLVE_MAX_TOKENS,
                                     temperature=0.0, usage=usage)

    def work(item):
        qid = item['id']
        man = manifest.get(qid, {})
        p_chunk = man.get('placebo_chunk')

        if p_chunk:
            prompt_text = prompts.build_solve_prompt(item, [p_chunk])
            injected = [p_chunk]
        else:
            prompt_text = prompts.build_solve_prompt(item, ())
            injected = []

        t0 = time.time()
        res = client.generate_detailed(prompt_text, max_tokens=config.SOLVE_MAX_TOKENS, temperature=0.0)
        latency = time.time() - t0
        corr, pred, ref = official_eval.score_with_detail(res.text, item.get('target', ''))

        sr = SolveResult(
            id=qid, task=item['task'], arm='memory_counterfactual_placebo', response=res.text,
            prediction=pred, reference=ref,
            correct=corr, outcome='truncated' if res.truncated else ('ok' if res.ok else 'infra_error'),
            error=res.error,
            prompt_tokens=res.prompt_tokens, completion_tokens=res.completion_tokens,
            finish_reason=res.finish_reason, latency_s=latency,
            n_injected=len(injected), injected=injected
        )
        return sr

    if todo:
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(work, it): it for it in todo}
            for fut in as_completed(futures):
                sr = fut.result()
                rec = sr.to_record()
                data.append_jsonl(results_path, rec)
                data.append_jsonl(inj_path, {
                    'id': sr.id, 'task': sr.task,
                    'n_injected': sr.n_injected, 'injected': sr.injected,
                    'retrieval_error': sr.error, 'correct': sr.correct
                })

    print("Placebo run successfully completed.")


if __name__ == '__main__':
    main()
