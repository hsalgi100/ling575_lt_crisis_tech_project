#!/usr/bin/env python3
"""
Build per-record gemma and qwen result files that match the Google Translate
JSONL schema, by joining:

  1. a per-record translations file (JSONL) -> supplies id / scenario /
     target languages for each record, and
  2. the aggregated bleu summary JSON (the by_language block) -> supplies
     bleu / chrf / chrf++ per language per model.

The summary only stores ONE score per language (a macro-average over records),
so every record of a given language receives that same language-level score.

If only one translations file is given, it is reused as the record skeleton for
both models (ids are the same across models).
"""

import argparse
import json
import re
import sys

METRICS = ["bleu", "chrf", "chrf++"]

# model -> (summary key in by_language, technology label written to output)
MODEL_CONFIG = {
    "gemma": {"summary_key": "gemma4", "label": "Gemma4"},
    "qwen":  {"summary_key": "qwen35", "label": "Qwen35"},
}


def scenario_from(file_field, record_id):
    """Derive the scenario name the way the Google file does.

    Google: file 'kc_shelter_in_place' / id 'kc-shelter_in_place-000'
            -> scenario 'shelter_in_place'.
    """
    if file_field:
        return re.sub(r"^kc[_-]", "", file_field)
    if record_id:
        # strip leading 'kc-' and a trailing '-<number>'
        return re.sub(r"-\d+$", "", re.sub(r"^kc[_-]", "", record_id))
    return None


def load_language_scores(summary_path, summary_key):
    """Return {lang: {bleu, chrf, chrf++}} for one model from the summary."""
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    by_language = data.get("by_language")
    if not isinstance(by_language, list):
        raise SystemExit("Summary has no 'by_language' list.")
    scores = {}
    for entry in by_language:
        lang = entry.get("lang")
        block = entry.get(summary_key)
        if lang and isinstance(block, dict):
            scores[lang] = {m: block.get(m) for m in METRICS}
    return scores


def load_records(translations_path):
    """Yield (id, scenario, [target_languages]) for each translation record."""
    with open(translations_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rid = rec.get("id")
            scenario = scenario_from(rec.get("file"), rid)
            langs = list((rec.get("translations") or {}).keys())
            yield rid, scenario, langs


def write_model_file(translations_path, scores, label, source_lang, out_path):
    n_rows = 0
    missing_langs = set()
    with open(out_path, "w", encoding="utf-8") as out:
        for rid, scenario, langs in load_records(translations_path):
            for lang in langs:
                metrics = scores.get(lang)
                if metrics is None:
                    missing_langs.add(lang)
                    metrics = {m: None for m in METRICS}
                row = {
                    "id": rid,
                    "scenario": scenario,
                    "technology": label,
                    "source_language": source_lang,
                    "target_language": lang,
                }
                for m in METRICS:
                    row[m] = metrics.get(m)
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_rows += 1
    return n_rows, missing_langs


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bleu-summary", required=True)
    p.add_argument("--gemma-translations")
    p.add_argument("--qwen-translations")
    p.add_argument("--gemma-out", default="gemma_results.jsonl")
    p.add_argument("--qwen-out", default="qwen_results.jsonl")
    p.add_argument("--source-lang", default="en")
    args = p.parse_args()

    if not args.gemma_translations and not args.qwen_translations:
        p.error("provide at least one of --gemma-translations / --qwen-translations")

    # If one model's translations are missing, reuse the other's as skeleton.
    gemma_tr = args.gemma_translations or args.qwen_translations
    qwen_tr = args.qwen_translations or args.gemma_translations

    jobs = [
        ("gemma", gemma_tr, args.gemma_out),
        ("qwen", qwen_tr, args.qwen_out),
    ]

    for model, tr_path, out_path in jobs:
        cfg = MODEL_CONFIG[model]
        scores = load_language_scores(args.bleu_summary, cfg["summary_key"])
        if not scores:
            print(f"[warn] no '{cfg['summary_key']}' scores found in summary; "
                  f"{model} rows will have null metrics.", file=sys.stderr)
        n_rows, missing = write_model_file(
            tr_path, scores, cfg["label"], args.source_lang, out_path)
        print(f"{cfg['label']}: wrote {n_rows} rows -> {out_path}")
        if missing:
            print(f"  [note] no summary score for languages: "
                  f"{', '.join(sorted(missing))} (metrics left null)",
                  file=sys.stderr)


if __name__ == "__main__":
    main()