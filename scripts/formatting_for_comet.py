#!/usr/bin/env python3
"""
Build per-record gemma and qwen COMET result files in the same row format as
the bleu/chrf files

NOTE: the summary holds one comet value per language, so
every record of a given language receives that same language-level value.

If only one translations file is given, it is reused as the skeleton for both
models (ids are the same across models).
"""

import argparse
import json
import re
import sys

# model -> (flat comet key in by_language, technology label written to output)
MODEL_CONFIG = {
    "gemma": {"comet_key": "gemma4_comet", "label": "Gemma4"},
    "qwen":  {"comet_key": "qwen35_comet", "label": "Qwen35"},
}


def scenario_from(file_field, record_id):
    """Derive the scenario name the way the Google file does.

    'kc_winter_storm' / 'kc-winter_storm-032' -> 'winter_storm'.
    """
    if file_field:
        return re.sub(r"^kc[_-]", "", file_field)
    if record_id:
        return re.sub(r"-\d+$", "", re.sub(r"^kc[_-]", "", record_id))
    return None


def load_language_comet(summary_path, comet_key):
    """Return {lang: comet_value} for one model from the COMET summary."""
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    by_language = data.get("by_language")
    if not isinstance(by_language, list):
        raise SystemExit("COMET summary has no 'by_language' list.")
    scores = {}
    for entry in by_language:
        lang = entry.get("lang")
        if lang is not None and comet_key in entry:
            scores[lang] = entry.get(comet_key)
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


def write_model_file(translations_path, comet_scores, label, source_lang, out_path):
    n_rows = 0
    missing_langs = set()
    with open(out_path, "w", encoding="utf-8") as out:
        for rid, scenario, langs in load_records(translations_path):
            for lang in langs:
                if lang in comet_scores:
                    comet = comet_scores[lang]
                else:
                    missing_langs.add(lang)
                    comet = None
                row = {
                    "id": rid,
                    "scenario": scenario,
                    "technology": label,
                    "source_language": source_lang,
                    "target_language": lang,
                    "comet": comet,
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_rows += 1
    return n_rows, missing_langs


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--comet-summary", required=True)
    p.add_argument("--gemma-translations")
    p.add_argument("--qwen-translations")
    p.add_argument("--gemma-out", default="gemma_comet_results.jsonl")
    p.add_argument("--qwen-out", default="qwen_comet_results.jsonl")
    p.add_argument("--source-lang", default="en")
    args = p.parse_args()

    if not args.gemma_translations and not args.qwen_translations:
        p.error("provide at least one of --gemma-translations / --qwen-translations")

    gemma_tr = args.gemma_translations or args.qwen_translations
    qwen_tr = args.qwen_translations or args.gemma_translations

    jobs = [
        ("gemma", gemma_tr, args.gemma_out),
        ("qwen", qwen_tr, args.qwen_out),
    ]

    for model, tr_path, out_path in jobs:
        cfg = MODEL_CONFIG[model]
        comet_scores = load_language_comet(args.comet_summary, cfg["comet_key"])
        if not comet_scores:
            print(f"[warn] no '{cfg['comet_key']}' values found in summary; "
                  f"{model} rows will have null comet.", file=sys.stderr)
        n_rows, missing = write_model_file(
            tr_path, comet_scores, cfg["label"], args.source_lang, out_path)
        print(f"{cfg['label']}: wrote {n_rows} rows -> {out_path}")
        if missing:
            print(f"  [note] no comet score for languages: "
                  f"{', '.join(sorted(missing))} (comet left null)",
                  file=sys.stderr)


if __name__ == "__main__":
    main()