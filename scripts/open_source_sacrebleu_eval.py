#!/usr/bin/env python3
"""
sacreBLEU + chrF evaluation: Gemma4 4B vs Qwen3.5 9B

WHAT THIS DOES
  Reads the JSON produced by the COMET translation script (default
  "translation_results.json") and re-scores the SAME translations with
  reference-based string metrics from the sacreBLEU package:
      - BLEU   (corpus + per-segment)
      - chrF   (character F-score)
      - chrF++ (chrF with word bigrams)

WHERE THE DATA COMES FROM
  gold reference : records[i]["evaluation"][lang]["gold"]
  Gemma output   : records[i]["translations"][lang]["gemma4"]
  Qwen output    : records[i]["translations"][lang]["qwen35"]
"""

import os
import sys
import glob
import json
import argparse
from pathlib import Path
from collections import defaultdict

from sacrebleu.metrics import BLEU, CHRF


DEFAULT_INPUT  = "translation_results.json"
DEFAULT_OUTPUT = "sacrebleu_results.json"

# Single, consistent, dependency-light tokenizer for all 16 languages.
DEFAULT_TOKENIZER = "flores200"

PER_LANG_TOKENIZER: dict = {}

LANG_NAMES = {
    "am": "Amharic", "km": "Khmer", "ko": "Korean", "lo": "Lao",
    "om": "Oromo", "ru": "Russian", "zh": "Traditional Chinese", "es": "Spanish",
    "tl": "Filipino (Tagalog)", "th": "Thai", "vi": "Vietnamese", "so": "Somali",
    "uk": "Ukrainian", "pa": "Punjabi", "fa": "Persian (Farsi)", "ja": "Japanese",
    "en": "English",
}


def lang_name(code: str) -> str:
    return LANG_NAMES.get(code, code)


# Metric objects (cached per tokenizer so we don't rebuild them constantly)
_bleu_corpus: dict = {}
_bleu_sent: dict = {}
_chrf = CHRF()                 # chrF  (char n-grams, no word order)
_chrfpp = CHRF(word_order=2)   # chrF++ (adds word bigrams)


def bleu_for(tokenizer: str, sentence: bool) -> BLEU:
    """Return a (cached) BLEU metric for a tokenizer. effective_order=True for
    sentence-level BLEU so short segments aren't zeroed by missing higher n-grams."""
    cache = _bleu_sent if sentence else _bleu_corpus
    if tokenizer not in cache:
        cache[tokenizer] = BLEU(tokenize=tokenizer, effective_order=sentence)
    return cache[tokenizer]


def tokenizer_for(lang: str, forced: str | None) -> str:
    if forced:
        return forced
    return PER_LANG_TOKENIZER.get(lang, DEFAULT_TOKENIZER)


# Optional: gold from the original source JSONL (mirrors the COMET script's loader)
def load_gold_from_data(data_dir: str) -> dict:
    """Returns {(file_stem, id): {lang: gold}} from every *.jsonl in data_dir."""
    out: dict = {}
    for fpath in sorted(glob.glob(os.path.join(data_dir, "*.jsonl"))):
        fname = Path(fpath).stem
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                gold = obj.get("translations", {}) or {}
                gold = {k: v for k, v in gold.items() if k != "en" and v}
                if gold:
                    out[(fname, obj.get("id"))] = gold
    return out


def gold_for_record(rec: dict, extra_gold: dict) -> dict:
    """Union of gold refs available for a record: from the JSON's evaluation block
    plus (optionally) the original source data, keyed by (file, id)."""
    gold = {}
    for lang, ev in (rec.get("evaluation") or {}).items():
        if isinstance(ev, dict) and ev.get("gold"):
            gold[lang] = ev["gold"]
    if extra_gold:
        for lang, ref in extra_gold.get((rec.get("file"), rec.get("id")), {}).items():
            gold.setdefault(lang, ref)
    return gold


def hyp_for(rec: dict, lang: str, model_key: str):
    """model_key is 'gemma4' or 'qwen35' as stored by the translation script."""
    tx = (rec.get("translations") or {}).get(lang) or {}
    return tx.get(model_key)


# Scoring
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  default=DEFAULT_INPUT,
                    help="JSON produced by the COMET translation script.")
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    ap.add_argument("--tokenizer", default=None,
                    help=f"Force one BLEU tokenizer for all languages "
                         f"(default per-language: {DEFAULT_TOKENIZER}).")
    ap.add_argument("--data", default=None,
                    help="Optional source-jsonl dir to supplement gold references.")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: input not found: {args.input}")
        sys.exit(1)

    with open(args.input, encoding="utf-8") as f:
        results = json.load(f)

    records = results.get("records", [])
    if not records:
        print("Error: no 'records' in input JSON.")
        sys.exit(1)

    target_langs = results.get("target_languages") or [c for c in LANG_NAMES if c != "en"]
    extra_gold = load_gold_from_data(args.data) if args.data else {}

    print(f"\nLoaded {len(records)} record(s) from {args.input}")
    print(f"BLEU tokenizer: {args.tokenizer or (DEFAULT_TOKENIZER + ' (per-language overrides allowed)')}")
    if extra_gold:
        print(f"Supplementing gold from {args.data}: {len(extra_gold)} keyed record(s)")
    print()

    # Per-language parallel buffers for corpus-level scoring.
    by_lang_gemma = defaultdict(list)
    by_lang_qwen  = defaultdict(list)
    by_lang_ref   = defaultdict(list) 

    skipped = 0
    out_records = []

    for rec in records:
        gold = gold_for_record(rec, extra_gold)
        rec_eval = {}

        for lang, ref in gold.items():
            if lang not in target_langs:
                continue
            g_hyp = hyp_for(rec, lang, "gemma4")
            q_hyp = hyp_for(rec, lang, "qwen35")
            # Fair head-to-head needs gold + both hypotheses present.
            if not ref or g_hyp is None or q_hyp is None:
                skipped += 1
                continue

            tok = tokenizer_for(lang, args.tokenizer)
            sb = bleu_for(tok, sentence=True)

            g_bleu = sb.sentence_score(g_hyp, [ref]).score
            q_bleu = sb.sentence_score(q_hyp, [ref]).score
            g_chrf = _chrf.sentence_score(g_hyp, [ref]).score
            q_chrf = _chrf.sentence_score(q_hyp, [ref]).score
            g_chpp = _chrfpp.sentence_score(g_hyp, [ref]).score
            q_chpp = _chrfpp.sentence_score(q_hyp, [ref]).score

            rec_eval[lang] = {
                "lang_name":   lang_name(lang),
                "gold":        ref,
                "tokenizer":   tok,
                "gemma4":      {"sentence_bleu": g_bleu, "chrf": g_chrf, "chrf++": g_chpp},
                "qwen35":      {"sentence_bleu": q_bleu, "chrf": q_chrf, "chrf++": q_chpp},
                "winner_bleu": "Gemma4" if g_bleu > q_bleu else "Qwen3.5",
                "winner_chrf": "Gemma4" if g_chrf > q_chrf else "Qwen3.5",
            }

            by_lang_gemma[lang].append(g_hyp)
            by_lang_qwen[lang].append(q_hyp)
            by_lang_ref[lang].append(ref)

        out_records.append({
            "id":         rec.get("id"),
            "scenario":   rec.get("scenario"),
            "file":       rec.get("file"),
            "source":     rec.get("source", {}),
            "evaluation": rec_eval,
        })

    # Corpus-level scores per language
    by_language = []
    for lang in sorted(by_lang_ref):
        refs   = by_lang_ref[lang]
        g_hyps = by_lang_gemma[lang]
        q_hyps = by_lang_qwen[lang]
        tok = tokenizer_for(lang, args.tokenizer)
        cb = bleu_for(tok, sentence=False)

        g_bleu = cb.corpus_score(g_hyps, [refs]).score
        q_bleu = cb.corpus_score(q_hyps, [refs]).score
        g_chrf = _chrf.corpus_score(g_hyps, [refs]).score
        q_chrf = _chrf.corpus_score(q_hyps, [refs]).score
        g_chpp = _chrfpp.corpus_score(g_hyps, [refs]).score
        q_chpp = _chrfpp.corpus_score(q_hyps, [refs]).score

        by_language.append({
            "lang":        lang,
            "lang_name":   lang_name(lang),
            "n":           len(refs),
            "tokenizer":   tok,
            "gemma4":      {"bleu": g_bleu, "chrf": g_chrf, "chrf++": g_chpp},
            "qwen35":      {"bleu": q_bleu, "chrf": q_chrf, "chrf++": q_chpp},
            "winner_bleu": "Gemma4" if g_bleu > q_bleu else "Qwen3.5",
            "winner_chrf": "Gemma4" if g_chrf > q_chrf else "Qwen3.5",
        })

    # Headline = macro-average across languages (each language weighted equally),
    # the standard way to summarize a multilingual benchmark.
    def macro(metric_key, model_key):
        vals = [bl[model_key][metric_key] for bl in by_language]
        return sum(vals) / len(vals) if vals else None

    summary = {
        "n_records":      len(out_records),
        "n_scored_pairs": sum(bl["n"] for bl in by_language),
        "n_skipped":      skipped,
        "macro_avg_over_languages": {
            "bleu":   {"gemma4": macro("bleu", "gemma4"),   "qwen35": macro("bleu", "qwen35")},
            "chrf":   {"gemma4": macro("chrf", "gemma4"),   "qwen35": macro("chrf", "qwen35")},
            "chrf++": {"gemma4": macro("chrf++", "gemma4"), "qwen35": macro("chrf++", "qwen35")},
        },
    }
    if by_language:
        summary["overall_winner_bleu"] = (
            "Gemma4" if macro("bleu", "gemma4") > macro("bleu", "qwen35") else "Qwen3.5")
        summary["overall_winner_chrf"] = (
            "Gemma4" if macro("chrf", "gemma4") > macro("chrf", "qwen35") else "Qwen3.5")

    output = {
        "metrics":          ["bleu", "chrf", "chrf++"],
        "default_tokenizer": args.tokenizer or DEFAULT_TOKENIZER,
        "target_languages": target_langs,
        "source_file":      args.input,
        "summary":          summary,
        "by_language":      by_language,
        "records":          out_records,
    }

    tmp = args.output + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    os.replace(tmp, args.output)

    # Console report
    if by_language:
        print("=" * 86)
        print("BY LANGUAGE (corpus scores; chrF is the most reliable across scripts)")
        print("=" * 86)
        print(f"{'Language':<22} {'Code':>4} {'N':>4} "
              f"{'G-BLEU':>7} {'Q-BLEU':>7} {'G-chrF':>7} {'Q-chrF':>7} {'chrF win':>9}")
        print("-" * 86)
        for r in by_language:
            print(f"{r['lang_name']:<22} {r['lang']:>4} {r['n']:>4} "
                  f"{r['gemma4']['bleu']:>7.2f} {r['qwen35']['bleu']:>7.2f} "
                  f"{r['gemma4']['chrf']:>7.2f} {r['qwen35']['chrf']:>7.2f} "
                  f"{r['winner_chrf']:>9}")
        print("=" * 86)
        m = summary["macro_avg_over_languages"]
        print(f"  Macro-avg BLEU  — Gemma4: {m['bleu']['gemma4']:.2f}   Qwen3.5: {m['bleu']['qwen35']:.2f}")
        print(f"  Macro-avg chrF  — Gemma4: {m['chrf']['gemma4']:.2f}   Qwen3.5: {m['chrf']['qwen35']:.2f}")
        print(f"  Overall winner (BLEU): {summary.get('overall_winner_bleu')}")
        print(f"  Overall winner (chrF): {summary.get('overall_winner_chrf')}")
    if skipped:
        print(f"\n  Skipped {skipped} pair(s) missing gold or a model hypothesis.")
    print(f"\n  Saved: {args.output}")


if __name__ == "__main__":
    main()