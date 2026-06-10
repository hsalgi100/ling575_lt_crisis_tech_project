#!/usr/bin/env python3
"""
Translation + Evaluation: Gemma4 4B vs Qwen3.5 9B — scored with COMET

WHAT THIS DOES
  For every record in the input JSONL(s):
    1. Pulls out the English source (source.eng).
    2. Translates that English into EVERY language in the LANG_NAMES table
       below (all codes except "en") — with both Gemma and Qwen.
    3. Where the record carries a "gold" translation for a language, scores
       both models' output against that gold with COMET.
    4. Emits one JSON record per input id containing:
         - the original info (id, scenario, source)
         - the full list of model translations (per language, both models)
         - the evaluation result (per language that had a gold, COMET + winner)

  Languages in the table that have NO gold for a given id are still translated
  and appear under "translations", but they cannot be COMET-scored (COMET is
  reference-based), so they are omitted from that record's "evaluation".

SETUP (run once):
  pip install llama-cpp-python unbabel-comet

JSONL format expected (one object per line):
  {
    "id": "kc-911_outage-000",
    "scenario": "911_outage",
    "source": {"eng": "English text"},
    "translations": {"am": "...gold...", "ko": "...gold...", ...}   # optional
  }

MEMORY --> 16 GB Mac:
"""

import os
import sys
import gc
import glob
import json
import hashlib
import argparse
from collections import defaultdict
from pathlib import Path
from llama_cpp import Llama

DEFAULT_GEMMA_PATH = "scripts/models/google_gemma-4-E4B-it-Q4_K_M.gguf"
DEFAULT_QWEN_PATH  = "scripts/models/Qwen_qwen3.5-9B-Q4_K_M.gguf"
DEFAULT_DATA_DIR   = "/data/kc"
DEFAULT_OUTPUT     = "translation_results.json"

# Inference config

N_GPU_LAYERS   = -1     
N_CTX          = 1024   
MAX_NEW_TOKENS = 256

LANG_NAMES = {
    "am": "Amharic",
    "km": "Khmer",
    "ko": "Korean",
    "lo": "Lao",
    "om": "Oromo",
    "ru": "Russian",
    "zh": "Traditional Chinese",
    "es": "Spanish",
    "tl": "Filipino (Tagalog)",
    "th": "Thai",
    "vi": "Vietnamese",
    "so": "Somali",
    "uk": "Ukrainian",
    "pa": "Punjabi",
    "fa": "Persian (Farsi)",
    "ja": "Japanese",
    "en": "English",
}

# Everything we translate into (the whole table minus the source language).
TARGET_LANGS = [code for code in LANG_NAMES if code != "en"]

def lang_name(code: str) -> str:
    return LANG_NAMES.get(code, code)


COMET_MODEL = "Unbabel/wmt22-comet-da"


def system_prompt(target_lang_code: str) -> str:
    name = lang_name(target_lang_code)
    return f"""\
# Role
You are a translation component within a translation module for Wireless Emergency Alerts (WEA). You translate one short alert at a time. You do not converse, explain, or ask questions—you return only the translated text.

# Inputs
1. **Source language**: English
2. **Target language**: {name}
3. **Source text**: the WEA alert text to be translated.

# Task
Translate the source text from English into {name}.

# Requirements
- **Accuracy over fluency**: Preserve every safety-critical detail exactly—hazard type, affected locations, times and dates, and protective-action instructions (e.g., evacuate, shelter in place, boil water, move to higher ground).
- **No alteration of meaning**: Do not add, omit, soften, or reinterpret any information. Keep imperative instructions imperative; do not hedge directive language.
- **Proper nouns**: Keep place names, road/route designations, and agency names as they appear in the source, unless a standard, official target-language form exists.
- **Brevity**: Keep the translation as concise as the source. Match its urgent, directive tone.
- **Numerals and units**: Preserve all numbers, units, and time formats; convert only when the target-language convention requires it, without changing the value.

# Output
- Return **only** the translated alert text.
- No preamble, labels, notes, explanations, quotation marks, or trailing text.
- If the source text is empty or untranslatable, return it unchanged."""

def gemma_prompt(source_text: str, target_lang_code: str) -> str:
    sp = system_prompt(target_lang_code)
    return (
        f"<start_of_turn>system\n{sp}<end_of_turn>\n"
        f"<start_of_turn>user\n{source_text}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )

def qwen_prompt(source_text: str, target_lang_code: str) -> str:
    sp = system_prompt(target_lang_code)
    return (
        f"<|im_start|>system\n{sp}<|im_end|>\n"
        f"<|im_start|>user\n{source_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
    )


# Model loading / unloading
def load_llm(model_path: str, label: str) -> Llama:
    print(f"Loading {label}: {os.path.basename(model_path)} ...")
    llm = Llama(model_path=model_path, n_gpu_layers=N_GPU_LAYERS, n_ctx=N_CTX, verbose=False)
    print(f"  {label} ready\n")
    return llm

def free_llm(llm, label: str = "") -> None:
    """Release a llama.cpp model and force its native buffers to be freed."""
    try:
        llm.close()          # frees the underlying llama_context/model
    except AttributeError:
        pass
    del llm
    gc.collect()
    if label:
        print(f"  {label} freed\n")


# Translation
def translate_gemma(llm: Llama, text: str, lang: str) -> str:
    out = llm(
        gemma_prompt(text, lang),
        max_tokens=MAX_NEW_TOKENS,
        temperature=1.0, top_p=0.95, top_k=64,
        stop=["<end_of_turn>", "<start_of_turn>"],
        echo=False,
    )
    return out["choices"][0]["text"].strip()

def translate_qwen(llm: Llama, text: str, lang: str) -> str:
    out = llm(
        qwen_prompt(text, lang),
        max_tokens=MAX_NEW_TOKENS,
        temperature=0.7, top_p=0.8, top_k=20,
        min_p=0.0, presence_penalty=1.5, repeat_penalty=1.0,
        stop=["<|im_end|>", "<|im_start|>", "<|endoftext|>"],
        echo=False,
    )
    raw = out["choices"][0]["text"]
    if "</think>" in raw:
        _, _, raw = raw.partition("</think>")
    return raw.strip()


# COMET
def load_comet():
    from comet import download_model, load_from_checkpoint
    print(f"Loading COMET ({COMET_MODEL}) ...")
    model = load_from_checkpoint(download_model(COMET_MODEL))
    print("  COMET ready\n")
    return model

def comet_score(comet_model, sources, hypotheses, references):
    data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(sources, hypotheses, references)]
    out = comet_model.predict(data, batch_size=8, gpus=0)
    return out.scores, out.system_score


# Data loading
def load_jsonl(path: str):
    """
    Returns a list of record dicts

    Every record with a usable English source is kept (regardless of whether it
    has any gold translations), because we translate into the full table either
    way. "en" entries and empty values are stripped from the gold dict.
    """
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            src = obj.get("source", {}).get("eng")
            if not src:
                continue
            gold = obj.get("translations", {}) or {}
            gold = {k: v for k, v in gold.items() if k != "en" and v}
            records.append({
                "id": obj.get("id"),
                "scenario": obj.get("scenario"),
                "source_eng": src,
                "gold": gold,
            })
    return records



# Checkpointing / resume
# Stable keys so checkpoints survive across runs regardless of record order.
def rec_key(rec) -> str:
    """A stable identifier for a record: file + id."""
    return f"{rec['file']}::{rec.get('id')}"

def pair_key(rec, lang: str) -> str:
    """A stable identifier for one (record, target-language) scoring pair."""
    return f"{rec_key(rec)}::{lang}"

def src_hash(text: str) -> str:
    """Short content hash of the source, used to invalidate stale checkpoints."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _append_jsonl(fh, obj) -> None:
    """Append one JSON line and force it to disk"""
    fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
    fh.flush()
    os.fsync(fh.fileno())


def load_translation_ckpt(path: str) -> dict:
    done = {}
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            k = o.get("key")
            if k is None:
                continue
            done[k] = {"src_hash": o.get("src_hash"), "tx": o.get("translations", {})}
    return done


def run_translation_phase(model_path, label, translate_fn, records, ckpt_path):
    """
    Translate every record into TARGET_LANGS with one model, resuming from
    `ckpt_path`. Returns a list parallel to `records`, each entry a {lang: hyp}
    dict. The model is only loaded if there is unfinished work.
    """
    done = load_translation_ckpt(ckpt_path)

    todo = []
    for rec in records:
        d = done.get(rec_key(rec))
        if d is None or d.get("src_hash") != src_hash(rec["source_eng"]):
            todo.append(rec)

    resumed = len(records) - len(todo)
    if resumed:
        print(f"  {label}: resuming — {resumed}/{len(records)} record(s) already checkpointed")

    if not todo:
        print(f"  {label}: nothing left to translate, skipping model load\n")
        return [done[rec_key(rec)]["tx"] for rec in records]

    llm = load_llm(model_path, label)
    fh = open(ckpt_path, "a", encoding="utf-8")
    try:
        for n, rec in enumerate(todo, 1):
            tx = {lang: translate_fn(llm, rec["source_eng"], lang) for lang in TARGET_LANGS}
            sh = src_hash(rec["source_eng"])
            done[rec_key(rec)] = {"src_hash": sh, "tx": tx}
            _append_jsonl(fh, {
                "key": rec_key(rec),
                "id": rec["id"],
                "file": rec["file"],
                "src_hash": sh,
                "translations": tx,
            })
            print(f"  {label} [{n}/{len(todo)}] {rec['id']}")
    finally:
        fh.close()
        free_llm(llm, label)

    return [done[rec_key(rec)]["tx"] for rec in records]


def load_score_ckpt(path: str):
    """Read a score checkpoint into (score_gemma, score_qwen), each {pair_key: float}."""
    sg, sq = {}, {}
    if not os.path.exists(path):
        return sg, sq
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            k, m, s = o.get("key"), o.get("model"), o.get("score")
            if k is None or s is None:
                continue
            if m == "gemma":
                sg[k] = float(s)
            elif m == "qwen":
                sq[k] = float(s)
    return sg, sq



# Output assembly + (atomic) write
def write_results(path, records, gemma_tx, qwen_tx, score_g, score_q, sys_g, sys_q):
    """
    Assemble the full output JSON from whatever is available so far and write it
    atomically (write to a .tmp, then os.replace). Safe to call after each phase:
    a translation that hasn't been produced yet shows up as null, and a language
    without a COMET score is simply absent from that record's "evaluation".
    Returns the output dict so callers can print from it.
    """
    out_records = []
    for i, rec in enumerate(records):
        g = gemma_tx[i] if i < len(gemma_tx) else {}
        q = qwen_tx[i] if i < len(qwen_tx) else {}
        translations = {
            lang: {
                "lang_name": lang_name(lang),
                "gemma4":    g.get(lang),
                "qwen35":    q.get(lang),
            }
            for lang in TARGET_LANGS
        }

        evaluation = {}
        for lang in rec["gold"]:
            k = pair_key(rec, lang)
            if k in score_g and k in score_q:
                gg, qq = score_g[k], score_q[k]
                evaluation[lang] = {
                    "lang_name":    lang_name(lang),
                    "gold":         rec["gold"][lang],
                    "gemma4_comet": gg,
                    "qwen35_comet": qq,
                    "winner":       "Gemma4" if gg > qq else "Qwen3.5",
                }

        out_records.append({
            "id":           rec["id"],
            "scenario":     rec["scenario"],
            "file":         rec["file"],
            "source":       {"eng": rec["source_eng"]},
            "translations": translations,
            "evaluation":   evaluation,
        })

    # Per-language aggregate over whatever has been scored so far
    lang_seg_g, lang_seg_q = defaultdict(list), defaultdict(list)
    for rec in records:
        for lang in rec["gold"]:
            k = pair_key(rec, lang)
            if k in score_g and k in score_q:
                lang_seg_g[lang].append(score_g[k])
                lang_seg_q[lang].append(score_q[k])

    by_language = []
    for lang in sorted(lang_seg_g):
        gmean = sum(lang_seg_g[lang]) / len(lang_seg_g[lang])
        qmean = sum(lang_seg_q[lang]) / len(lang_seg_q[lang])
        by_language.append({
            "lang":         lang,
            "lang_name":    lang_name(lang),
            "n":            len(lang_seg_g[lang]),
            "gemma4_comet": gmean,
            "qwen35_comet": qmean,
            "winner":       "Gemma4" if gmean > qmean else "Qwen3.5",
        })

    n_scored = sum(len(v) for v in lang_seg_g.values())

    overall_winner = None
    if sys_g is not None and sys_q is not None:
        overall_winner = "Gemma4" if sys_g > sys_q else "Qwen3.5"

    output = {
        "comet_model": COMET_MODEL,
        "target_languages": TARGET_LANGS,
        "summary": {
            "system_comet":   {"gemma4": sys_g, "qwen35": sys_q},
            "overall_winner": overall_winner,
            "n_records":      len(out_records),
            "n_scored_pairs": n_scored,
        },
        "by_language": by_language,
        "records": out_records,
    }

    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)   # atomic on the same filesystem; won't corrupt a prior checkpoint
    return output


# Main
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gemma",  default=DEFAULT_GEMMA_PATH)
    p.add_argument("--qwen",   default=DEFAULT_QWEN_PATH)
    p.add_argument("--data",   default=DEFAULT_DATA_DIR)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--fresh", action="store_true",
                   help="Delete any existing checkpoints before starting (no resume).")
    p.add_argument("--keep-checkpoints", action="store_true",
                   help="Keep the per-phase checkpoint files after a successful run.")
    return p.parse_args()


def main():
    args = parse_args()

    for label, path in [("--gemma", args.gemma), ("--qwen", args.qwen)]:
        if not path:
            print(f"Error: {label} path required. Pass as CLI arg or set the default at the top of this file.")
            sys.exit(1)
        if not os.path.exists(path):
            print(f"Error: file not found: {path}")
            sys.exit(1)

    jsonl_files = sorted(glob.glob(os.path.join(args.data, "*.jsonl")))
    if not jsonl_files:
        print(f"Error: no .jsonl files found in '{args.data}'")
        sys.exit(1)

    print(f"\nFound {len(jsonl_files)} JSONL file(s)")
    print(f"Translating into {len(TARGET_LANGS)} languages: {', '.join(TARGET_LANGS)}\n")

    # Phase 0: read every file once, flatten into one record list
    records = []   # each: dict with id/scenario/source_eng/gold/file
    for fpath in jsonl_files:
        fname = Path(fpath).stem
        recs = load_jsonl(fpath)
        if not recs:
            print(f"  Skipping {fname}: no usable rows")
            continue
        for r in recs:
            r["file"] = fname
        records.extend(recs)
        print(f"  Loaded {fname}: {len(recs)} record(s)")
    if not records:
        print("No records to process.")
        sys.exit(1)

    n_jobs = len(records) * len(TARGET_LANGS)
    print(f"\nTotal: {len(records)} records × {len(TARGET_LANGS)} languages "
          f"= {n_jobs} translations per model\n")

    # Warn if record keys aren't unique — resume relies on them being stable & unique.
    keyset = [rec_key(r) for r in records]
    if len(set(keyset)) != len(keyset):
        dupes = sorted({k for k in keyset if keyset.count(k) > 1})
        print(f"  WARNING: duplicate record keys detected ({len(dupes)}); "
              f"resume/scoring may be unreliable for: {', '.join(dupes[:5])}"
              f"{' ...' if len(dupes) > 5 else ''}\n")

    # Checkpoint files live next to the output.
    gemma_ckpt = args.output + ".gemma.partial.jsonl"
    qwen_ckpt  = args.output + ".qwen.partial.jsonl"
    score_ckpt = args.output + ".scores.partial.jsonl"
    ckpt_files = [gemma_ckpt, qwen_ckpt, score_ckpt]

    if args.fresh:
        removed = [p for p in ckpt_files if os.path.exists(p)]
        for p in removed:
            os.remove(p)
        if removed:
            print(f"  --fresh: removed {len(removed)} existing checkpoint file(s)\n")

    # Phase 1: Gemma translates everything (resumable), then is freed
    gemma_tx = run_translation_phase(args.gemma, "Gemma4 4B", translate_gemma, records, gemma_ckpt)
    write_results(args.output, records, gemma_tx, [], {}, {}, None, None)
    print(f"  Results JSON updated after Gemma phase -> {args.output}\n")

    # Phase 2: Qwen translates everything (resumable), then is freed
    qwen_tx = run_translation_phase(args.qwen, "Qwen3.5 9B", translate_qwen, records, qwen_ckpt)
    write_results(args.output, records, gemma_tx, qwen_tx, {}, {}, None, None)
    print(f"  Results JSON updated after Qwen phase -> {args.output}\n")

    # Phase 3: build scorable triples, then COMET-score them
    # We can only score (record, lang) pairs where a gold reference exists AND
    # the language is one we translated into.
    triples = []   # (pair_key, src, ref, mt_gemma, mt_qwen)
    for i, rec in enumerate(records):
        for lang, ref in rec["gold"].items():
            if lang not in TARGET_LANGS:
                continue
            triples.append((pair_key(rec, lang), rec["source_eng"], ref,
                            gemma_tx[i][lang], qwen_tx[i][lang]))

    score_g, score_q = load_score_ckpt(score_ckpt)
    sys_g = sys_q = None
    if triples:
        need_g = [t for t in triples if t[0] not in score_g]
        need_q = [t for t in triples if t[0] not in score_q]

        if need_g or need_q:
            comet = load_comet()
            scf = open(score_ckpt, "a", encoding="utf-8")
            try:
                if need_g:
                    print(f"Scoring {len(need_g)} Gemma pair(s) with COMET ...")
                    seg, _ = comet_score(comet, [t[1] for t in need_g],
                                         [t[3] for t in need_g], [t[2] for t in need_g])
                    for t, s in zip(need_g, seg):
                        score_g[t[0]] = float(s)
                        _append_jsonl(scf, {"model": "gemma", "key": t[0], "score": float(s)})
                else:
                    print("Gemma scores already checkpointed, skipping")

                if need_q:
                    print(f"Scoring {len(need_q)} Qwen pair(s) with COMET ...")
                    seg, _ = comet_score(comet, [t[1] for t in need_q],
                                         [t[4] for t in need_q], [t[2] for t in need_q])
                    for t, s in zip(need_q, seg):
                        score_q[t[0]] = float(s)
                        _append_jsonl(scf, {"model": "qwen", "key": t[0], "score": float(s)})
                else:
                    print("Qwen scores already checkpointed, skipping")
            finally:
                scf.close()
        else:
            print("All COMET scores already checkpointed, skipping COMET load")

        # System score = mean of segment scores over all scorable pairs.
        gvals = [score_g[t[0]] for t in triples if t[0] in score_g]
        qvals = [score_q[t[0]] for t in triples if t[0] in score_q]
        if gvals:
            sys_g = sum(gvals) / len(gvals)
        if qvals:
            sys_q = sum(qvals) / len(qvals)
        print()
    else:
        print("No gold translations found for any target language — "
              "translations will be produced but not scored.\n")

    # Final write (now with COMET scores)
    output = write_results(args.output, records, gemma_tx, qwen_tx,
                           score_g, score_q, sys_g, sys_q)
    by_language = output["by_language"]

    # Console report
    if by_language:
        print("\n" + "=" * 70)
        print("BY LANGUAGE (scored pairs only)")
        print("=" * 70)
        print(f"{'Language':<22} {'Code':>5} {'N':>5} {'Gemma4':>9} {'Qwen3.5':>9} {'Winner':>10}")
        print("-" * 70)
        for r in by_language:
            print(f"{r['lang_name']:<22} {r['lang']:>5} {r['n']:>5} "
                  f"{r['gemma4_comet']:>9.4f} {r['qwen35_comet']:>9.4f} {r['winner']:>10}")
        print("=" * 70)
        print(f"  System COMET — Gemma4: {sys_g:.4f}   Qwen3.5: {sys_q:.4f}")
        print(f"  Overall winner: {output['summary']['overall_winner']}")

    # Note any table languages that were translated but never had a gold to score against
    scored_langs = {r["lang"] for r in by_language}
    unscored = [l for l in TARGET_LANGS if l not in scored_langs]
    if unscored:
        print(f"\n  Translated but unscored (no gold in data): "
              f"{', '.join(unscored)}")

    print(f"\n  Saved: {args.output}")

    # Clean up checkpoints on a successful finish
    if args.keep_checkpoints:
        print("  --keep-checkpoints: leaving checkpoint files in place")
    else:
        removed = [p for p in ckpt_files if os.path.exists(p)]
        for p in removed:
            os.remove(p)
        if removed:
            print(f"  Cleaned up {len(removed)} checkpoint file(s)")


if __name__ == "__main__":
    main()