"""Command line entry point for minimind-corpus-budget."""

import argparse
import json
import os
import sys

from .probe import Probe

DEFAULT_TOKENIZER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokenizer.json")

EXIT_OK, EXIT_FOUND, EXIT_USAGE = 0, 1, 2


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("%s:%d: invalid JSON (%s)" % (path, lineno, exc.msg))
            if not isinstance(obj, dict):
                raise ValueError("%s:%d: expected a JSON object" % (path, lineno))
            records.append(obj)
    if not records:
        raise ValueError("%s: no samples found" % path)
    return records


def build_parser():
    p = argparse.ArgumentParser(
        prog="minimind-corpus-budget",
        description="Predict MiniMind SFT/pretrain supervision loss before you start training.",
    )
    p.add_argument("corpus", help="JSONL file (conversations or text records)")
    p.add_argument("--max-seq-len", type=int, default=340,
                   help="the --max_seq_len you will train with (default: 340, MiniMind distill default)")
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER,
                   help="path to MiniMind model/tokenizer.json (default: bundled copy)")
    p.add_argument("--mode", choices=["auto", "sft", "pretrain"], default="auto",
                   help="record format; auto = 'conversations' -> sft, else pretrain")
    p.add_argument("--json", action="store_true", help="emit a machine-readable report")
    p.add_argument("--samples", type=int, default=5,
                   help="how many offending samples to include in the report (default 5)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.max_seq_len < 8:
        print("error: --max-seq-len must be >= 8", file=sys.stderr)
        return EXIT_USAGE
    if not os.path.isfile(args.tokenizer):
        print("error: tokenizer not found: %s" % args.tokenizer, file=sys.stderr)
        return EXIT_USAGE
    try:
        records = load_jsonl(args.corpus)
        probe = Probe(args.tokenizer, args.max_seq_len)
        report = probe.run(records, mode=args.mode, sample_limit=max(0, args.samples))
    except (OSError, ValueError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return EXIT_USAGE

    report["corpus"] = args.corpus
    report["tokenizer"] = os.path.basename(args.tokenizer)

    if args.json:
        json.dump(report, sys.stdout, ensure_ascii=False, sort_keys=True, indent=2)
        sys.stdout.write("\n")
    else:
        print("corpus            : %s" % args.corpus)
        print("max_seq_len       : %d" % report["max_seq_len"])
        print("samples           : %d" % report["samples"])
        print("tokens / kept     : %d / %d (waste %d)"
              % (report["tokens_total"], report["tokens_kept"], report["token_waste"]))
        print("truncated samples : %d (%.2f%%)"
              % (report["truncated_samples"], 100 * report["truncated_ratio"]))
        print("answer cut        : %d" % report["answer_cut_samples"])
        print("dead (all -100)   : %d (%.2f%%)  -> %d wasted forward-pass tokens"
              % (report["dead_samples"], 100 * report["dead_ratio"],
                 report["dead_supervision_tokens"]))
        print("supervision ratio : %.4f" % report["supervision_ratio"])
        if report["problems"]:
            print("\noffending samples:")
            for s in report["problems"]:
                print("  #%d %-11s tokens=%d supervised=%d  %r"
                      % (s["index"], s["kind"], s["tokens"], s["supervised_tokens"],
                         s["render_head"][:70]))
        print("\nverdict: %s" % report["status"])
    return EXIT_FOUND if report["status"] == "loss" else EXIT_OK
