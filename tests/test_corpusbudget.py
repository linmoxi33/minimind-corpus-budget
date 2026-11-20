"""Tests for minimind-corpus-budget.  stdlib only: python3 -m unittest discover -s tests"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from corpusbudget.bpe import Tokenizer  # noqa: E402
from corpusbudget.probe import Probe, generate_labels, render_conversations  # noqa: E402

TOKENIZER = os.path.join(ROOT, "corpusbudget", "tokenizer.json")
BIN = os.path.join(ROOT, "bin", "minimind-corpus-budget")
SFT_IN = os.path.join(ROOT, "examples", "in", "sft_demo.jsonl")
PRE_IN = os.path.join(ROOT, "examples", "in", "pretrain_demo.jsonl")


def run(args, stdin=None):
    return subprocess.run(
        [sys.executable, BIN] + args,
        capture_output=True,
        text=True,
        input=stdin,
        cwd=ROOT,
    )


class TokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = Tokenizer(TOKENIZER)

    def test_vocab_matches_shipped_size(self):
        self.assertEqual(len(self.tok.vocab), 6400)

    def test_roundtrip_english(self):
        text = "MiniMind trains a 64M model in two hours."
        self.assertEqual(self.tok.decode(self.tok.encode(text)), text)

    def test_roundtrip_chinese_punctuation(self):
        text = "你好，世界！这是一次字节级往返测试。"
        self.assertEqual(self.tok.decode(self.tok.encode(text)), text)

    def test_roundtrip_whitespace_and_code(self):
        text = "  def f(x):\n\treturn x + 1\n\n"
        self.assertEqual(self.tok.decode(self.tok.encode(text)), text)

    def test_special_tokens_are_single_ids(self):
        ids = self.tok.encode("<|im_start|>assistant\n")
        self.assertEqual(ids[0], self.tok.vocab["<|im_start|>"])
        self.assertEqual(ids[-1], self.tok.encode("\n")[-1])

    def test_empty_text(self):
        self.assertEqual(self.tok.encode(""), [])


class LabelMaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.probe = Probe(TOKENIZER, 340)

    def test_labels_mask_prompt_and_keep_answer(self):
        conv = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello there"},
        ]
        ids = self.probe.tok.encode(render_conversations(conv))[:340]
        labels = generate_labels(ids, self.probe.bos_id, self.probe.eos_id, 340)
        self.assertEqual(labels[0], -100)
        supervised = [t for t in labels if t != -100]
        self.assertGreater(len(supervised), 0)

    def test_overflowing_prefix_yields_all_minus_100(self):
        conv = [
            {"role": "system", "content": "长" * 2000},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
        row = self.probe.probe_sft(conv)
        self.assertEqual(row["kind"], "dead")
        self.assertEqual(row["supervised_tokens"], 0)

    def test_long_answer_is_answer_cut_not_dead(self):
        conv = [
            {"role": "user", "content": "summarise"},
            {"role": "assistant", "content": "很长" * 900},
        ]
        row = self.probe.probe_sft(conv)
        self.assertEqual(row["kind"], "answer_cut")
        self.assertGreater(row["supervised_tokens"], 0)

    def test_pretrain_truncation_is_reported(self):
        row = self.probe.probe_pretrain("字" * 3000)
        self.assertTrue(row["truncated"])
        self.assertEqual(row["kept_tokens"], 340)


class CliBehaviourTests(unittest.TestCase):
    def test_sft_demo_json_is_parseable_and_matches_snapshot(self):
        res = run([SFT_IN, "--max-seq-len", "340", "--json"])
        self.assertEqual(res.returncode, 1)
        report = json.loads(res.stdout)
        snapshot = json.load(open(os.path.join(ROOT, "examples", "out", "sft_demo.json"), encoding="utf-8"))
        for key in ("samples", "tokens_total", "dead_samples", "answer_cut_samples",
                    "truncated_samples", "status", "supervised_tokens"):
            self.assertEqual(report[key], snapshot[key], key)
        self.assertEqual(report["dead_samples"], 2)

    def test_clean_corpus_exits_zero(self):
        res = run([PRE_IN, "--max-seq-len", "340", "--json"])
        self.assertEqual(res.returncode, 0)
        self.assertEqual(json.loads(res.stdout)["status"], "clean")

    def test_small_window_creates_dead_samples(self):
        res = run([SFT_IN, "--max-seq-len", "64", "--json"])
        self.assertEqual(res.returncode, 1)
        self.assertGreaterEqual(json.loads(res.stdout)["dead_samples"], 4)

    def test_missing_file_is_usage_error(self):
        res = run(["/tmp/definitely-not-here.jsonl"])
        self.assertEqual(res.returncode, 2)
        self.assertIn("error:", res.stderr)

    def test_bad_json_line_is_usage_error(self):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
            fh.write("{not json}\n")
            path = fh.name
        try:
            res = run([path])
            self.assertEqual(res.returncode, 2)
            self.assertIn("invalid JSON", res.stderr)
        finally:
            os.unlink(path)

    def test_empty_corpus_is_usage_error(self):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
            path = fh.name
        try:
            self.assertEqual(run([path]).returncode, 2)
        finally:
            os.unlink(path)

    def test_human_output_names_the_verdict(self):
        res = run([SFT_IN, "--max-seq-len", "340"])
        self.assertEqual(res.returncode, 1)
        self.assertIn("dead (all -100)", res.stdout)
        self.assertIn("verdict: loss", res.stdout)


if __name__ == "__main__":
    unittest.main()
