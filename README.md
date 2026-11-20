# minimind-corpus-budget

Know how much supervision your MiniMind SFT run will silently throw away — **before** you spend the 2 hours and the GPU.

Upstream: https://github.com/jingyaogong/minimind

## Why this exists

MiniMind is a from-scratch 64M-parameter LLM that people actually fine-tune on their own chat data.
Its `SFTDataset` renders ChatML, then right-truncates to `max_seq_len`, and only **after** truncation
does it scan for the `<|im_start|>assistant\n` marker to build the label row
([lm_dataset.py L114](https://github.com/jingyaogong/minimind/blob/master/dataset/lm_dataset.py#L114),
[L91-L112](https://github.com/jingyaogong/minimind/blob/master/dataset/lm_dataset.py#L91-L112)).
When the system prompt, tool schemas and user turn already fill the window, the marker is cut off, so
`labels` becomes **all `-100`**: the sample is still forwarded and back-propagated, contributes no
gradient signal, prints no warning and does not move the loss. Upstream hit this on its own data —
1.97% (17,821 / 905,718) of `sft_t2t_mini.jsonl` at `max_seq_len=340`, concentrated in tool samples —
and filed it as [issue #887](https://github.com/jingyaogong/minimind/issues/887). The issue states the
mechanism and a fix sketch, but nothing lets *your* corpus answer "how many of my rows are dead, and
which ones?" without a full training setup with torch + datasets.

This tool answers that in seconds, on CPU, offline. It is not a linter with a rule list: it is a
simulator that runs MiniMind's own tokenizer and its own label-scan algorithm over your JSONL and
reports the supervision budget you are about to buy.

## What it does / does not do

Does:

- byte-level BPE encoding with the **bundled upstream `model/tokenizer.json`** (6,400 tokens, Apache-2.0),
  verified lossless: `decode(encode(x)) == x`;
- renders the ChatML template MiniMind uses, including the `# Tools` block and `<think>` wrapping;
- ports `generate_labels()` line for line and classifies every row as `ok`, `answer_cut` (answer
  truncated mid-way) or `dead` (all `-100`, zero supervision);
- reports total tokens, tokens kept, token waste, supervision ratio, dead-sample ratio and the wasted
  forward-pass tokens, plus the first N offending rows;
- `--json` for CI, exit codes for gating.

Does not:

- run torch, download data, or read/write your checkpoints;
- emulate Jinja features it does not need (`add_generation_prompt`, custom templates);
- judge data *quality* (duplicates, toxicity, language mix) — only the truncation/supervision budget.

## Install

Python 3.8+, no third-party packages, no network.

```bash
git clone git@github-linmoxi33:linmoxi33/minimind-corpus-budget.git
cd minimind-corpus-budget
./bin/minimind-corpus-budget examples/in/sft_demo.jsonl --max-seq-len 340
```

Point it at your own file and at the tokenizer from your MiniMind checkout:

```bash
./bin/minimind-corpus-budget /data/my_sft.jsonl --max-seq-len 340 \
    --tokenizer /path/to/minimind/model/tokenizer.json
```

`--mode sft|pretrain|auto` (default `auto`: rows with `conversations` are SFT, rows with `text` are
pretrain). `--samples N` controls how many offending rows are echoed. `--json` prints the full report.

```console
$ ./bin/minimind-corpus-budget examples/in/sft_demo.jsonl --max-seq-len 340
corpus            : examples/in/sft_demo.jsonl
max_seq_len       : 340
samples           : 9
tokens / kept     : 4300 / 1887 (waste 2413)
truncated samples : 4 (44.44%)
answer cut        : 2
dead (all -100)   : 2 (22.22%)  -> 680 wasted forward-pass tokens
supervision ratio : 0.2825

offending samples:
  #1 dead        tokens=723 supervised=0  '<|im_start|>system\n你是一个知识丰富的AI…'
  #3 dead        tokens=1216 supervised=0  '<|im_start|>system\n你是 minimind…'
  #6 answer_cut  tokens=356 supervised=16 …
  #8 answer_cut  tokens=1478 supervised=316 …

verdict: loss
```

JSON mode (`examples/out/sft_demo.json`) is stable and snapshot-tested:

```bash
./bin/minimind-corpus-budget examples/in/sft_demo.jsonl --json \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['dead_samples'], d['status'])"
# 2 loss
```

Gate a data pipeline:

```bash
./bin/minimind-corpus-budget my_sft.jsonl --json > budget.json || echo "refusing to train"
```

| exit | meaning |
|---|---|
| 0 | no truncation damage: every row keeps supervision |
| 1 | at least one `dead` or `answer_cut` row was found |
| 2 | usage or input error (missing file, bad JSON line, empty corpus, bad flag) |

## Tests

```bash
python3 -m unittest discover -s tests -v   # 17 tests, 14 behavioural
```

## Limitations

- Token counts come from a dependency-free reimplementation of byte-level BPE. Merge application and
  special-token handling are exact, and round-trip is byte-exact on every test string, but the
  pre-tokenizer is a Unicode-category scanner rather than the exact PCRE used by `tokenizers`; for
  exotic scripts the boundary may differ by a token, so treat counts near the boundary (±2 tokens) as
  approximate. The `dead`/`answer_cut` verdict on a clearly-overflowing prefix is not affected.
- `pre_processing_chat` randomly injects a system prompt (20%) and `post_processing_chat` randomly
  strips empty `<think>` blocks; the simulator is deterministic and does not inject, so real runs can
  differ slightly (injection only makes prefixes longer, i.e. more dead rows).
- `train_full_sft.py`'s own system-prompt plumbing is outside the dataset class and is not simulated.

## 本轮创新

第一个把 MiniMind 的 truncation→label-scan 顺序当成可离线复算的「监督预算」来测量：上游 issue #887
只给了一次全量统计和修复思路，没有给任何人在自己数据上预演的能力；本项目用上游自带 tokenizer +
逐行移植的 `generate_labels` 在 CPU 上复现同一判定，并对每条受损样本给出 token 数与渲染前缀，
把「训练完才发现 loss 不动」变成训练前可断言的退出码。与既有 `-lint/-audit` 模板不同，这里没有规则
清单，只有对上游真实算法的一次可对照复算。

## License

MIT for this tool. `corpusbudget/tokenizer.json` is copied verbatim from
[jingyaogong/minimind](https://github.com/jingyaogong/minimind) (`model/tokenizer.json`), Apache-2.0,
© Jingyao Gong; see `LICENSE-minimind` and `NOTICE`.
