"""Pre-flight supervision budget emulator for MiniMind SFT / pretrain corpora.

It reproduces -- on plain CPU, without torch, without a GPU -- what
`dataset/lm_dataset.py` will do to every sample once training starts:

  * ChatML rendering of `conversations`,
  * right-side truncation to `max_seq_len`,
  * the exact assistant-marker scan that `SFTDataset.generate_labels` uses.

The output is the supervision budget: how many samples end up with an
all ``-100`` label row (trained on but learning nothing), how many keep a
truncated answer, and how many tokens of supervision survive.
"""

import json

from .bpe import Tokenizer

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

TOOLS_HEADER = (
    "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>"
)
TOOLS_FOOTER = (
    "\n</tools>\n\nFor each function call, return a json object with function name and "
    'arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n'
    '{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
)


def render_conversations(conversations):
    """Render MiniMind's ChatML template (the non-generation-prompt branch)."""
    msgs = [dict(m) for m in conversations]
    if not msgs:
        return ""
    tools = None
    if msgs[0].get("role") == "system" and msgs[0].get("tools"):
        tools = msgs[0]["tools"]
        if isinstance(tools, str):
            tools = json.loads(tools)
    parts = []
    if tools:
        parts.append(IM_START + "system\n")
        if msgs[0].get("role") == "system":
            parts.append(str(msgs[0].get("content") or "") + "\n\n")
        parts.append(TOOLS_HEADER)
        for tool in tools:
            parts.append("\n" + json.dumps(tool, ensure_ascii=False))
        parts.append(TOOLS_FOOTER + IM_END + "\n")
    elif msgs[0].get("role") == "system":
        parts.append(IM_START + "system\n" + str(msgs[0].get("content") or "") + IM_END + "\n")

    for msg in msgs:
        role = msg.get("role")
        content = msg.get("content")
        content = "" if not isinstance(content, str) else content
        if role == "system" and msgs and msg is msgs[0]:
            continue
        if role == "assistant":
            reasoning = msg.get("reasoning_content")
            if isinstance(reasoning, str):
                reasoning = reasoning.strip("\n")
            elif "</think>" in content:
                reasoning = content.split("</think>")[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
                content = content.split("</think>")[-1].lstrip("\n")
            else:
                reasoning = ""
            body = IM_START + "assistant\n<think>\n" + reasoning + "\n</think>\n\n" + content.lstrip("\n")
            for call in msg.get("tool_calls") or []:
                fn = call.get("function", call)
                args = fn.get("arguments", {})
                if not isinstance(args, str):
                    args = json.dumps(args, ensure_ascii=False)
                body += '\n<tool_call>\n{"name": "' + str(fn.get("name", "")) + '", "arguments": ' + args + "}\n</tool_call>"
            parts.append(body + IM_END + "\n")
        elif role == "tool":
            parts.append(IM_START + "tool\n" + content + IM_END + "\n")
        elif role == "user":
            parts.append(IM_START + "user\n" + content + IM_END + "\n")
        else:  # system or unknown role appearing later in the thread
            name = role if isinstance(role, str) else "system"
            parts.append(IM_START + name + "\n" + content + IM_END + "\n")
    return "".join(parts)


def generate_labels(input_ids, bos_id, eos_id, max_length):
    """Line-for-line port of SFTDataset.generate_labels (no torch)."""
    labels = [-100] * len(input_ids)
    i = 0
    nb, ne = len(bos_id), len(eos_id)
    while i < len(input_ids):
        if input_ids[i : i + nb] == bos_id:
            start = i + nb
            end = start
            while end < len(input_ids):
                if input_ids[end : end + ne] == eos_id:
                    break
                end += 1
            while start < min(end + ne, max_length):
                labels[start] = input_ids[start]
                start += 1
            i = end + ne if end < len(input_ids) else len(input_ids)
        else:
            i += 1
    return labels


def _classify(labels, ids, bos_id, eos_id):
    supervised = sum(1 for x in labels if x != -100)
    if supervised == 0:
        return "dead", 0, 0
    # did the answer get cut by the window?
    at = -1
    for i in range(len(ids) - len(bos_id) + 1):
        if ids[i : i + len(bos_id)] == bos_id:
            at = i
            break
    closed = False
    if at >= 0:
        for i in range(at + len(bos_id), len(ids) - len(eos_id) + 1):
            if ids[i : i + len(eos_id)] == eos_id:
                closed = True
                break
    kind = "ok" if closed else "answer_cut"
    return kind, supervised, len(ids) - supervised


class Probe:
    def __init__(self, tokenizer_path, max_seq_len=340):
        self.tok = Tokenizer(tokenizer_path)
        self.max_seq_len = int(max_seq_len)
        self.bos_id = self.tok.encode(IM_START + "assistant\n")
        self.eos_id = self.tok.encode(IM_END + "\n")
        self.pad_id = 0

    # -- sft ---------------------------------------------------------------

    def probe_sft(self, conversations):
        prompt = render_conversations(conversations)
        full = self.tok.encode(prompt)
        ids = full[: self.max_seq_len]
        ids = ids + [self.pad_id] * (self.max_seq_len - len(ids))
        labels = generate_labels(ids, self.bos_id, self.eos_id, self.max_seq_len)
        kind, sup, masked = _classify(labels, ids, self.bos_id, self.eos_id)
        return {
            "kind": kind,
            "tokens": len(full),
            "kept_tokens": min(len(full), self.max_seq_len),
            "truncated": len(full) > self.max_seq_len,
            "supervised_tokens": sup,
            "masked_tokens": masked,
            "prompt": prompt,
            "ids": ids,
        }

    # -- pretrain ----------------------------------------------------------

    def probe_pretrain(self, text):
        full = self.tok.encode(str(text))
        kept = full[: self.max_seq_len - 2]
        padding = self.max_seq_len - (len(kept) + 2)
        return {
            "kind": "ok" if len(full) <= self.max_seq_len - 2 else "truncated_text",
            "tokens": len(full),
            "kept_tokens": len(kept) + 2,
            "truncated": len(full) > self.max_seq_len - 2,
            "supervised_tokens": len(kept) + 2,
            "masked_tokens": padding,
            "prompt": str(text),
            "ids": ([self.pad_id] + kept + [self.pad_id]),
        }

    # -- driver ------------------------------------------------------------

    def run(self, records, mode="auto", sample_limit=5):
        rows, problems = [], []
        for idx, rec in enumerate(records):
            fmt = mode
            if fmt == "auto":
                fmt = "sft" if "conversations" in rec else "pretrain"
            if fmt == "sft":
                if "conversations" not in rec:
                    raise ValueError("record %d has no 'conversations'" % idx)
                row = self.probe_sft(rec["conversations"])
            else:
                if "text" not in rec:
                    raise ValueError("record %d has no 'text'" % idx)
                row = self.probe_pretrain(rec["text"])
            row["index"] = idx
            rows.append(row)
            if row["kind"] != "ok":
                problems.append(row)

        total_tokens = sum(r["tokens"] for r in rows)
        kept_tokens = sum(r["kept_tokens"] for r in rows)
        sup = sum(r["supervised_tokens"] for r in rows)
        dead = [r for r in rows if r["kind"] == "dead"]
        cut = [r for r in rows if r["kind"] == "answer_cut"]
        trunc = [r for r in rows if r["truncated"]]

        def _sample(r):
            return {
                "index": r["index"],
                "kind": r["kind"],
                "tokens": r["tokens"],
                "max_seq_len": self.max_seq_len,
                "supervised_tokens": r["supervised_tokens"],
                "render_head": r["prompt"][:160],
            }

        return {
            "mode": mode,
            "max_seq_len": self.max_seq_len,
            "samples": len(rows),
            "tokens_total": total_tokens,
            "tokens_kept": kept_tokens,
            "token_waste": total_tokens - kept_tokens,
            "supervised_tokens": sup,
            "supervision_ratio": round(sup / kept_tokens, 6) if kept_tokens else 0.0,
            "truncated_samples": len(trunc),
            "truncated_ratio": round(len(trunc) / len(rows), 6) if rows else 0.0,
            "answer_cut_samples": len(cut),
            "dead_samples": len(dead),
            "dead_ratio": round(len(dead) / len(rows), 6) if rows else 0.0,
            "dead_supervision_tokens": sum(r["kept_tokens"] for r in dead),
            "status": "loss" if (dead or cut) else "clean",
            "problems": [_sample(r) for r in problems[:sample_limit]],
            "problem_count": len(problems),
        }
