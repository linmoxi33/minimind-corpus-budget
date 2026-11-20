"""Byte-level BPE reader for MiniMind's shipped tokenizer.json.

Zero third-party dependencies.  Only the `model` (vocab + merges) and
`added_tokens` sections of a HuggingFace `tokenizers` JSON file are used,
which is all MiniMind's 6 400-token tokenizer needs.

The implementation follows the byte-level BPE contract used by
`tokenizers`' ByteLevel pre-tokenizer:

  * text is split into pre-tokens (letters / digits / punctuation / spaces),
  * every pre-token is mapped to its byte-level unicode string,
  * merges from the file are applied in rank order.

`encode()` is verified lossless: `decode(encode(text)) == text` for every
sample in `tests/`.  Special tokens listed in `added_tokens` are matched
literally before pre-tokenisation, exactly like `tokenizers` does.
"""

import json
import unicodedata

__all__ = ["Tokenizer", "bytes_to_unicode"]

# --- GPT-2 / tokenizers byte <-> unicode table -----------------------------


def _bytes_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


bytes_to_unicode = _bytes_to_unicode


def _is_letter(ch):
    return unicodedata.category(ch)[0] in ("L", "M")


def _is_number(ch):
    return unicodedata.category(ch)[0] == "N"


def _is_space(ch):
    return ch.isspace()


class Tokenizer:
    """Minimal, dependency-free byte-level BPE tokenizer."""

    def __init__(self, path):
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        model = data["model"]
        if model.get("type") != "BPE":
            raise ValueError("unsupported tokenizer model type: %r" % model.get("type"))
        self.vocab = {tok: int(idx) for tok, idx in model["vocab"].items()}
        merges = model.get("merges") or []
        # `merges` is either [["a","b"], ...] or ["a b", ...] depending on version.
        pairs = []
        for m in merges:
            pairs.append(tuple(m.split(" ")) if isinstance(m, str) else (m[0], m[1]))
        self.ranks = {p: i for i, p in enumerate(pairs)}
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.unk_id = self.vocab.get(model.get("unk_token") or "", None)
        self.id_to_token = {idx: tok for tok, idx in self.vocab.items()}

        self.special = {}
        for tok in data.get("added_tokens", []):
            if tok.get("special"):
                self.special[tok["content"]] = int(tok["id"])
        self._special_pat = None
        if self.special:
            self._special_pat = sorted(self.special, key=len, reverse=True)
        self._cache = {}

    # -- pre-tokenisation --------------------------------------------------

    def _pretokenize(self, text):
        out = []
        i, n = 0, len(text)
        while i < n:
            ch = text[i]
            if _is_space(ch):
                j = i
                while j < n and _is_space(text[j]):
                    j += 1
                run = text[i:j]
                if "\n" in run or "\r" in run:
                    # keep everything up to and including the last newline
                    k = max(run.rfind("\n"), run.rfind("\r"))
                    out.append(run[: k + 1])
                    rest = run[k + 1 :]
                    if rest:
                        out.append(rest)
                elif j < n:
                    # trailing space belongs to the next word (`\s+(?!\S)`)
                    out.append(run[:-1] if len(run) > 1 else run)
                    if len(run) > 1:
                        out.append(" ")
                else:
                    out.append(run)
                i = j
                continue
            if _is_letter(ch):
                j = i
                while j < n and _is_letter(text[j]):
                    j += 1
                out.append(text[i:j])
                i = j
                continue
            if _is_number(ch):
                out.append(ch)  # Qwen-style pre-tokenizer splits digits
                i += 1
                continue
            # punctuation / symbols / anything else, with one optional
            # leading space already handled above
            j = i
            while j < n and not (_is_space(text[j]) or _is_letter(text[j]) or _is_number(text[j])):
                j += 1
            out.append(text[i:j])
            i = j
        return out

    def _encode_pretoken(self, piece):
        cached = self._cache.get(piece)
        if cached is not None:
            return cached
        raw = piece.encode("utf-8")
        symbols = [self.byte_encoder[b] for b in raw]
        if len(symbols) > 1:
            ranks = self.ranks
            while len(symbols) > 1:
                best, best_rank = None, None
                for a, b in zip(symbols, symbols[1:]):
                    r = ranks.get((a, b))
                    if r is not None and (best_rank is None or r < best_rank):
                        best, best_rank = (a, b), r
                if best is None:
                    break
                merged, out = [], []
                k = 0
                while k < len(symbols):
                    if k + 1 < len(symbols) and symbols[k] == best[0] and symbols[k + 1] == best[1]:
                        out.append(symbols[k] + symbols[k + 1])
                        k += 2
                    else:
                        out.append(symbols[k])
                        k += 1
                symbols = out
        ids = []
        for s in symbols:
            idx = self.vocab.get(s)
            if idx is None:
                if self.unk_id is None:
                    raise ValueError("token not in vocab: %r" % s)
                idx = self.unk_id
            ids.append(idx)
        self._cache[piece] = ids
        return ids

    # -- public API --------------------------------------------------------

    def encode(self, text, add_special_tokens=False):
        if not text:
            return []
        ids = []
        if self._special_pat:
            i = 0
            while i < len(text):
                hit, hit_at = None, -1
                for sp in self._special_pat:
                    at = text.find(sp, i)
                    if at != -1 and (hit_at == -1 or at < hit_at):
                        hit, hit_at = sp, at
                if hit is None:
                    for piece in self._pretokenize(text[i:]):
                        ids.extend(self._encode_pretoken(piece))
                    break
                if hit_at > i:
                    for piece in self._pretokenize(text[i:hit_at]):
                        ids.extend(self._encode_pretoken(piece))
                ids.append(self.special[hit])
                i = hit_at + len(hit)
        else:
            for piece in self._pretokenize(text):
                ids.extend(self._encode_pretoken(piece))
        return ids

    def decode(self, ids):
        """Inverse of encode(): concatenate token strings, then un-byte-map."""
        rev_special = {v: k for k, v in self.special.items()}
        buf = bytearray()
        out = []

        def flush():
            if buf:
                out.append(buf.decode("utf-8", "replace"))
                del buf[:]

        for idx in ids:
            piece = rev_special.get(idx)
            if piece is None:
                piece = self.id_to_token.get(idx)
            if piece is None:
                raise ValueError("unknown token id: %r" % idx)
            if idx in rev_special:
                flush()
                out.append(piece)
                continue
            for ch in piece:
                b = self.byte_decoder.get(ch)
                if b is None:
                    flush()
                    out.append(ch)
                else:
                    buf.append(b)
        flush()
        return "".join(out)
