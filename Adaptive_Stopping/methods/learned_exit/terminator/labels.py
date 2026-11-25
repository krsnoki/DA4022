"""Token labels for TERMINATOR training (step B).

A thinking token gets label 1 once the model's final answer has already
appeared in the thinking text, and 0 before that. The final answer is the
last \\boxed{...} in the answer section (after </think>).

The paper locates the first appearance with a second reasoning model. Here we
use string matching on a normalised view of the thinking text instead:

1. Build a normalised copy of the thinking text (whitespace, $, \\left,
   \\right, thin spaces removed; dfrac/tfrac -> frac), keeping a map back to
   original character offsets.
2. Search for the normalised answer. A match must not sit inside a larger
   number or word (boundary check).
3. If the answer string also appears in the problem statement (for example
   the answer is "3" and the problem mentions 3), a match only counts when it
   is boxed or follows "=", "is", "be", "get", "equals" or "answer".
4. The earliest valid match wins. Its last character is mapped to a token
   index; that token and every later thinking token are labelled 1.
Traces with no match are dropped from training.
"""

from __future__ import annotations

import bisect
import re
from typing import List, Optional, Sequence, Tuple

_DROP_SEQS = ["\\left", "\\right", "\\!", "\\,", "\\;", "\\:", "\\displaystyle"]
_REPLACE = [("\\dfrac", "\\frac"), ("\\tfrac", "\\frac")]
_CONTEXT_RE = re.compile(r"(=|\bis|\bbe|\bget|\bgives|\bequals|\banswer|boxed|:|\\approx)$", re.IGNORECASE)


def normalized_view(text: str) -> Tuple[str, List[int]]:
    """Return (normalised text, index map) where map[i] is the original offset
    of normalised character i."""
    out: List[str] = []
    idx: List[int] = []
    i = 0
    n = len(text)
    while i < n:
        matched = False
        for src, dst in _REPLACE:
            if text.startswith(src, i):
                for k, ch in enumerate(dst):
                    out.append(ch)
                    idx.append(i + min(k + 1, len(src) - 1) if k else i)
                i += len(src)
                matched = True
                break
        if matched:
            continue
        for seq in _DROP_SEQS:
            if text.startswith(seq, i) and not (i + len(seq) < n and text[i + len(seq)].isalpha()):
                i += len(seq)
                matched = True
                break
        if matched:
            continue
        ch = text[i]
        if ch.isspace() or ch == "$":
            i += 1
            continue
        # thousands separator: "27,648" -> "27648"
        if (ch == "," and i > 0 and text[i - 1].isdigit() and text[i + 1 : i + 4].isdigit()
                and len(text[i + 1 : i + 4]) == 3 and not text[i + 4 : i + 5].isdigit()):
            i += 1
            continue
        out.append(ch)
        idx.append(i)
        i += 1
    return "".join(out), idx


def normalize_for_match(ans: str) -> str:
    s, _ = normalized_view(ans)
    s = re.sub(r"\\(?:text|textbf|mathrm|mbox)\{([^{}]*)\}", r"\1", s)
    return s.rstrip(".")


def answer_forms(ans: str) -> List[str]:
    """Normalised answer plus a plain-text variant (\\frac{a}{b} -> a/b,
    \\sqrt{a} -> sqrt(a) is not attempted)."""
    forms = [ans]
    slash = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", ans)
    if slash != ans:
        forms.append(slash)
    return forms


def _short_number(ans: str) -> bool:
    return bool(re.fullmatch(r"-?\d{1,2}", ans))


def _boundary_ok(norm: str, start: int, end: int, ans: str, text: str = None,
                 idx: List[int] = None) -> bool:
    before = norm[start - 1] if start > 0 else ""
    before2 = norm[start - 2] if start > 1 else ""
    after = norm[end] if end < len(norm) else ""
    after2 = norm[end + 1] if end + 1 < len(norm) else ""
    if text is not None:
        # whitespace or $ in the original text separates tokens
        o0, o1 = idx[start] - 1, idx[end - 1] + 1
        if o0 >= 0 and (text[o0].isspace() or text[o0] == "$"):
            before = ""
        if o1 < len(text) and (text[o1].isspace() or text[o1] == "$"):
            after = ""
    if ans[0].isalnum() and before and (before.isalnum() or before in "_^"):
        return False
    if ans[-1].isalnum() and after and (after.isalnum() or after == "_"):
        return False
    # decimals: "7" inside "7.5" or "0.7"
    if ans[-1].isdigit() and after == "." and after2.isdigit():
        return False
    if ans[0].isdigit() and before == "." and before2.isdigit():
        return False
    # "1" inside "1/2" or "-3" when the answer is "3"
    if ans[0].isdigit() and before == "-":
        return False
    if ans[-1].isdigit() and after == "/":
        return False
    # "\frac{1}{2}" must not continue as "\frac{1}{2}x" etc.
    if ans[-1] == "}" and after == "{":
        return False
    return True


def _has_context(text: str, start: int) -> bool:
    """True when the original text right before start reads like a result."""
    window = text[max(0, start - 20) : start]
    window = re.sub(r"[\s$\\(\[{*]+$", "", window)
    window = re.sub(r"\\boxed$|\\fbox$", "boxed", window)
    return bool(_CONTEXT_RE.search(window))


def find_answer_char_span(
    think_text: str, answer: Optional[str], problem: Optional[str] = None
) -> Optional[Tuple[int, int]]:
    """First (start, end) character span of the answer in think_text, with end
    exclusive, or None when the answer does not appear."""
    if not answer or not think_text:
        return None
    ans = normalize_for_match(answer)
    if not ans:
        return None
    norm, idx = normalized_view(think_text)
    pnorm, pidx = normalized_view(problem) if problem else ("", [])
    best = None
    for form in answer_forms(ans):
        # one or two digit numbers show up everywhere, so they always need context
        need_context = _short_number(form) or any(
            _boundary_ok(pnorm, m.start(), m.end(), form, problem, pidx)
            for m in re.finditer(re.escape(form), pnorm)
        )
        for m in re.finditer(re.escape(form), norm):
            s, e = m.start(), m.end()
            if not _boundary_ok(norm, s, e, form, think_text, idx):
                continue
            if need_context and not _has_context(think_text, idx[s]):
                continue
            span = (idx[s], idx[e - 1] + 1)
            if best is None or span[0] < best[0]:
                best = span
            break
    return best

def _byte_decoder():
    from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

    return {v: k for k, v in bytes_to_unicode().items()}


def token_char_starts(tokenizer, ids: Sequence[int]) -> Tuple[str, List[int]]:
    """Decode ids and return (text, starts) with starts[i] the character offset
    where token i begins. Exact for byte-level BPE tokenizers (Qwen); falls back
    to per-token decoding otherwise."""
    toks = tokenizer.convert_ids_to_tokens(list(ids)) if hasattr(
        tokenizer, "convert_ids_to_tokens") else None
    if toks is not None:
        try:
            dec = _byte_decoder()
            byte_chunks = []
            for t in toks:
                if all(c in dec for c in t):
                    byte_chunks.append(bytes(dec[c] for c in t))
                else:
                    byte_chunks.append(t.encode("utf-8"))
            data = b"".join(byte_chunks)
            text = data.decode("utf-8", errors="replace")
            # char offset of each byte boundary
            byte_starts = []
            pos = 0
            for chunk in byte_chunks:
                byte_starts.append(pos)
                pos += len(chunk)
            char_of_byte = _char_index_of_bytes(data)
            starts = [char_of_byte[b] for b in byte_starts]
            return text, starts
        except Exception:
            pass
    starts, parts, pos = [], [], 0
    for t in ids:
        piece = tokenizer.decode([t])
        starts.append(pos)
        parts.append(piece)
        pos += len(piece)
    return "".join(parts), starts


def _char_index_of_bytes(data: bytes) -> List[int]:
    """For each byte offset (plus the end), the index of the character that
    contains it in data.decode('utf-8', errors='replace')."""
    out = []
    ch = -1
    i = 0
    n = len(data)
    while i < n:
        b = data[i]
        if b < 0x80:
            k = 1
        elif b >> 5 == 0b110:
            k = 2
        elif b >> 4 == 0b1110:
            k = 3
        elif b >> 3 == 0b11110:
            k = 4
        else:
            k = 1
        try:
            data[i : i + k].decode("utf-8")
        except UnicodeDecodeError:
            k = 1
        ch += 1
        out.extend([ch] * k)
        i += k
    out.append(ch + 1)
    return out[: n + 1]


def char_to_token(starts: Sequence[int], char_pos: int) -> int:
    """Index of the token that contains character char_pos."""
    return max(0, bisect.bisect_right(list(starts), char_pos) - 1)


def answer_token_position(
    think_text: str, starts: Sequence[int], answer: Optional[str], problem: Optional[str] = None
) -> Optional[int]:
    """Token index from which labels are 1 (the token holding the last character
    of the first answer occurrence), or None."""
    span = find_answer_char_span(think_text, answer, problem)
    if span is None:
        return None
    return char_to_token(starts, span[1] - 1)


def token_labels(n_tokens: int, pos: Optional[int]) -> Optional[List[int]]:
    """0 before pos, 1 from pos on. None when pos is None (trace dropped)."""
    if pos is None:
        return None
    pos = max(0, min(pos, n_tokens))
    return [0] * pos + [1] * (n_tokens - pos)
