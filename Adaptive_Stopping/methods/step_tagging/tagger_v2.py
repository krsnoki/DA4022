"""Rule-based ReasonType tagger, version 2 (``keyword_reasontype.v2``).

v2 extends v1's rule table. The changes come **only** from reading v1's output on
our own calibration traces (GSM8K train and MATH train, 4,914 reasoning steps),
in particular the steps v1 left as ``other``. None come from the authors' 99
labelled validation steps, which stay a held-out check
(``experiments/st_es/tagger_analysis.py``).

| Observed on calibration traces | v2 change |
|---|---|
| Short planning lines ending in ":" ("Let me write that down:", "Which is:") | Self-talk (the paper's Table 2 puts "plan of action" under Self-Talk). |
| Plans such as "Let me denote / break it down / set up" | Self-talk opening cue. |
| Lists of given values ("Blue: 10", "Day 1: 78") | Context repetition. |
| Result statements ("So, Manny will have $45 left") | Interpretation. |
| "In conclusion …" | Final conclusion. |
| "Wait, …, let me check …" tagged as self-talk | Stronger verification cue anywhere in the step; "I don't see any mistakes", "same result", "not equal" also count. |
| "Wait, maybe another way" tagged as self-talk | Stronger exploration cue anywhere in the step. |
| "I have this geometry problem", a quoted problem, "Let x be …" setup | Problem re-statement / setup. |
| "Where C(n, k) is …", named theorems (Vieta, Pythagoras, …) | Definition recall. |
| "That's 20", "which is 6" | Counted as a calculation. |

v1 (``tagger.KeywordStepTagger``) stays the tagger of the reported main results.
"""

from __future__ import annotations

import copy
import re

from Adaptive_Stopping.methods.step_tagging import tagger as v1

_LEAD = v1._LEAD
_EXTRA: dict[str, list[tuple[int, str, str]]] = {
    "verification": [
        (4, r"\b(?:let me|let's|i(?:'ll| should| need to| want to| can))\s+(?:just\s+)?(?:double[- ]?check|check|verify|"
            r"recheck|re-?verify|confirm|make sure)\b|\bcheck (?:if|whether|that)\b|\bto make sure\b", "body"),
        (3, r"\bi don'?t (?:see|think there(?:'s| is| are)) any (?:issues?|mistakes?|errors?|problems?|complications?)\b|"
            r"\b(?:same|consistent) (?:result|answer)\b|\bnot equal\b|\bthat'?s (?:correct|right)\b", "body"),
    ],
    "exploration": [
        (3, r"\b(?:another|different|alternative) (?:way|approach|method|strategy)\b|\balternatively\b", "body"),
    ],
    "self_talk": [
        (3, _LEAD + r"(?:let me|let's|i(?:'ll| will| need to| should| have to))\s+(?:first\s+|now\s+|just\s+)?"
                    r"(?:write|denote|break|set up|organi[sz]e|list|lay out|start|go (?:through|step)|think|"
                    r"figure out|see|try to|represent|define|express)\b", "open"),
        (2, r"^[^\n]{0,80}:\s*$", "body"),  # a short planning line ending in a colon
    ],
    "context_repetition": [
        (3, r"^\s*(?:[-*]\s*)?[A-Za-z][\w' ]{0,30}:\s*\$?\d[\d,.]*(?:\s*[A-Za-z%]+)?\s*(?:$|\n|/)", "body"),
    ],
    "interpretation": [
        (2, _LEAD + r"(?:so|therefore|thus|hence)[,]?\s+(?:\S+\s+){0,5}?(?:has|have|had|will|would|spent|earned|"
                    r"needs?|made|gets?|got|paid|kept|costs?|ends? up|left|is|are)\b[^.\n]{0,40}\$?\d", "open"),
    ],
    "final_conclusion": [
        (4, _LEAD + r"(?:so|therefore|thus|hence)?[,]?\s*(?:in conclusion|to conclude|in the end)\b", "open"),
        (2, _LEAD + r"(?:so|therefore|thus|hence)?[,]?\s*putting it all together\b", "open"),
    ],
    "problem_restatement": [
        (4, _LEAD + r"(?:i |we )?(?:have|need|'ve got|am given|are given|'m given|got)\s+(?:this|a|the|an)"
                    r"(?:\s+\w+){1,2}\s+(?:problem|question)", "open"),
        (4, r"^\s*[\"“][^\"”]{60,}", "open"),  # the problem quoted back verbatim
        (3, r"^(?:let|suppose|assume|denote)\b.{0,60}\b(?:be|denote|represent|=)", "open"),
    ],
    "definition_recall": [
        (3, r"^where\b.{0,60}\b(?:is|are|denotes|represents)\b", "open"),
        (2, r"\b(?:vieta|pythag\w*|euclid\w*|binomial theorem|law of (?:sines|cosines)|quadratic formula|fermat|"
            r"euler|de moivre|heron|stewart)\b", "body"),
    ],
    "formula_substitution": [
        (1, r"\b(?:that'?s|which is|which gives|makes|totals?)\s*\$?-?\d", "body"),
    ],
}


def _build_rules():
    src = copy.deepcopy(v1._RULES_SRC)
    for tag, extra in _EXTRA.items():
        src[tag] = src.get(tag, []) + extra
    return {tag: [(w, re.compile(s, re.IGNORECASE | re.DOTALL | re.MULTILINE), scope) for w, s, scope in rules]
            for tag, rules in src.items()}


_RULES_V2 = _build_rules()


class KeywordStepTaggerV2(v1.KeywordStepTagger):
    """v1 scoring and tie-breaking, with the extended rule table above."""

    name = "keyword_reasontype.v2"

    def scores(self, step_text: str) -> dict[str, int]:
        text = step_text or ""
        head = v1._OPEN_STRIP.sub("", text, count=1)[:v1.OPEN_WINDOW]
        out = {}
        for tag, rules in _RULES_V2.items():
            out[tag] = sum(w for w, rx, scope in rules if (rx.search(head) if scope == "open" else rx.search(text)))
        return out
