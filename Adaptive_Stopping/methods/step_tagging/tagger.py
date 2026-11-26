"""Step taggers: map one reasoning step (text only) to a ReasonType tag.

The paper labels steps with GPT-4o-mini and trains one binary BERT classifier
per step type on those labels. We have neither API budget nor labelled data,
so this module provides a **rule-based substitute**, ``KeywordStepTagger``,
labelled *approximate* everywhere it is used. It is black-box like the paper's
tagger (it reads only the generated step text), deterministic, and runs on CPU
in microseconds per step. Its agreement with GPT-4o-mini labels is measured on
the authors' released validation steps by
``Adaptive_Stopping/experiments/st_es/validate_tagger.py``.

Design: each tag has *opening* cues (matched at the start of the step, where
discourse markers such as "Wait," or "Let me check" sit; strong weight) and
*body* cues (anywhere in the step; weak weight). Each cue counts once. The
highest-scoring tag wins; ties follow ``_PRIORITY``; no cue at all gives ``other``.
"""

from __future__ import annotations

import re
from typing import Iterable, Protocol

from Adaptive_Stopping.methods.step_tagging.taxonomy import TAG_IDS


class StepTagger(Protocol):
    name: str

    def tag(self, step_text: str) -> str:
        """Return a ReasonType short id (see ``taxonomy.TAG_IDS``)."""


_LEAD = r"^(?:(?:wait|hmm+|hm|um+|uh|oh|ah|okay|ok|alright|all right|well|so|now|but|and|actually|hold on|first|then|next|also)[,.!:]?\s+)*"

# tag -> list of (weight, compiled regex, scope) where scope is "open" (start of step) or "body".
_RULES_SRC: dict[str, list[tuple[int, str, str]]] = {
    "verification": [
        (4, _LEAD + r"(?:let me |let's |let us |i(?:'ll| will| should| can| want to| need to) |we (?:can|should) |to |just to )?"
                    r"(?:double[- ]?check|check|verify|confirm|make sure|re-?verify|recheck|re-?calculate|recompute|"
                    r"re-?count|cross[- ]?check|sanity[- ]?check|test (?:this|it|that|with))", "open"),
        (4, _LEAD + r"just to (?:be (?:sure|safe|thorough)|double[- ]?check|verify|confirm|make sure)", "open"),
        (4, _LEAD + r"(?:yes|yep|yeah|correct|right|that's right|that's correct|good)[,.!]", "open"),
        (4, _LEAD + r"(?:that|this|it|everything) (?:checks out|matches|seems (?:right|correct|consistent)|is (?:correct|right|consistent)|makes sense)", "open"),
        (4, _LEAD + r"(?:let me |let's )?(?:plug(?:ging)?|substitut(?:e|ing)|put(?:ting)?) .{0,40}back", "open"),
        (4, _LEAD + r"(?:another|one more|a quick) (?:check|way to (?:check|verify|confirm)|sanity check)", "open"),
        (4, _LEAD + r"(?:alternatively|also)[,]? (?:let me |i can |we can |i could )?(?:check|verify|confirm)", "open"),
        (2, r"\b(?:double[- ]?check|sanity check|verif(?:y|ied|ying|ication)|checks out|that matches|consistent with|"
            r"is correct|seems correct|seems right|confirm(?:s|ed|ing)?)\b", "body"),
    ],
    "exploration": [
        (4, _LEAD + r"(?:alternatively|another (?:approach|way|method|idea|strategy)|a different (?:approach|way|method)|"
                    r"maybe (?:there's|there is) (?:a|another) (?:better|simpler|different|easier|smarter)|"
                    r"let me try (?:a different|another|something else|again)|let's try (?:a different|another)|"
                    r"what if|instead[,]? (?:let me|let's|i|we)|perhaps (?:a|another) (?:different|better))", "open"),
        (4, _LEAD + r"(?:maybe|perhaps) (?:i|we) (?:can|could|should) (?:use|try|approach|think about (?:it|this) differently|consider using|instead)", "open"),
        (2, r"\b(?:alternatively|another approach|different approach|another way|other approach|instead of)\b", "body"),
    ],
    "self_talk": [
        (3, r"^(?:but )?(?:wait|hmm+|hm+|um+|uh|oh|hold on|ah)\b[,.!?]?", "open"),
        (3, _LEAD + r"(?:let me (?:think|see|reconsider|re-?read|go (?:back|through)|take a step back|start (?:over|again)|"
                    r"focus|re-?examine|reflect|make sure i)|let's see|i (?:think|wonder|guess|'m (?:not sure|confused|getting)|"
                    r"am (?:not sure|confused)|might have|may have|should be careful|need to be careful)|did i|am i|is that right)", "open"),
        (1, r"\b(?:i think|i'm not sure|i wonder|let me think|did i make a mistake|i might have|hmm|wait)\b", "body"),
    ],
    "problem_restatement": [
        (4, _LEAD + r"(?:i |we )?(?:have|need|'ve got|am given|are given|'m given|got|was given) (?:this|a|the|an) "
                    r"(?:problem|question|math problem|word problem|task)", "open"),
        (4, _LEAD + r"(?:i need to|i have to|we need to|we have to|i'm supposed to|i am supposed to|"
                    r"the problem (?:asks|is asking|wants|says|states|gives|tells)|the question (?:asks|is asking|is)|"
                    r"we(?:'re| are) (?:asked|given|told)|i(?:'m| am) (?:asked|told)) "
                    r"(?:to )?(?:find|figure out|determine|calculate|compute|solve|work out|prove|show|evaluate|how|what|me|us|that)", "open"),
        (3, _LEAD + r"(?:the problem|the question|we are given|given that|we're given|i'm given|the problem statement)\b", "open"),
        (3, _LEAD + r"(?:let me|let's) (?:start by |begin by |first )?(?:understand|read|parse|restate|break down|rephrase|write down what|"
                    r"make sure i understand|figure out what)", "open"),
    ],
    "definition_recall": [
        (3, _LEAD + r"(?:i (?:know|remember|recall)|from what i remember|recall that|remember that|by definition)", "open"),
        (3, _LEAD + r"(?:the|a|an) (?:formula|theorem|definition|rule|property|law|identity|equation) (?:for|of|is|says|states)", "open"),
        (2, r"\b(?:recall(?:ed|ing)? (?:that|the|how)|remember(?:ed|ing)? (?:that|the|how)|by definition|is defined as|"
            r"the definition of|the formula for|formula (?:is|states)|theorem (?:says|states)|states that|"
            r"the rule (?:is|for|says)|properties of|identity)\b", "body"),
    ],
    "formula_substitution": [
        (2, r"\d[\d,.]*\s*(?:[+\-*×x÷/]|\\times|\\cdot|\\div|times|plus|minus|divided by|multiplied by)\s*\$?\(?\d[\d,.]*\)?\s*"
            r"(?:=|is|equals|gives|would be)\s*\$?-?\d", "body"),
        (1, r"\b(?:plug(?:ging)? in|substitut(?:e|ing|ion)|calculat(?:e|ing)|comput(?:e|ing)|multiply(?:ing)?|"
            r"add(?:ing)? (?:up|them|that|these)|subtract(?:ing)?|divid(?:e|ing))\b", "body"),
        (1, r"(?:=|\bis|\bequals|\bgives|\bgets?|\bgot)\s*\$?-?\d", "body"),
    ],
    "symbolic_transformation": [
        (2, r"\b(?:simplif(?:y|ies|ying|ied)|factor(?:ing|ize|ise|s|ed)?|expand(?:ing|ed)?|rewrit(?:e|ing|ten)|"
            r"rearrang(?:e|ing|ed)|combin(?:e|ing) like terms|isolat(?:e|ing)|solv(?:e|ing) for|both sides|"
            r"cancel(?:s|ing|led)? out|distribut(?:e|ing)|common denominator|complet(?:e|ing) the square)\b", "body"),
        (1, r"[a-z]\s*\^|\\frac\{[^}]*[a-z]|\\sqrt\{[^}]*[a-z]|\b\d*[a-z]\s*[+\-]\s*\d+\s*=|\b[xyznk]\s*=", "body"),
    ],
    "edge_case": [
        (3, _LEAD + r"(?:case \d|in (?:the|this) case|consider(?:ing)? (?:the )?(?:case|possibility|scenario)|"
                    r"what about (?:when|if|the case)|if .{0,25}(?:is|=|were) (?:zero|0|negative|positive|odd|even))", "open"),
        (2, r"\b(?:case \d|edge case|special case|corner case|boundary|quadrant|negative (?:value|number|case)|"
            r"is undefined|domain|restriction|could .{0,15} be (?:zero|negative))\b", "body"),
    ],
    "pattern_recognition": [
        (3, _LEAD + r"(?:i notice|notice that|i see that|i observe|observe that|it seems like (?:each|every|there's a pattern)|"
                    r"there(?:'s| is) a pattern)", "open"),
        (2, r"\b(?:pattern|symmetr(?:y|ic|ical)|repeat(?:s|ing)|cycl(?:e|ic|es)|periodic|recurs(?:ive|ion)|"
            r"in general|generaliz(?:e|ing)|every time|each time)\b", "body"),
    ],
    "heuristics": [
        (3, _LEAD + r"(?:intuitively|my intuition|i guess|probably|my guess|it might be (?:easier|better|helpful)|"
                    r"(?:the )?(?:best|easiest|simplest|quickest|most straightforward) (?:way|approach|method))", "open"),
        (2, r"\b(?:intuitive(?:ly)?|intuition|roughly|ballpark|estimat(?:e|ing)|rule of thumb|my guess|"
            r"it seems (?:reasonable|plausible|like a good)|a good (?:idea|approach|strategy|way))\b", "body"),
    ],
    "interpretation": [
        (3, _LEAD + r"(?:that means|this means|which means|in other words|so (?:he|she|they|it|there) (?:has|have|had|will|would|spent|"
                    r"earned|needs|made|gets|got|paid|is|are))", "open"),
        (2, r"\b(?:approximately|approx|about \d|roughly \d|decimal|that means|this means|which means|in other words|"
            r"meaning that|in total|in context|percent(?:age)?)\b|≈|\\approx|\d+\.\d{3,}", "body"),
    ],
    "context_repetition": [
        (3, _LEAD + r"(?:as (?:i|we) (?:said|mentioned|found|calculated|computed|noted|saw|established)|again[,]|so far|"
                    r"earlier[,]? (?:i|we)|to (?:recap|summarize)|recapping|so (?:to )?summarize|we (?:already )?(?:have|know|found) that|"
                    r"(?:i|we) (?:already )?(?:found|calculated|computed|determined|got) (?:that|earlier))", "open"),
        (1, r"\b(?:as (?:i|we) (?:said|mentioned|found) (?:earlier|before)|so far|recap|to summarize|from (?:before|earlier|above))\b", "body"),
    ],
    "final_conclusion": [
        (6, r"\\boxed\s*\{", "body"),
        (4, r"^(?:\*\*)?\s*final answer|" + _LEAD + r"(?:thus|therefore|hence|so)?[,]?\s*(?:the )?(?:final )?answer (?:is|should be|would be)", "open"),
        (2, r"\b(?:the (?:final )?answer is|so the answer|final answer)\b", "body"),
    ],
}

_PRIORITY = ["verification", "exploration", "final_conclusion", "problem_restatement", "self_talk",
             "definition_recall", "edge_case", "pattern_recognition", "heuristics", "interpretation",
             "context_repetition", "symbolic_transformation", "formula_substitution", "other"]

_RULES = {tag: [(w, re.compile(src, re.IGNORECASE | re.DOTALL), scope) for w, src, scope in rules]
          for tag, rules in _RULES_SRC.items()}
_OPEN_STRIP = re.compile(r"^[\s*#>\-]*(?:step \d+[:.)]?|\d+[.)])?[\s*#:\-]*", re.IGNORECASE)
OPEN_WINDOW = 200


class KeywordStepTagger:
    """Rule-based ReasonType tagger (approximate substitute for GPT-4o-mini / BERT taggers)."""

    name = "keyword_reasontype.v1"

    def scores(self, step_text: str) -> dict[str, int]:
        text = step_text or ""
        head = _OPEN_STRIP.sub("", text, count=1)[:OPEN_WINDOW]
        out = {}
        for tag, rules in _RULES.items():
            s = 0
            for weight, rx, scope in rules:
                if (rx.search(head) if scope == "open" else rx.search(text)):
                    s += weight
            out[tag] = s
        return out

    def tag(self, step_text: str) -> str:
        if not step_text or not step_text.strip():
            return "other"
        sc = self.scores(step_text)
        best = max(sc.values())
        if best <= 0:
            return "other"
        for tag in _PRIORITY:
            if sc.get(tag, 0) == best:
                return tag
        return "other"  # pragma: no cover

    def tag_all(self, texts: Iterable[str]) -> list[str]:
        return [self.tag(t) for t in texts]


assert set(_PRIORITY) == set(TAG_IDS), "priority list must cover the taxonomy"
