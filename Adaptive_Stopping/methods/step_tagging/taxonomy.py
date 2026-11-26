"""ReasonType: the 14-class taxonomy of reasoning-step types used by ST-ES.

Label names follow the paper (Fig. 2) and the label set the authors used with
GPT-4o-mini (MIT-licensed ``yannisbelkhiter/TRACES`` and
``yannisbelkhiter/ReasonType_Human_validation``). Numbers are the paper's.
The one-line descriptions paraphrase the authors' annotator guide.
"""

from __future__ import annotations

# (short id, paper number, official label, phase, description)
REASONTYPE = [
    ("problem_restatement", 1, "Problem Re-statement / Setup", "early",
     "Restates or rephrases the problem without adding new content."),
    ("context_repetition", 2, "Context Repetition / Paraphrasing", "early",
     "Repeats given information or prior results without transforming them."),
    ("definition_recall", 3, "Definition Recall", "early",
     "Recalls a definition, theorem or known fact."),
    ("formula_substitution", 4, "Formula Substitution / Plugging In", "mid",
     "Substitutes values into a formula or carries out a calculation."),
    ("symbolic_transformation", 5, "Symbolic Transformation / Rewriting Sums", "mid",
     "Algebraically rewrites or simplifies an expression."),
    ("edge_case", 6, "Quadrant/Edge Case Consideration", "mid",
     "Considers a specific case, region, sign or edge condition."),
    ("pattern_recognition", 7, "Pattern Recognition / Symmetry", "mid",
     "Notices a pattern, regularity or symmetry."),
    ("verification", 8, "Verification / Sanity Check", "late",
     "Checks a prior result for correctness or consistency."),
    ("heuristics", 9, "Heuristics / Intuition", "late",
     "Uses an informal heuristic or intuition to pick an approach."),
    ("exploration", 10, "Alternative Approach Exploration", "late",
     "Tries a different method or explores another path."),
    ("interpretation", 11, "Numerical Approximation / Interpretation", "late",
     "Approximates a quantity or interprets what a result means."),
    ("self_talk", 12, "Meta-Cognition / Self-Talk", "late",
     "Reflects on its own reasoning process ('wait', 'let me reconsider')."),
    ("final_conclusion", 13, "Final Conclusion / Boxed Answer", "end",
     "States the final answer."),
    ("other", 14, "Other", "end", "Does not fit any other category."),
]

TAG_IDS = [t[0] for t in REASONTYPE]
LABEL_TO_ID = {t[2]: t[0] for t in REASONTYPE}
ID_TO_LABEL = {t[0]: t[2] for t in REASONTYPE}
ID_TO_NUMBER = {t[0]: t[1] for t in REASONTYPE}

# Label variants that appear in the authors' released files.
LABEL_TO_ID.update({"Other - please, specify": "other"})

# The seven step types for which the paper trains BERT step-taggers (Sec. 6, Fig. 4).
PAPER_BERT_TAGS = ["problem_restatement", "definition_recall", "formula_substitution", "exploration",
                   "heuristics", "verification", "self_talk"]


def to_tag_id(label_or_id: str) -> str:
    """Map an official label (or a short id) to the short id."""
    if label_or_id in ID_TO_LABEL:
        return label_or_id
    if label_or_id in LABEL_TO_ID:
        return LABEL_TO_ID[label_or_id]
    raise KeyError(f"unknown ReasonType label {label_or_id!r}")
