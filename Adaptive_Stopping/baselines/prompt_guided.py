"""Prompt-guided baselines of the ST-ES paper (Sec. 4 "Baselines", App. D, Fig. 12).

The model is simply told to be brief; there is no early stopping. The paper
compares ST-ES against four variants:

* ``P(0)user``: a zero-shot instruction prefixed to the user prompt;
* ``P(0)system``: a zero-shot system prompt;
* ``P(1)system`` / ``P(3)system``: the system prompt plus 1 or 3 example
  verification steps that the model should *not* produce.

The instruction texts are those of Fig. 12. The few-shot example steps are
copied from the authors' MIT-licensed code (``yannisbelkhiter/TRACES``,
``src/lrm_inference/utils.py``), which implements the same baselines.

Pairing with our standard prompt. Our standard run uses the model card's
"reason step by step … \\boxed{}" directive. The system-prompt variants keep that
exact user prompt and only add the system message, so they differ from standard
generation by the instruction alone. ``P(0)user`` *replaces* the directive with
the paper's user instruction, which asks for a boxed answer itself, as in the
authors' code. The DeepSeek-R1 model card advises against system prompts; the
paper uses them for these baselines anyway, and so do we.
"""

from __future__ import annotations

from Dataloaders.evaluation.gsm8k_math import MathExample, build_messages

SYSTEM_ZS = "Respond concisely and confidently. Skip validations and over-verification steps."
USER_ZS = "Please do not reason extensively, be succinct, and put your final answer within \\boxed{{}}. {question}"

# Example verification steps (authors' code, MIT; also printed in Fig. 12 of the paper).
FS_1 = ("Wait, let me double-check. If I plug in x = -3 into the denominator, (-3)^2 + (-3) -6 = 9 -3 -6 = 0. "
        "Yep, that works. For x =2: 2^2 +2 -6 =4 +2 -6=0. Correct. So both roots are valid. \n\n")
FS_2 = ("Therefore, the graph of y=2/(x\u00b2 +x -6) has vertical asymptotes at x= -3 and x=2, so that's two vertical "
        "asymptotes. \n\nI don't think there's any chance that I made a mistake here, but maybe I should check by "
        "graphing the function or plugging in values close to -3 and 2 to see if the function does go to infinity.")
FS_3 = ("Another test with n=3. Let's compute manually. \n\nAll non-empty subsets:\n\nSingle elements: {1}, {2}, {3} "
        "with sums 1,2,3.\n\nPairs: {1,2} \u2192 2-1=1; {1,3}\u21923-1=2; {2,3}\u21923-2=1.\n\nTriple: {1,2,3} \u21923 -2 +1=2."
        "\n\nTotal sum: 1+2+3 +1+2+1 +2 = 12.\n\nUsing the formula: contributions from each k:\n\nk=3: 3*2^{2}*1= 3*4=12. "
        "\n\nk=1 and k=2 contribute 0. \n\nSo total sum 12, which matches. \n\n")

VARIANTS = {
    # id: (paper name, description)
    "pguided_user_zs.v1": ("P(0)user", "zero-shot user instruction (replaces the CoT/boxed directive)"),
    "pguided_system_zs.v1": ("P(0)system", "zero-shot system prompt + standard user prompt"),
    "pguided_system_fs1.v1": ("P(1)system", "system prompt with 1 example + standard user prompt"),
    "pguided_system_fs3.v1": ("P(3)system", "system prompt with 3 examples + standard user prompt"),
}


def system_prompt(variant: str) -> str | None:
    if variant == "pguided_system_zs.v1":
        return SYSTEM_ZS
    if variant == "pguided_system_fs1.v1":
        return f"{SYSTEM_ZS} Here is an example: \n Example 1: {FS_1}"
    if variant == "pguided_system_fs3.v1":
        return f"{SYSTEM_ZS} Here are some examples: \n Example 1: {FS_1} \n Example 2: {FS_2} \n Example 3: {FS_3}"
    return None


def build_guided_messages(example: MathExample, variant: str, standard_prompt_id: str = "math_cot_boxed.v1") -> list[dict]:
    """Chat messages for one prompt-guided baseline."""
    if variant not in VARIANTS:
        raise KeyError(f"unknown prompt-guided variant {variant!r}; known: {sorted(VARIANTS)}")
    if variant == "pguided_user_zs.v1":
        return [{"role": "user", "content": USER_ZS.format(question=example.question.strip())}]
    return [{"role": "system", "content": system_prompt(variant)}] + build_messages(example, standard_prompt_id)
