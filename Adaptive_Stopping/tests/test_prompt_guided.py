import pytest

from Adaptive_Stopping.baselines.prompt_guided import (
    FS_1, FS_2, FS_3, SYSTEM_ZS, VARIANTS, build_guided_messages, system_prompt,
)
from Dataloaders.evaluation.gsm8k_math import MathExample, build_messages

EX = MathExample("u", "text.gsm8k.test", 0, "  What is 2+2? ", "4", "number")


def test_user_variant_follows_paper_figure_12():
    msgs = build_guided_messages(EX, "pguided_user_zs.v1")
    assert msgs == [{"role": "user", "content": "Please do not reason extensively, be succinct, and put your final "
                                                 "answer within \\boxed{}. What is 2+2?"}]


@pytest.mark.parametrize("variant,n_examples", [("pguided_system_zs.v1", 0), ("pguided_system_fs1.v1", 1),
                                                ("pguided_system_fs3.v1", 3)])
def test_system_variants_add_only_a_system_message(variant, n_examples):
    msgs = build_guided_messages(EX, variant)
    assert msgs[0]["role"] == "system" and msgs[0]["content"].startswith(SYSTEM_ZS)
    assert msgs[1:] == build_messages(EX)  # same user prompt as the standard run
    shown = [fs for fs in (FS_1, FS_2, FS_3) if fs in msgs[0]["content"]]
    assert len(shown) == n_examples


def test_variant_registry_and_errors():
    assert [VARIANTS[v][0] for v in VARIANTS] == ["P(0)user", "P(0)system", "P(1)system", "P(3)system"]
    assert system_prompt("pguided_user_zs.v1") is None
    with pytest.raises(KeyError):
        build_guided_messages(EX, "nope")


def test_messages_render_with_backbone_chat_template_if_cached():
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained(
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", revision="ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562",
            local_files_only=True)
    except Exception:
        pytest.skip("backbone tokenizer not cached")
    text = tok.apply_chat_template(build_guided_messages(EX, "pguided_system_zs.v1"), tokenize=False,
                                   add_generation_prompt=True)
    assert SYSTEM_ZS in text and text.index(SYSTEM_ZS) < text.index("What is 2+2?") and text.endswith("<think>\n")
