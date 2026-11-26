"""CPU end-to-end smoke tests on a tiny random Qwen2 model with an in-memory character tokenizer.

Nothing is downloaded. These check the mechanics of cached generation, live
early stopping, answer forcing and replay/live agreement, not answer quality.
"""

import string

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

from Adaptive_Stopping.baselines.step_count import StepCountStopping  # noqa: E402
from Adaptive_Stopping.common.cached_generation import GenJob, generate_full_traces  # noqa: E402
from Adaptive_Stopping.common.forced_answer import force_answers, force_answers_shared_prefix  # noqa: E402
from Adaptive_Stopping.common.live_generation import StepPolicyCriteria, generate_with_policy  # noqa: E402
from Adaptive_Stopping.common.replay import replay_policy  # noqa: E402
from Adaptive_Stopping.common.steps import segment_token_ids  # noqa: E402
from Adaptive_Stopping.methods.step_tagging.policy import StepTagEarlyStopping  # noqa: E402

PAD, EOS = 0, 1


@pytest.fixture(scope="module")
def tiny():
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    vocab = {"<pad>": PAD, "<eos>": EOS, "<unk>": 2, "</think>": 3}
    for ch in string.printable:
        vocab.setdefault(ch, len(vocab))
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")
    tk.decoder = decoders.Fuse()
    tk.add_special_tokens(["<pad>", "<eos>", "</think>"])
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")
    cfg = Qwen2Config(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                      eos_token_id=EOS, pad_token_id=PAD, bos_token_id=EOS)
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(cfg).eval()
    return tok, model, vocab


def newline_model(model, vocab):
    """Copy of the tiny model whose LM head always prefers '\\n' (so '\\n\\n' closes a step every 2 tokens)."""
    import copy

    m = copy.deepcopy(model)
    head = torch.nn.Linear(m.config.hidden_size, m.config.vocab_size, bias=True)
    torch.nn.init.zeros_(head.weight)
    torch.nn.init.zeros_(head.bias)
    head.bias.data[vocab["\n"]] = 20.0
    m.lm_head = head
    return m


def test_tokenizer_roundtrip(tiny):
    tok, _, _ = tiny
    text = "a = 1\n\nWait, check\n</think>\n\n\\boxed{2}"
    ids = tok(text, add_special_tokens=False)["input_ids"]
    assert tok.decode(ids, skip_special_tokens=False) == text
    steps = segment_token_ids(ids, tok)
    assert [s.text for s in steps] == text.split("\n\n")


def test_cached_generation_with_continuations(tiny):
    tok, model, _ = tiny
    jobs = [GenJob(uid=f"j{i}", prompt_ids=tok(p, add_special_tokens=False)["input_ids"])
            for i, p in enumerate(["what is 2+2?", "hi", "solve x"])]
    finished = []
    stats = generate_full_traces(model, jobs, eos_ids=[EOS], pad_id=PAD, max_new_tokens=20, seed=0,
                                 sampling={"do_sample": True, "temperature": 0.6, "top_p": 0.95, "top_k": 50},
                                 chunk_tokens=7, max_batch=2, on_finish=finished.append, log=lambda *_: None)
    assert {j.uid for j in finished} == {"j0", "j1", "j2"}
    for j in jobs:
        assert j.finish_reason in ("eos", "length") and len(j.gen_ids) <= 20
        if j.finish_reason == "length":
            assert len(j.gen_ids) == 20 and j.rounds >= 3  # 20 tokens in chunks of 7
    assert stats["rounds"] >= 3


def test_batched_rounds_match_plain_greedy_generation(tiny):
    """Per-row prefill + assembled padded cache + continuation rounds == plain per-prompt greedy decoding."""
    tok, model, _ = tiny
    prompts = ["what is 2+2?", "hi", "solve x for me please"]
    jobs = [GenJob(uid=str(i), prompt_ids=tok(p, add_special_tokens=False)["input_ids"]) for i, p in enumerate(prompts)]
    generate_full_traces(model, jobs, eos_ids=[EOS], pad_id=PAD, max_new_tokens=15, seed=0,
                         sampling={"do_sample": False}, chunk_tokens=4, max_batch=3, log=lambda *_: None)
    for j in jobs:
        ids = torch.tensor([j.prompt_ids])
        ref = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=15,
                             do_sample=False, eos_token_id=[EOS], pad_token_id=PAD)[0, ids.shape[1]:].tolist()
        if EOS in ref:
            ref = ref[:ref.index(EOS)]
        assert j.gen_ids == ref


def test_chunked_row_prefill_matches_one_shot(tiny):
    """Chunked per-row prefill (used for long contexts) gives the same continuation as one-shot prefill."""
    from Adaptive_Stopping.common import cached_generation as cg

    tok, model, _ = tiny
    ctx = tok("a long prompt that spans several prefill chunks 0123456789", add_special_tokens=False)["input_ids"]
    width = len(ctx)
    outs = []
    for chunk in (4, 2048):
        cache = cg._prefill_rows(model, [ctx], width, torch.device("cpu"), max_new=6, prefill_chunk=chunk)
        ids = torch.tensor([ctx])
        g = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), cache=cache, max_new_tokens=6,
                           do_sample=False, eos_token_id=[EOS], pad_token_id=PAD, disable_compile=True)
        outs.append(g[0, width:].tolist())
    assert outs[0] == outs[1]


def test_shared_prefix_forced_answers_match_fresh_prefill(tiny):
    tok, model, _ = tiny
    full = tok("question: 3*4?\n\nstep one\n\nstep two\n\nstep three", add_special_tokens=False)["input_ids"]
    cuts = [10, 18, 27, len(full)]
    shared = force_answers_shared_prefix(model, tok, full, cuts, eos_ids=[EOS], pad_id=PAD, max_new_tokens=8,
                                         max_batch=3, prefill_chunk=7)  # chunked prefill must not change answers
    fresh = force_answers(model, tok, [full[:c] for c in cuts], eos_ids=[EOS], pad_id=PAD, max_new_tokens=8,
                          max_batch=2)
    assert [shared[c] for c in cuts] == fresh
    assert all(len(a) <= 8 for a in fresh)


def test_live_policy_stop_matches_replay(tiny):
    tok, model, vocab = tiny
    m = newline_model(model, vocab)
    prompt = tok("go", add_special_tokens=False)["input_ids"]
    common = dict(max_new_tokens=40, sampling={"do_sample": True, "temperature": 0.6, "top_p": 0.95, "top_k": 50},
                  seed=0, eos_ids=[EOS], pad_id=PAD, answer_budget=5)
    base = generate_with_policy(m, tok, prompt, None, **common)
    assert base["finish_reason"] == "length" and len(base["gen_ids"]) == 40 and not base["stopped"]
    live = generate_with_policy(m, tok, prompt, StepCountStopping(3), **common)
    assert live["stopped"] and live["finish_reason"] == "policy"
    assert live["cut"] == 6 and len(live["gen_ids"]) == 6       # '\n\n' x 3
    assert len(live["answer_ids"]) == 5                         # never closes the box -> full answer budget
    assert live["gen_ids"] == base["gen_ids"][:6]               # same seed: live run is a prefix of standard
    steps = segment_token_ids(base["gen_ids"], tok)
    rep = replay_policy(StepCountStopping(3), steps, len(base["gen_ids"]))
    assert rep.cut_tokens == live["cut"]


def test_live_forced_answer_equals_fresh_prefill(tiny):
    """The live path reuses generate()'s KV cache for the forced answer; it must match a fresh prefill."""
    import copy

    from Adaptive_Stopping.common.live_generation import continue_with_exit_prompt

    tok, model, _ = tiny
    prompt = tok("solve: 12*3", add_special_tokens=False)["input_ids"]
    for seed, n in [(0, 5), (1, 9), (2, 14)]:
        ids = torch.tensor([prompt])
        torch.manual_seed(seed)
        out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=n, do_sample=True,
                             temperature=1.0, top_p=1.0, top_k=0, eos_token_id=[EOS], pad_token_id=PAD,
                             return_dict_in_generate=True)
        gen = out.sequences[0, len(prompt):].tolist()
        live = continue_with_exit_prompt(model, tok, out, eos_ids=[EOS], pad_id=PAD, answer_budget=8)
        fresh = force_answers(model, tok, [prompt + gen], eos_ids=[EOS], pad_id=PAD, max_new_tokens=8)[0]
        assert live == fresh, (seed, live, fresh)


def test_step_policy_criteria_simulated_generate(tiny):
    tok, _, _ = tiny
    text = "Okay, so I need to find x.\n\nWait, let me check this.\n\nYes, that matches.\n\nmore"
    ids = tok(text, add_special_tokens=False)["input_ids"]
    prompt = [5, 6]
    crit = StepPolicyCriteria(StepTagEarlyStopping("verification", 0), tok, len(prompt))
    seq = list(prompt)
    stopped_at = None
    for n, t in enumerate(ids, start=1):
        seq.append(t)
        if bool(crit(torch.tensor([seq]))[0]):
            stopped_at = n
            break
    first_two = len(tok("Okay, so I need to find x.\n\nWait, let me check this.\n\n", add_special_tokens=False)["input_ids"])
    assert stopped_at == first_two == crit.cut and crit.stop_step == 1
    assert crit.policy_time_s >= 0.0
