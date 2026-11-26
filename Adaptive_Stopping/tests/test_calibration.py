from Adaptive_Stopping.methods.step_tagging.calibration import Config, build_grid, select_configs


def rows():
    return [
        {"kind": "st_es", "key": "a", "delta": 0, "k": None, "accuracy": 0.60, "mean_tokens": 300},
        {"kind": "st_es", "key": "b", "delta": 1, "k": None, "accuracy": 0.72, "mean_tokens": 450},
        {"kind": "st_es", "key": "c", "delta": 2, "k": None, "accuracy": 0.76, "mean_tokens": 500},
        {"kind": "st_es", "key": "d", "delta": 3, "k": None, "accuracy": 0.76, "mean_tokens": 500},  # tie on tokens/acc
        {"kind": "st_es", "key": "e", "delta": 5, "k": None, "accuracy": 0.80, "mean_tokens": 700},
        {"kind": "step_count", "key": "k5", "delta": None, "k": 5, "accuracy": 0.79, "mean_tokens": 650},
    ]


def test_select_cheapest_config_meeting_each_target():
    picks = select_configs(rows(), baseline_accuracy=0.80, kind="st_es", targets=(0.85, 0.90, 0.95, 1.0, 1.1))
    assert picks[0.85]["key"] == "b"     # 0.72 >= 0.68
    assert picks[0.90]["key"] == "b"     # 0.72 >= 0.72 (boundary is inclusive)
    assert picks[0.95]["key"] == "c"     # 0.76 >= 0.76, tie with d broken by smaller delta
    assert picks[1.0]["key"] == "e"
    assert picks[1.1] is None            # unreachable target


def test_select_respects_kind():
    picks = select_configs(rows(), baseline_accuracy=0.80, kind="step_count", targets=(0.9, 0.99))
    assert picks[0.9]["key"] == "k5" and picks[0.99] is None


def test_grid_excludes_other_and_covers_paper_deltas():
    grid = build_grid()
    st = [c for c in grid if c.kind == "st_es"]
    assert all(c.tag != "other" for c in st)
    assert {c.delta for c in st} == set(range(21))
    assert len({c.key for c in grid}) == len(grid)
    c = Config("st_es", tag="verification", delta=3)
    assert Config.from_dict(c.to_dict()) == c
