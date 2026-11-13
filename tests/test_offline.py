"""Fast tests that need no GPU, model download or dataset download."""
import pandas as pd
import pytest
import torch

from spare.config import SpareConfig
from spare.editor import LowLevelEditor
from spare.evaluation import generation_entropy
from spare.linguistic import linguistic_subgroups
from spare.prompting import build_label_space, one_shot_messages, parse_output
from spare.ripple_protect import build_ripple_protect
from spare.utils import WRONG_TOKEN, RunContext, norm_label

LABELS = pd.DataFrame({"label": ["caution_and_advice", "injured_or_dead_people", "sympathy_and_support"]})


def test_parse_output_variants():
    _, _, rev = build_label_space(LABELS)
    assert parse_output("caution and advice", rev) == "caution_and_advice"
    assert parse_output("Category: Injured or dead people.", rev) == "injured_or_dead_people"
    assert parse_output("sympathy_and_support", rev) == "sympathy_and_support"
    assert parse_output("no idea", rev) is None


def test_one_shot_prompt_layout():
    _, verb, _ = build_label_space(LABELS)
    msg = one_shot_messages("domain", verb, "bridge collapsed", [("stay inside", "caution_and_advice")])
    content = msg[0]["content"]
    assert content.index("Examples:") < content.index("Text: bridge collapsed")
    assert content.endswith("Category:")


def test_norm_label():
    assert norm_label(" Caution and-Advice ") == "caution_and_advice"
    assert norm_label("") == WRONG_TOKEN


def test_ripple_protect(tmp_path):
    ctx = RunContext(SpareConfig(dataset="HumAID", out_dir=tmp_path, n_samples=6))
    rows = []
    for i, (true, pred) in enumerate([("a", "a"), ("a", "b"), ("b", "b"), ("b", "a"), ("b", "c"),
                                      ("c", WRONG_TOKEN)]):
        rows.append({"sample_id": str(i), "text": f"t{i}", "true_label": true, "predicted_label": pred,
                     "is_correct": true == pred, "invalid_model_output": pred == WRONG_TOKEN,
                     "shot": "one", "prompt": "domain", "iteration": 0})
    pd.DataFrame(rows).to_csv(ctx.path("one_shot_domain_predictions.csv"), index=False)
    ripple, protect, summary = build_ripple_protect(ctx)
    assert set(ripple["sample_id"]) == {"1", "3", "4"}           # invalid output "5" is excluded
    assert set(protect["sample_id"]) == {"0", "2"}
    assert dict(zip(summary["true_label"], summary["ripple_size"])) == {"a": 1, "b": 2}


def test_editor_update_is_masked_rank_one():
    d_m, d_h = 6, 4
    editor = LowLevelEditor(d_m, d_h, layer_ids=[0], d_r=3, C=2)
    mask = torch.zeros(1, d_m)
    mask[0, [1, 4]] = 1.0
    dW = editor.delta_w(torch.randn(d_m), torch.randn(d_h), 0, mask)
    assert dW.shape == (d_h, d_m)
    assert torch.count_nonzero(dW[:, [0, 2, 3, 5]]) == 0
    assert torch.linalg.matrix_rank(dW) <= 1


def test_generation_entropy_and_linguistic():
    assert generation_entropy("one two") != generation_entropy("one two three four")
    df = pd.DataFrame({"text": ["Help! #flood @news", "is anyone safe?", "BRIDGE DOWN 12", "pray"],
                       "correct_W0": [False, True, False, True], "correct_WT": [True, True, True, False]})
    t = linguistic_subgroups(df).set_index("subgroup")
    assert t.loc["Has Hashtags", "n"] == 1 and t.loc["Has Hashtags", "accuracy_after"] == 100
    assert t.loc["Uppercase Heavy", "n"] == 1
    with pytest.raises(KeyError):
        t.loc["Unknown"]
