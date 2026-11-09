from copy import deepcopy

import pytest

from eval.inspection_recording import CONDITION_REPEATS
from eval.inspection_secondary_statistics import sr_cluster_intervals, sr_root_table


def data():
    metadata, episodes = [], []
    for level in (1, 2, 3):
        for index in range(2):
            root = f"fixture-{level}-{index}"
            metadata.append(dict(case_id=root, lineage_group=root, domain="ID", level=level))
            for condition, count in CONDITION_REPEATS.items():
                for repeat in range(count):
                    episodes.append(dict(case_id=root, condition=condition, repeat_ordinal=repeat,
                                         status="completed", SR=index))
    return metadata, episodes


def test_correlated_repeats_are_not_extra_roots_and_batch_is_invariant():
    metadata, episodes = data()
    table = sr_root_table(metadata, episodes, require_main_shape=False)
    assert table.values.shape == (6, 8)
    a = sr_cluster_intervals(table, draws=999, batch_size=7)
    b = sr_cluster_intervals(table, draws=999, batch_size=256)
    assert a == b
    assert a["domains"]["ID"]["independent_roots"] == 6
    assert set(a["domains"]["ID"]["level_macro_root_mean_SR"].values()) == {.5}


@pytest.mark.parametrize("fault", ["missing", "duplicate", "unknown", "boolean", "nonbinary", "failed", "bad_repeat"])
def test_full_denominator_and_binary_contract(fault):
    metadata, rows = data()
    rows = deepcopy(rows)
    if fault == "missing":
        rows.pop()
    elif fault == "duplicate":
        rows.append(rows[0])
    else:
        field, value = {"unknown": ("case_id", "other"), "boolean": ("SR", True), "nonbinary": ("SR", .5),
                        "failed": ("status", "operational_failure"), "bad_repeat": ("repeat_ordinal", False)}[fault]
        rows[0][field] = value
    with pytest.raises(ValueError):
        sr_root_table(metadata, rows, require_main_shape=False)
