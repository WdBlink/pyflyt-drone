"""Use the preserved projection source with two justified main checkpoint counts."""

from hashlib import sha256
import json
from pathlib import Path
import sys


BASE = Path(__file__).with_name("project-main-only-owner-space-after-70.py")
EXPECTED_SHA = "a5bda2f60631d5b2b422c71013b1a591f55f50f2db78ff317163d72fb1348b1e"


def main(status_path):
    original = BASE.read_bytes()
    assert sha256(original).hexdigest() == EXPECTED_SHA
    source = original.decode()
    for old, new in ((1400, 1000), (2000, 1100)):
        before = f'("recent_regular_plus_exception_reserve_{old}",{old},'
        after = f'("recent_regular_plus_exception_reserve_{new}",{new},'
        assert source.count(before) == 1
        source = source.replace(before, after)
    namespace = {"__file__": str(BASE), "__name__": "inspection_main_round_count_variant"}
    exec(compile(source, str(BASE), "exec"), namespace)
    result = namespace["main"](status_path)
    left = sum(result["remaining_slots"].values())
    assert left == 1226
    routine_bound = (left + 1) // 2
    result["source_variant"] = {
        "base_sha256": EXPECTED_SHA,
        "modified_template_counts": {"1400": 1000, "2000": 1100},
        "other_projections_unchanged": True,
    }
    result["round_storage_planning"] = {
        "full_per_slot_receipts_still_required": True,
        "normal_rounds_for_two_new_verified_slots_each": routine_bound,
        "extra_round_places_in_1000_shape": 1000 - routine_bound,
        "extra_round_places_in_1100_shape": 1100 - routine_bound,
        "historical_large_rounds_reserved_on_top_for_owner_bytes": 64,
        "review_before_admitting_growth_outside_observed_shape": True,
        "role": "Sizing reserve, not a task budget or authority to skip checkpoints or effects.",
    }
    return result


if __name__ == "__main__":
    print(json.dumps(main(Path(sys.argv[1])), indent=2, sort_keys=True, allow_nan=False))
