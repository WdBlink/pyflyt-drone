"""Re-encode only the remaining main phase at the verified order-87 barrier with a larger usage receipt allowance."""

from hashlib import sha256
import json
from pathlib import Path
import sys


BASE = Path(__file__).with_name("project-main-only-owner-space-after-70.py")
EXPECTED_SHA = "a5bda2f60631d5b2b422c71013b1a591f55f50f2db78ff317163d72fb1348b1e"
WIRE_MARGIN_PERCENT = 105
EXCEPTIONAL_WIRE_COPIES = 64


def replace(source, old, new, count=1):
    assert source.count(old) == count, old
    return source.replace(old,new)


def main(status_path):
    original = BASE.read_bytes()
    assert sha256(original).hexdigest() == EXPECTED_SHA
    source = original.decode()
    for old,new,count in (
        ("COUNT = 70","COUNT = 87",1),
        ("after-0070.json","after-0087.json",3),
        ("2e4ee66fffa15429fc98b57e7699191ac0856d0ec9dedfabd97e32b56239e632",
         "d8d7a11afb9855ff97462dbedea2780ec4d34eadead888cc55d52311d842d115",1),
        ("d1c00bb0672f43752da2decdea34f32387d17c6d9dbe7c9f31bd95867710096f",
         "97ac6ffbdfe8af9d717d6cabf19198e4597ee129962231c2491daa61659d9a11",1),
        ("main-ID-L2-13--B4-M2--2.json","main-ID-L3-15--B3-M1--0.json",1),
        ("main-ID-L2-13-B4-M2-2","main-ID-L3-15-B3-M1-0",1),
        ('{"classic": 12, "llm": 58}','{"classic": 14, "llm": 73}',1),
        ("(204, 1022)","(202, 1007)",1),
        ('"round-132"','"round-144"',1),
        ('("recent_regular_plus_exception_reserve_1400",1400,','("recent_regular_plus_exception_reserve_900",900,',1),
        ('("recent_regular_plus_exception_reserve_2000",2000,','("recent_regular_plus_exception_reserve_1000",1000,',1),
        ("44488936","44492257",1),
        ('maxima = read(source_flight)["observed_maxima"]',
         'maxima = read(source_flight)["observed_maxima"]\n    maxima["llm"]["usage_receipt_bytes"] = 16384',1),
    ):
        source = replace(source,old,new,count)
    namespace = {"__file__": str(BASE), "__name__": "inspection_main_wire_variant"}
    exec(compile(source,str(BASE),"exec"),namespace)
    result = namespace["main"](status_path)
    shape_results = {}
    overhead_round_commands = ("status","prepare-round","begin-round","apply-round")
    for label, row in result["scenarios"].items():
        if not label.startswith("recent_regular"):
            continue
        projection = row["projection"]
        assert row["owner_state_within_limit"] and row["document_within_limit"]
        measured = projection["native_peak_encoded_by_command"]
        wire = {key: {name:(size*WIRE_MARGIN_PERCENT+99)//100 for name,size in fields.items()}
                for key,fields in measured.items()}
        previous_upper = row["conservative_wire_maxima"]
        counts = projection["native_command_counts_in_shape"]
        extra_exception_wire = EXCEPTIONAL_WIRE_COPIES * sum(
            sum(previous_upper[key].values()) for key in overhead_round_commands
        )
        n = counts["budget-reserve"]
        assert n == counts["budget-settle"] == 2014
        totals = dict(row["main_other_counters"])
        totals.update(
            episode_native_stdout=n*(wire["budget-reserve"]["stdout"]+wire["budget-settle"]["stdout"]),
            additional_archive_native_stdout=n*wire["budget-reserve"]["stdout"],
            native_cli_traces=sum(count*wire[key]["trace"] for key,count in counts.items()),
            unique_operation_results=sum(count*wire[key].get("result_object",0) for key,count in counts.items()),
            separate_retained_round_command_stdout=sum(counts[key]*wire[key]["stdout"] for key in overhead_round_commands),
            extra_exception_wire=extra_exception_wire,
        )
        # Original reserved 1 GiB misc, 10 GiB, and state-pair allowance remain.
        assert totals["filesystem_headroom"] == 10*1024**3 and totals["miscellaneous_and_analysis"] == 1024**3
        other = sum(totals.values())
        floor = other+result["full_cap2_raw_main_bytes"]+result["unchanged_pack_temporary_bytes"]
        shape_results[label] = {"phase_only_wire_maxima_with_margin":wire,"extra_exception_wire_bytes":extra_exception_wire,
            "main_other_counters":totals,"main_other_bytes":other,"main_full_cap2_floor_bytes":floor,
            "same_volume_margin_bytes":result["same_volume_free_bytes"]-floor}
    assert set(shape_results)=={"recent_regular_plus_exception_reserve_900","recent_regular_plus_exception_reserve_1000"}
    result["phase_wire_adaptation"] = {
        "preserved_base_source_sha256":EXPECTED_SHA,
        "phase_wire_margin_percent":WIRE_MARGIN_PERCENT,
        "separate_exception_wire_reserve_count":EXCEPTIONAL_WIRE_COPIES,
        "regular_two_new_callback_round_count":(sum(result["remaining_slots"].values())+1)//2,
        "requires_native_command_and_size_envelope_recheck":True,
        "results":shape_results,
    }
    return result


if __name__ == "__main__":
    print(json.dumps(main(Path(sys.argv[1])),indent=2,sort_keys=True,allow_nan=False))
