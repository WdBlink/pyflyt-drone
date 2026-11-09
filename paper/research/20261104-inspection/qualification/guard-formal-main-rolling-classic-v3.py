"""Use the reviewed main guard with the order-91 classic allowance."""

from hashlib import sha256
from pathlib import Path
import sys


def main():
    if sys.flags.optimize:
        raise SystemExit("guard requires Python assertions")
    base = Path(__file__).with_name("guard-formal-main-rolling.py")
    original = base.read_bytes()
    assert sha256(original).hexdigest() == "2143908dffccf08be68542465874567beed52aac6af35eb92ae345f7df932feb"
    source = original.decode()
    replacements = (
        ("after-0072-three-id-roots-rolling-capacity-admission.v1.json",
         "after-0091-classic-buffered-rolling-capacity-admission.v1.json"),
        ("inspection-main-three-id-roots-rolling-capacity-v1",
         "inspection-main-classic-buffered-rolling-capacity-v1"),
        ("assert 72 <= index < 126", "assert 91 <= index < 126"),
        ('"first": 72, "end_exclusive": 126', '"first": 91, "end_exclusive": 126'),
        ('assert admission["native_admission_round"] == 134',
         'assert admission["native_admission_round"] == 147'),
        ("    if index > 72:", "    if index >= 91:"),
        ('        assert match(prior, slots[index - 1]) and prior["native_capacity_decision_sha256"] == expected',
         '        predecessor = "82c530225792e0574853d74602276344e5bebd0fce3e8d29a3c0f012b4c19674"\n'
         '        prior_decision = predecessor if index == 91 else expected\n'
         '        assert match(prior, slots[index - 1]) and prior["native_capacity_decision_sha256"] == prior_decision'),
        ('    assert raw == {"classic": 29546687, "llm": 44488936} and admission["temporary_bytes"] == 88977872',
         '    assert raw == {"classic": 33554432, "llm": 44492257} and admission["temporary_bytes"] == 88984514\n'
         '    assert admission["component_allowance"]["classic"]["episode_nonwire_bytes"] == raw["classic"]\n'
         '    assert admission["component_allowance"]["llm"]["usage_receipt_bytes"] == 16384\n'
         '    assert sum(admission["component_allowance"]["llm"].values()) == raw["llm"]'),
    )
    for before, after in replacements:
        assert source.count(before) == 1, before
        source = source.replace(before, after)
    exec(compile(source, str(base), "exec"), {"__file__": __file__, "__name__": "__main__"})


if __name__ == "__main__":
    main()
