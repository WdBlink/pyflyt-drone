"""Use the reviewed main guard with the order-87 usage receipt allowance."""

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
         "after-0087-receipt-buffered-rolling-capacity-admission.v1.json"),
        ("inspection-main-three-id-roots-rolling-capacity-v1",
         "inspection-main-usage-receipt-buffered-rolling-capacity-v1"),
        ("assert 72 <= index < 126", "assert 87 <= index < 126"),
        ('"first": 72, "end_exclusive": 126', '"first": 87, "end_exclusive": 126'),
        ('assert admission["native_admission_round"] == 134',
         'assert admission["native_admission_round"] == 144'),
        ("    if index > 72:", "    if index >= 87:"),
        ('        assert match(prior, slots[index - 1]) and prior["native_capacity_decision_sha256"] == expected',
         '        predecessor = "429e89c07ea0717dd06f5fbea70db292cc7cc44e968fbf555f94e3069178738d"\n'
         '        prior_decision = predecessor if index == 87 else expected\n'
         '        assert match(prior, slots[index - 1]) and prior["native_capacity_decision_sha256"] == prior_decision'),
        ('    assert raw == {"classic": 29546687, "llm": 44488936} and admission["temporary_bytes"] == 88977872',
         '    assert raw == {"classic": 29546687, "llm": 44492257} and admission["temporary_bytes"] == 88984514\n'
         '    assert admission["component_allowance"]["llm"]["usage_receipt_bytes"] == 16384\n'
         '    assert sum(admission["component_allowance"]["llm"].values()) == raw["llm"]'),
    )
    for before, after in replacements:
        assert source.count(before) == 1, before
        source = source.replace(before, after)
    exec(compile(source, str(base), "exec"), {"__file__": __file__, "__name__": "__main__"})


if __name__ == "__main__":
    main()
