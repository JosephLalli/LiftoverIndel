import ast
from pathlib import Path
import unittest


SOURCE = Path(__file__).parents[1] / "liftover_indels.py"


def load_liftover_functions():
    tree = ast.parse(SOURCE.read_text())
    wanted = {
        "Unliftable",
        "rev_comp",
        "add_original_info_tags",
        "trim_identical_suffix",
        "preprocess_variant",
        "perform_clean_liftover",
    }
    selected = [node for node in tree.body if getattr(node, "name", None) in wanted]
    namespace = {"target_genome_seq": {}}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


class FakeVariant:
    def __init__(self, ref, alt, pos=101):
        self.CHROM = "source"
        self.REF = ref
        self.ALT = [alt]
        self.POS = pos
        self.INFO = {}
        self.ID = None

    @property
    def start(self):
        return self.POS - 1

    @property
    def end(self):
        return self.start + len(self.REF)

    @property
    def is_snp(self):
        return len(self.REF) == len(self.ALT[0]) == 1

    def set_pos(self, start):
        self.POS = start + 1


class FakeLiftOver:
    def __init__(self, strand="+"):
        self.strand = strand
        self.calls = []

    def convert_coordinate(self, chrom, pos):
        self.calls.append((chrom, pos))
        return [("target", pos + 100, self.strand)]


class SuffixPreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_liftover_functions()

    def test_trim_identical_suffix_keeps_one_base_and_difference(self):
        trim = self.module["trim_identical_suffix"]
        self.assertEqual(trim("AC", "CC"), ("A", "C"))
        self.assertEqual(trim("CAT", "TAT"), ("C", "T"))
        for ref, alt in (trim("AC", "CC"), trim("CAT", "TAT")):
            self.assertTrue(ref and alt)
            self.assertNotEqual(ref, alt)

    def test_preprocess_variant_trims_both_regressions_without_moving_position(self):
        preprocess = self.module["preprocess_variant"]
        for original_ref, original_alt, expected_ref, expected_alt in (
            ("AC", "CC", "A", "C"),
            ("CAT", "TAT", "C", "T"),
        ):
            with self.subTest(original_ref=original_ref, original_alt=original_alt):
                var = FakeVariant(original_ref, original_alt, pos=101)
                preprocess(var)
                self.assertEqual((var.REF, var.ALT), (expected_ref, [expected_alt]))
                self.assertEqual(var.POS, 101)
                self.assertNotEqual(var.REF, var.ALT[0])

    def test_preprocessing_runs_before_coordinate_conversion_and_preserves_provenance(self):
        perform = self.module["perform_clean_liftover"]
        for original_ref, original_alt, expected_ref, expected_alt in (
            ("AC", "CC", "A", "C"),
            ("CAT", "TAT", "C", "T"),
        ):
            with self.subTest(original_ref=original_ref, original_alt=original_alt):
                var = FakeVariant(original_ref, original_alt)
                liftover = FakeLiftOver()
                perform(var, liftover)
                self.assertEqual((var.REF, var.ALT), (expected_ref, [expected_alt]))
                self.assertNotEqual(var.REF, var.ALT[0])
                self.assertEqual(liftover.calls, [("source", 100), ("source", 101)])
                self.assertEqual(var.INFO["Original_REF"], original_ref)
                self.assertEqual(var.INFO["Original_ALT"], original_alt)
                self.assertEqual(var.INFO["SRC_REF_ALT"], f"{original_ref},{original_alt}")

    def test_preprocessed_equal_length_allele_stays_distinct_on_reverse_strand(self):
        perform = self.module["perform_clean_liftover"]
        var = FakeVariant("AC", "CC")
        perform(var, FakeLiftOver(strand="-"))
        self.assertEqual((var.REF, var.ALT), ("T", ["G"]))
        self.assertNotEqual(var.REF, var.ALT[0])


if __name__ == "__main__":
    unittest.main()
