"""Tests for the Python binding that need no reference data.

The struct-layout tests are the important ones: they pin the ctypes declarations
against the layout the C header and the Rust library agree on, so a change to the
ABI cannot drift past the binding silently.

Run with:  python3 -m unittest discover -s python/tests
"""

import ctypes
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import liftover_indels as li
from liftover_indels import _Options, _Result


def library_available():
    try:
        li.library_path()
        return True
    except li.LiftoverError:
        return False


AVAILABLE = library_available()
needs_lib = unittest.skipUnless(AVAILABLE, "libliftover_indels.so not built")


class LayoutTests(unittest.TestCase):
    """Values checked against examples/abi_layout.rs and examples/cpp/abi_layout.cpp."""

    def test_options_layout(self):
        self.assertEqual(ctypes.sizeof(_Options), 40)
        self.assertEqual(_Options.realign_enabled.offset, 0)
        self.assertEqual(_Options.realign_distance.offset, 8)
        self.assertEqual(_Options.realign_flank.offset, 16)
        self.assertEqual(_Options.realign_max_window.offset, 24)
        self.assertEqual(_Options.threads.offset, 32)

    def test_result_layout(self):
        self.assertEqual(ctypes.sizeof(_Result), 56)
        self.assertEqual(_Result.status.offset, 0)
        self.assertEqual(_Result.chrom.offset, 8)
        self.assertEqual(_Result.pos.offset, 16)
        self.assertEqual(_Result.ref_allele.offset, 24)
        self.assertEqual(_Result.alt_allele.offset, 32)
        self.assertEqual(_Result.flipped.offset, 40)
        self.assertEqual(_Result.realigned.offset, 44)
        self.assertEqual(_Result.message.offset, 48)

    def test_string_fields_keep_their_pointer(self):
        # c_char_p would convert to bytes on access and lose the pointer, leaving
        # the library unable to free it. These must stay raw pointers.
        declared = dict(_Result._fields_)
        for name in ("chrom", "ref_allele", "alt_allele", "message"):
            self.assertIs(declared[name], ctypes.c_void_p, f"{name} must be c_void_p")
        # And behaviourally: reading one yields an address, not bytes.
        r = _Result()
        buf = ctypes.create_string_buffer(b"chr21")
        r.chrom = ctypes.cast(buf, ctypes.c_void_p).value
        self.assertIsInstance(r.chrom, int)
        self.assertEqual(ctypes.string_at(r.chrom), b"chr21")

    def test_status_values_match_the_c_constants(self):
        self.assertEqual(int(li.Status.OK), 0)
        self.assertEqual(int(li.Status.UNLIFTABLE), 1)
        self.assertEqual(int(li.Status.MULTIPLE_OVERLAPS), 2)
        self.assertEqual(int(li.Status.REF_MISMATCH), 3)
        self.assertEqual(int(li.Status.ERROR), -1)


class LiftDataclassTests(unittest.TestCase):
    def test_ok_property(self):
        self.assertTrue(li.Lift(status=li.Status.OK).ok)
        self.assertFalse(li.Lift(status=li.Status.UNLIFTABLE).ok)

    def test_a_failed_lift_carries_no_coordinates(self):
        r = li.Lift(status=li.Status.REF_MISMATCH, message="nope")
        self.assertIsNone(r.chrom)
        self.assertIsNone(r.pos)
        self.assertEqual(r.message, "nope")


@needs_lib
class LibraryTests(unittest.TestCase):
    def test_version_is_reported(self):
        v = li.version()
        self.assertRegex(v, r"^\d+\.\d+\.\d+")

    def test_library_path_is_a_file(self):
        self.assertTrue(os.path.isfile(li.library_path()))

    def test_env_var_overrides_the_search(self):
        real = li.library_path()
        old = os.environ.get("LIFTOVER_INDELS_LIB")
        os.environ["LIFTOVER_INDELS_LIB"] = real
        try:
            self.assertEqual(li.library_path(), real)
        finally:
            if old is None:
                del os.environ["LIFTOVER_INDELS_LIB"]
            else:
                os.environ["LIFTOVER_INDELS_LIB"] = old

    def test_opening_a_missing_chain_raises_rather_than_crashing(self):
        with self.assertRaises(li.LiftoverError) as cm:
            li.LiftOver("/nonexistent.chain", "/nonexistent.bcf", "/nonexistent.fa")
        # The message should name what actually went wrong.
        self.assertTrue(str(cm.exception))

    def test_lifting_after_close_raises(self):
        lo = li.LiftOver.__new__(li.LiftOver)
        lo._handle = None
        with self.assertRaises(li.LiftoverError):
            lo.lift("chr1", 0, "A", "T")


if __name__ == "__main__":
    unittest.main()
