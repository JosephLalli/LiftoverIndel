"""Indel-aware liftover of variants between genome references.

A binding to the liftover_indels C library, which is the same engine the command
line tool uses. There is nothing to compile: this loads the shared library through
ctypes.

    from liftover_indels import LiftOver

    with LiftOver("chm13v2-grch38.chain", "grch38-chm13v2.sort.bcf",
                  "GRCh38.fasta", contigs=["chr21"]) as lo:
        r = lo.lift("chr21", 20000049, "C", "T")
        if r.ok:
            print(r.chrom, r.pos, r.ref, r.alt, r.flipped, r.realigned)
        else:
            print(r.status.name, r.message)

Coordinates are 0-based throughout, matching the C API and cyvcf2's ``variant.start``.
The alleles you pass are the source assembly's.

Loading the chain, the assembly differences and the target reference dominates the
cost of a lift, so build one :class:`LiftOver` and reuse it. Naming only the contigs
you need keeps the rest of the target reference out of memory.

This works at the allele level. A result with ``flipped`` set means REF and ALT were
swapped, and sample genotypes must be rewritten to match -- that rewrite needs every
record at the position, which only the caller has. For the same reason, the rule that
a position may flip only once is the caller's, through ``already_flipped``.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import enum
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

__all__ = [
    "LiftOver",
    "Lift",
    "Status",
    "LiftoverError",
    "library_path",
    "version",
]

_LIB_NAME = "libliftover_indels.so"


class Status(enum.IntEnum):
    """Outcome of a lift. The three failure kinds are the three sidecar files the
    command line tool writes."""

    OK = 0
    UNLIFTABLE = 1
    MULTIPLE_OVERLAPS = 2
    REF_MISMATCH = 3
    ERROR = -1


class LiftoverError(RuntimeError):
    """Raised for a failure that is the caller's or the library's fault -- bad
    arguments, an unreadable input, an internal error.

    An ordinary variant that does not lift is *not* an error: it comes back as a
    :class:`Lift` whose status is UNLIFTABLE, MULTIPLE_OVERLAPS or REF_MISMATCH.
    """


@dataclass(frozen=True)
class Lift:
    """The outcome of lifting one variant."""

    status: Status
    chrom: Optional[str] = None
    #: 0-based position in the target assembly.
    pos: Optional[int] = None
    ref: Optional[str] = None
    alt: Optional[str] = None
    #: REF and ALT were swapped; genotypes must be rewritten by the caller.
    #: Only meaningful when the status is OK.
    flipped: bool = False
    #: The representation came from haplotype realignment. OK only.
    realigned: bool = False
    #: Why it did not lift; None when the status is OK.
    message: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status is Status.OK


class _Options(ctypes.Structure):
    _fields_ = [
        ("realign_enabled", ctypes.c_int),
        ("realign_distance", ctypes.c_longlong),
        ("realign_flank", ctypes.c_longlong),
        ("realign_max_window", ctypes.c_longlong),
        ("threads", ctypes.c_int),
    ]


class _Result(ctypes.Structure):
    # The char* fields are c_void_p, not c_char_p, on purpose: ctypes converts a
    # c_char_p field to bytes on attribute access and the original pointer is then
    # unrecoverable, so the library could never free it. Read them with string_at
    # and let liftover_indels_result_dispose free the originals.
    _fields_ = [
        ("status", ctypes.c_int),
        ("chrom", ctypes.c_void_p),
        ("pos", ctypes.c_longlong),
        ("ref_allele", ctypes.c_void_p),
        ("alt_allele", ctypes.c_void_p),
        ("flipped", ctypes.c_int),
        ("realigned", ctypes.c_int),
        ("message", ctypes.c_void_p),
    ]


def _candidate_paths() -> Iterator[Path]:
    """Where to look for the shared library, most specific first."""
    env = os.environ.get("LIFTOVER_INDELS_LIB")
    if env:
        yield Path(env)
    here = Path(__file__).resolve().parent
    yield here / _LIB_NAME
    # Running from a source checkout: python/liftover_indels/ -> target/release/
    yield here.parent.parent / "target" / "release" / _LIB_NAME
    yield Path(sys.prefix) / "lib" / _LIB_NAME
    yield Path.home() / "usr" / "local" / "lib" / _LIB_NAME
    yield Path("/usr/local/lib") / _LIB_NAME


def library_path() -> str:
    """Absolute path of the shared library that would be loaded.

    Set ``LIFTOVER_INDELS_LIB`` to override the search.
    """
    for p in _candidate_paths():
        if p.is_file():
            return str(p)
    found = ctypes.util.find_library("liftover_indels")
    if found:
        return found
    searched = "\n  ".join(str(p) for p in _candidate_paths())
    raise LiftoverError(
        f"could not find {_LIB_NAME}. Build it with 'cargo build --release', or set "
        f"LIFTOVER_INDELS_LIB to its path. Searched:\n  {searched}"
    )


_lib = None


def _load():
    global _lib
    if _lib is not None:
        return _lib
    lib = ctypes.CDLL(library_path())

    lib.liftover_indels_version.argtypes = []
    lib.liftover_indels_version.restype = ctypes.c_char_p

    lib.liftover_indels_options_init.argtypes = [ctypes.POINTER(_Options)]
    lib.liftover_indels_options_init.restype = None

    lib.liftover_indels_result_init.argtypes = [ctypes.POINTER(_Result)]
    lib.liftover_indels_result_init.restype = None

    lib.liftover_indels_open.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.c_size_t,
        ctypes.POINTER(_Options),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.liftover_indels_open.restype = ctypes.c_void_p

    lib.liftover_indels_close.argtypes = [ctypes.c_void_p]
    lib.liftover_indels_close.restype = None

    lib.liftover_indels_lift.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_longlong,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.POINTER(_Result),
    ]
    lib.liftover_indels_lift.restype = ctypes.c_int

    lib.liftover_indels_result_dispose.argtypes = [ctypes.POINTER(_Result)]
    lib.liftover_indels_result_dispose.restype = None

    lib.liftover_indels_string_free.argtypes = [ctypes.c_void_p]
    lib.liftover_indels_string_free.restype = None

    _lib = lib
    return lib


def version() -> str:
    """Version of the loaded library."""
    return _load().liftover_indels_version().decode()


def _text(ptr) -> Optional[str]:
    if not ptr:
        return None
    return ctypes.string_at(ptr).decode()


class LiftOver:
    """A loaded liftover: chain file, assembly differences and target reference.

    :param chain: UCSC ``.over.chain`` file, optionally gzipped.
    :param ref_diffs: assembly-differences VCF/BCF, in **target** coordinates.
    :param target_fasta: target reference FASTA, optionally gzipped.
    :param contigs: restrict the load to these contigs. ``None`` loads everything,
        which for a whole human reference is several gigabytes.
    :param realign: enable haplotype realignment near assembly differences.
    :param realign_distance: how far to look for a nearby difference, in bp.
    :param realign_flank: bases added each side of the realignment window.
    :param realign_max_window: hard cap on the realignment window, in bp.
    :param threads: reader threads for the differences file.

    The instance is safe to share between threads. The underlying engine is
    immutable once loaded, and each thread gets its own result buffer.
    """

    def __init__(
        self,
        chain: str | os.PathLike,
        ref_diffs: str | os.PathLike,
        target_fasta: str | os.PathLike,
        contigs: Optional[Sequence[str]] = None,
        *,
        realign: bool = True,
        realign_distance: int = 50,
        realign_flank: int = 20,
        realign_max_window: int = 200,
        threads: int = 2,
    ) -> None:
        lib = _load()

        opts = _Options()
        lib.liftover_indels_options_init(ctypes.byref(opts))
        opts.realign_enabled = 1 if realign else 0
        opts.realign_distance = realign_distance
        opts.realign_flank = realign_flank
        opts.realign_max_window = realign_max_window
        opts.threads = threads

        names = [str(c).encode() for c in contigs] if contigs else []
        arr = (ctypes.c_char_p * len(names))(*names) if names else None

        err = ctypes.c_void_p()
        handle = lib.liftover_indels_open(
            os.fsencode(chain),
            os.fsencode(ref_diffs),
            os.fsencode(target_fasta),
            arr,
            len(names),
            ctypes.byref(opts),
            ctypes.byref(err),
        )
        if not handle:
            message = _text(err) or "liftover_indels_open failed"
            if err:
                lib.liftover_indels_string_free(err)
            raise LiftoverError(message)
        self._handle = handle
        self._lib = lib
        # The engine itself is safe to lift from concurrently, so the only thing
        # standing between this object and thread-safety is the result buffer.
        # Keeping one per thread buys reuse without sharing.
        self._local = threading.local()

    def lift(
        self,
        chrom: str,
        pos: int,
        ref: str,
        alt: str,
        already_flipped: bool = False,
    ) -> Lift:
        """Lift one variant.

        :param chrom: source contig.
        :param pos: **0-based** source position.
        :param ref: source REF allele.
        :param alt: source ALT allele. The variant must be biallelic.
        :param already_flipped: set when an earlier variant at this same position
            already flipped. A variant that would itself flip is then reported as
            REF_MISMATCH, which is what the reference implementation does.
        :raises LiftoverError: for bad arguments or an internal failure. A variant
            that simply does not lift is returned, not raised.
        """
        if self._handle is None:
            raise LiftoverError("this LiftOver has been closed")
        res = getattr(self._local, "result", None)
        if res is None:
            res = _Result()
            self._lib.liftover_indels_result_init(ctypes.byref(res))
            self._local.result = res
        status = self._lib.liftover_indels_lift(
            self._handle,
            chrom.encode(),
            pos,
            ref.encode(),
            alt.encode(),
            1 if already_flipped else 0,
            ctypes.byref(res),
        )
        try:
            if status == Status.OK:
                return Lift(
                    status=Status.OK,
                    chrom=_text(res.chrom),
                    pos=res.pos,
                    ref=_text(res.ref_allele),
                    alt=_text(res.alt_allele),
                    flipped=bool(res.flipped),
                    realigned=bool(res.realigned),
                )
            message = _text(res.message)
            if status == Status.ERROR:
                raise LiftoverError(message or "liftover_indels_lift failed")
            return Lift(status=Status(status), message=message)
        finally:
            # Frees all of the strings together. Must happen before the next lift,
            # which overwrites the struct without freeing what it held.
            self._lib.liftover_indels_result_dispose(ctypes.byref(res))

    def lift_many(
        self, variants: Iterable[tuple[str, int, str, str]]
    ) -> Iterator[Lift]:
        """Lift an iterable of ``(chrom, pos, ref, alt)`` tuples, lazily.

        A convenience over :meth:`lift`; it does not batch across the boundary, so
        it costs the same per variant.
        """
        for chrom, pos, ref, alt in variants:
            yield self.lift(chrom, pos, ref, alt)

    def close(self) -> None:
        """Release the engine. Idempotent."""
        handle = getattr(self, "_handle", None)
        self._handle = None
        lib = getattr(self, "_lib", None)
        if handle and lib is not None:
            lib.liftover_indels_close(handle)

    def __enter__(self) -> "LiftOver":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
