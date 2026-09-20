//! Index of assembly differences between the source and target references.
//!
//! The differences are supplied in *target* coordinates. Queries are half-open
//! range overlaps, matching Python's `intervaltree` slice: an interval is
//! returned when `interval.start < end && interval.end > start`.

use std::collections::HashMap;

/// One record from the assembly-differences VCF.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RefDiff {
    /// 0-based inclusive start in target coordinates.
    pub start: i64,
    /// 0-based exclusive end in target coordinates.
    pub end: i64,
    pub ref_allele: String,
    pub alt_allele: String,
}

impl RefDiff {
    pub fn is_indel(&self) -> bool {
        self.ref_allele.len() != self.alt_allele.len()
    }

    /// Distance from this difference to the half-open span `[start, end)`;
    /// zero when they touch or overlap.
    pub fn distance_to_span(&self, start: i64, end: i64) -> i64 {
        if self.end <= start {
            start - self.end
        } else if self.start >= end {
            self.start - end
        } else {
            0
        }
    }
}

struct ContigIndex {
    diffs: Vec<RefDiff>,
    max_end: Vec<i64>,
}

#[derive(Default)]
pub struct RefDiffIndex {
    contigs: HashMap<String, ContigIndex>,
}

impl RefDiffIndex {
    pub fn build(per_contig: HashMap<String, Vec<RefDiff>>) -> Self {
        let mut contigs = HashMap::with_capacity(per_contig.len());
        for (name, mut diffs) in per_contig {
            diffs.sort_by_key(|d| (d.start, d.end));
            let mut max_end = Vec::with_capacity(diffs.len());
            let mut running = i64::MIN;
            for d in &diffs {
                running = running.max(d.end);
                max_end.push(running);
            }
            contigs.insert(name, ContigIndex { diffs, max_end });
        }
        RefDiffIndex { contigs }
    }

    pub fn has_contig(&self, chrom: &str) -> bool {
        self.contigs.contains_key(chrom)
    }

    pub fn contig_count(&self) -> usize {
        self.contigs.len()
    }

    pub fn total_diffs(&self) -> usize {
        self.contigs.values().map(|c| c.diffs.len()).sum()
    }

    /// Differences overlapping `[start, end)`, in ascending `(start, end)` order.
    ///
    /// An empty or inverted range yields nothing, as a null slice does in Python.
    pub fn overlapping(&self, chrom: &str, start: i64, end: i64) -> Vec<&RefDiff> {
        let Some(contig) = self.contigs.get(chrom) else {
            return Vec::new();
        };
        if start >= end {
            return Vec::new();
        }
        let hi = contig.diffs.partition_point(|d| d.start < end);
        let mut out: Vec<&RefDiff> = Vec::new();
        let mut i = hi;
        while i > 0 {
            if contig.max_end[i - 1] <= start {
                break;
            }
            let d = &contig.diffs[i - 1];
            if d.start < end && d.end > start {
                out.push(d);
            }
            i -= 1;
        }
        out.reverse();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn diff(start: i64, r: &str, a: &str) -> RefDiff {
        RefDiff {
            start,
            end: start + r.len() as i64,
            ref_allele: r.to_string(),
            alt_allele: a.to_string(),
        }
    }

    fn index() -> RefDiffIndex {
        let mut m = HashMap::new();
        m.insert(
            "chr1".to_string(),
            vec![
                diff(10, "A", "G"),
                diff(20, "ACGT", "A"),
                diff(100, "T", "TTTT"),
            ],
        );
        RefDiffIndex::build(m)
    }

    #[test]
    fn overlap_is_half_open_on_both_sides() {
        let idx = index();
        // The SNV at 10 occupies [10, 11).
        assert_eq!(idx.overlapping("chr1", 10, 11).len(), 1);
        assert_eq!(idx.overlapping("chr1", 11, 12).len(), 0);
        assert_eq!(idx.overlapping("chr1", 9, 10).len(), 0);
        // A span ending exactly at the start does not overlap.
        assert_eq!(idx.overlapping("chr1", 0, 10).len(), 0);
        assert_eq!(idx.overlapping("chr1", 0, 11).len(), 1);
    }

    #[test]
    fn a_deletion_spans_its_whole_ref_allele() {
        let idx = index();
        for p in 20..24 {
            assert_eq!(idx.overlapping("chr1", p, p + 1).len(), 1, "pos {p}");
        }
        assert_eq!(idx.overlapping("chr1", 24, 25).len(), 0);
    }

    #[test]
    fn empty_or_inverted_ranges_return_nothing() {
        let idx = index();
        assert!(idx.overlapping("chr1", 10, 10).is_empty());
        assert!(idx.overlapping("chr1", 12, 10).is_empty());
    }

    #[test]
    fn unknown_contig_is_empty() {
        assert!(index().overlapping("chrZ", 0, 1000).is_empty());
    }

    #[test]
    fn wide_range_returns_all_in_order() {
        let idx = index();
        let got = idx.overlapping("chr1", 0, 1000);
        assert_eq!(got.len(), 3);
        assert_eq!(got[0].start, 10);
        assert_eq!(got[2].start, 100);
    }

    #[test]
    fn distance_to_span_is_zero_when_overlapping() {
        let d = diff(20, "ACGT", "A");
        assert_eq!(d.distance_to_span(22, 23), 0);
        // [10,15) ends before the diff starts at 20.
        assert_eq!(d.distance_to_span(10, 15), 5);
        // [30,35) starts after the diff ends at 24.
        assert_eq!(d.distance_to_span(30, 35), 6);
    }

    #[test]
    fn indel_detection_follows_allele_lengths() {
        assert!(!diff(1, "A", "G").is_indel());
        assert!(diff(1, "ACGT", "A").is_indel());
        assert!(diff(1, "A", "ACGT").is_indel());
    }
}
