//! Print how rust-htslib decodes each sample's genotype, to compare against
//! cyvcf2's `genotype.array()`.
//!
//! Usage: gt_dump <vcf/bcf>

use rust_htslib::bcf::{Read, Reader};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut reader = Reader::from_path(&args[1]).expect("open");
    let header = reader.header().clone();
    let n = header.sample_count() as usize;
    let names: Vec<String> = (0..n)
        .map(|i| String::from_utf8_lossy(header.samples()[i]).into_owned())
        .collect();

    let mut rec = reader.empty_record();
    while let Some(r) = reader.read(&mut rec) {
        r.expect("read");
        println!("POS={}", rec.pos() + 1);
        let gts = rec.genotypes().expect("genotypes");
        for s in 0..n {
            let g = gts.get(s);
            println!("  {:13} len={} {:?}", names[s], g.len(), &*g);
        }
        break;
    }
}
