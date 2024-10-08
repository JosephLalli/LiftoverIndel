# LiftoverIndel
Tool to liftover variants between references in an indel-aware manner. Importantly, this tool identifies variants that overlap a region of the target reference that has an indel relative to the origin reference and incorporates that indel into the lifted variant 

Currently in minimal-viable-product mode, but should work for most CHM13/GRCh38 reference liftovers. Please note that 

### Please note:
- Input variants must be biallelic.
- Output variants should be left-aligned and sorted.
- BCF or VCF files can both be read, but using bcf files allows for a *much* quicker liftover run. I encourage the conversion of all files, even the reference liftover vcfs, to bcf format for this purpose. 
- Output format should be autodetected from the provided output file extension.
- **Only genotypes** are lifted over; while some of the other info/format fields may be correct, these fields are explicitly not touched during liftover. Some/most non-GT fields **will be wrong**. If lifting these fields is important to you, please post an issue and I will do my best to add this feature.

pip installation/requirements.txt is not set up, but requirements are:
- pyLiftover
- tqdm
- intervaltree
- cyvcf2
- Bio

Once requirements are installed, simply run the following command:
```
python3 liftover_indels.py \
    vcf_to_lift.bcf \
    vcf_of_assembly_differences.bcf \ # This file needs to be in target assembly coordinates.
    lifted_over_output.bcf \  #This is stdout by default.
    chainfile.chain \
    target_fasta.fasta
```

    
Assembly differences vcf can either be generated by you, or in the CHM13/GRCh38 liftover context it can be obtained from the [HPRC AWS bucket](https://s3-us-west-2.amazonaws.com/human-pangenomics/index.html?prefix=T2T/CHM13/assemblies/chain/v1_nflo/).
<br>
[GRCh38-CHM13](https://s3-us-west-2.amazonaws.com/human-pangenomics/T2T/CHM13/assemblies/chain/v1_nflo/grch38-chm13v2.sort.vcf.gz) vcf.gz file (GRCh38 coordinates). Use when lifting CHM13 -> GRCh38 coordinates.
<br>
[CHM13-GRCh38](https://s3-us-west-2.amazonaws.com/human-pangenomics/T2T/CHM13/assemblies/chain/v1_nflo/chm13v2-grch38.sort.vcf.gz) vcf.gz file(CHM13 coordinates). Use when lifting GRCh38 -> CHM13 coordinates.