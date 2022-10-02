## Changelog

## v2.0.2
- Fixed installation issues.

### v2.0.1
- Updated DC2 neural network weights.

### v2.0
- Retrained with the updated dataset based on *[SamCC-Turbo](https://github.com/labstructbioinf/samcc_turbo)* labels.
- Faster inference time by applying *[SeqVec](https://github.com/rostlab/SeqVec)* embeddings instead of *psiblast* profiles.
- Heptad register prediction (*a* and *d* core positions).
- No maximum sequence length limit.
- Convenient interface for using *DeepCoil* within python scripts.
- Automated peak detection for improved output readability.
- Simplified installation with *pip*.

### v1.1
Added output filtering options <code>-min_residue_score</code> and <code>-min_segment_length</code>.

### v1.0
Initial release - corresponds to the version presented in the original paper [DOI:10.1093/bioinformatics/bty1062](https://doi.org/10.1093/bioinformatics/bty1062)

