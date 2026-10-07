# Third-party data and dependencies

The repository downloads the active-matter simulation dataset from the Polymathic AI official mirror. Dataset files are not redistributed in this source release. The dataset card identifies the license as Creative Commons Attribution 4.0 (CC BY 4.0). Retain dataset attribution and comply with that license when redistributing downloaded data or derived data products.

- Dataset card: https://huggingface.co/datasets/polymathic-ai/active_matter
- Dataset documentation: https://polymathic-ai.org/the_well/datasets/active_matter/
- Dataset scientific reference: Maddu, Weady, and Shelley (2024), *Learning fast, accurate, and stable closures of a kinetic theory of an active fluid*, Journal of Computational Physics 504, 112869.

The MIT license applies to the study code and documentation. Third-party Python packages and FFmpeg retain their respective licenses. Dependencies are installed from their distributors rather than included as vendored third-party source. The `vendor/active_matter_repr` directories contain the study's own reference representation implementation, duplicated to preserve isolated release fingerprints.
