# Release verification

Release: 1.0.0. Verification date: 2026-10-06.

## Executed checks

- Python 3.12 CPU environment check with PyTorch 2.6.0+cpu.
- Representation suite: 38 tests passed, including numerical operators, data auditing, synthetic training, forecasting, compatibility, and resumption.
- Posthoc suite: 18 tests passed, including complete synthetic validation analysis and read-only source checks.
- Frozen final-evaluation suite: 21 tests passed, including synthetic training-to-freeze-to-test evaluation, no fitting during final evaluation, held-out field-access guards during freeze, and tamper rejection.
- Publication suite: 24 existing tests passed. A further table-legend and asset-numbering regression check passed after public metadata cleanup, giving 25 publication tests in this release. Three targeted publication integration tests also passed after cleanup.
- Orchestration suite: 5 tests passed, including configuration identity preservation, isolated module imports, a non-executing end-to-end command plan, and complete-budget enforcement.
- Verified the source bytes of all 29 Python modules in the final-evaluation implementation and its representation vendor against the frozen v0.4.0 package. No evaluator code was changed.
- Verified the 27 posthoc and vendor Python modules against v0.3.0 and the representation source fingerprint against v0.2.0.
- Resolved an upstream revision and matched its complete 82-file content-hash catalog against the recorded study data hashes. The pinned catalog contains 55,809,409,024 bytes of source data.
- Regenerated the 15 numerical publication CSVs from the actual frozen test summary archive and compared them with the v1.4.0 publication outputs: all numerical tables were unchanged. All five table captions were emitted.
- Checked the public source release for private server paths, manuscript-generation modules, generated research images, manuscript drafts, caches, and deprecated package outputs.

The complete suite contains 107 tests: 38 representation, 18 posthoc, 21 final-evaluation, 25 publication, and 5 orchestration tests. Synthetic fixtures are labeled as software tests and are not research evidence.

## Execution limits

A fresh full-data GPU training run was not performed during release assembly. The complete 52 GiB dataset was not redownloaded. Full real-data results are inherited from the recorded study and the frozen final-test summary, rather than certified as a new independent replication. The public runner connects the original study modules and the verified paper configurations; its full-resolution execution must be performed on suitable research hardware.

The CUDA environment, container build, and hosted GitHub Actions execution were not exercised during this release verification. Their installation and CI definitions are provided for subsequent execution. Exact checkpoint or result hashes after retraining may differ across hardware and numerical libraries despite the deterministic settings. Every new run records its own identities.

## Source identities

| Component | Scientific source fingerprint |
|---|---|
| Representation v0.2.0 | `9e85ef699942c8679b712c0af82ab7d8bc3bc4c50897758eb3dbd8809cf5475b` |
| Posthoc v0.3.0 | `8fd9d61549bb41840f8bc1bd06520beec9124b4d575f0b0187c89af56b7957c0` |
| Final evaluation v0.4.0 | `6e2b25cfb2f5543799f5c6d35ce3d8dcba971b974e4038affcdf233ed0984d2d` |

`provenance/source_manifest.json` records component and orchestration file checksums. Publication v1.4.1 changes public metadata and supplies standalone table captions; numerical estimators, plotted values, and frozen evaluation logic are unchanged.
