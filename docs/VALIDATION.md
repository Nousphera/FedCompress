# Validation — 2026-10-15

Python 3.11 / macOS ARM64 / CPU, TensorFlow 2.15.1, TFMOT 0.7.5,
Flower 1.13.1 and NumPy 1.26.4:

- 18 tests passed, including a real two-round, two-client FedCompress run with
  convolution clustering, OOD self-distillation, score feedback, a bounded
  cluster increase and standard Flower protobuf serialization.
- Numerical tests check effective-rank degeneracy, weighted client scores,
  interval scheduling, score-only plateau decisions, centroid cardinality,
  unchanged teacher parameters and lossless compressed transport.
- Separate one-round FedAvg and client-clustering CPU smokes passed.
- Legacy launcher/utility regressions continue to pass.

The first four method regression fixtures failed before their fixes. No paper
accuracy/CCR/MCR table, external StyleGAN corpus, GPU/Ray run or hardware
inference benchmark was reproduced. Synthetic fixtures are not method efficacy
measurements. The configured Linux CI has not been observed remotely.
