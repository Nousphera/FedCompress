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

## Small participation regression

A focused regression now constructs the actual server/strategy and checks that
`participation=0.01` with two available clients schedules one fit instruction.
Before the minimum-client floor, it scheduled zero. FedLN covers both its base
server and NA-FedAvg override; FedCompress covers its compressed strategy.
These focused checks passed after the repair.

## Recovered dataset adapters

Five generated-data adapter tests passed (7.60 s), including real YAMNet
optimization for SpeechCommands/VoxForge and decoded WAV scaling. Two actual
two-client training tests passed (39.63 s): PathMNIST and VoxForge, each with
local clustering, stock Flower messages, server self-distillation and model
export. The CIFAR-100 class-count fixture uses generated arrays.

Optional TFDS 4.9.4 / tensorflow-metadata 1.15.0 imports and builder metadata
were checked in an isolated dependency directory against the maintained base:
SpeechCommands 0.0.3 has 12 classes and VoxForge 1.0.0 has six; both expose
`(audio, label)`. `pip check` passed for that combined import environment.
No full corpus download or TFDS construction was run.

The combined suite, run from the repository root, passed **26 tests in
76.48 s** after restoration. Compiler and whitespace checks also passed.
