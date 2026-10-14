# Reproducibility notes

## Scope of validation

This maintenance pass fixes launcher and utility defects in a copy of the historical research repository. It does not reproduce the experiments or establish compatibility with a current training stack.

The regression suite uses Python's standard library. Launcher tests substitute a small executable child for `src/main.py`; they check actual subprocess behavior without training. Utility tests check integer timeout parsing, client-size metric recording, and cleanup of temporary model/ZIP files, including a failed model save.

```bash
python -B -S -m unittest discover -s tests -v
```

These tests were run with Python 3.14.4 on macOS. That is a test-runtime observation, not a supported training environment. TensorFlow and Flower were not installed in that environment. Installation, dataset loading, HDF5 model loading, end-to-end Flower/Ray execution, GPU behavior, and numerical results remain unverified.

## Historical environment

`requirements.txt` is retained unchanged. Its main pins include:

| Component | Version |
| --- | --- |
| TensorFlow GPU | 2.10.1 |
| Keras | 2.10.0 |
| Flower | 1.1.0 |
| Ray | 2.1.0 |
| TensorFlow Model Optimization | 0.7.3 |
| TensorFlow Addons | 0.19.0 |
| NumPy | 1.23.5 |

PyPI's release files for [TensorFlow GPU 2.10.1](https://pypi.org/project/tensorflow-gpu/2.10.1/#files) include CPython 3.7–3.10 wheels for Linux x86-64 and Windows x86-64, and no macOS wheel. [Ray 2.1.0](https://pypi.org/project/ray/2.1.0/#files) also has a CPython 3.10 Linux x86-64 wheel. Linux x86-64 with Python 3.10 is therefore a candidate for reconstructing the environment, not a validated complete dependency solution. Wheel availability does not establish CUDA/cuDNN compatibility or successful execution.

Do not treat installing the unchanged requirements on the latest Python, TensorFlow, or Keras as supported. Record Python, OS, GPU, driver, CUDA/cuDNN, and resolved package versions when qualifying a training environment. Dependency upgrades should be evaluated separately against the original stack and results.

The launcher requests a nonzero GPU fraction by default (`gpu_usage = 0.08` in `configs.py`). CPU-only experiments require explicit resource configuration and have not been tested. GPU visibility is left to the caller and Ray; the ineffective historical `CUDA_VSIBLE_DEVICES` writes were removed rather than forcing workers onto device 0.

## Data and assets

| Input | Loader expectation | Included? |
| --- | --- | --- |
| CIFAR-10 | Keras `cifar10.load_data()`, which can download/cache the dataset | No |
| StyleGAN images | Image files under `$TFDS_DATA_DIR/raw/stylegan_oriented/` | No |
| Speech Commands | TensorFlow Datasets `speech_commands` | No |
| Speech training labels | `$TFDS_DATA_DIR/speech_commands/train_labels.npy`, aligned to the loader's training order | No |
| LibriSpeech audio | Audio files under `$TFDS_DATA_DIR/raw/librispeech/` | No |
| ResNet20 initialization | `assets/init_models/resnet20_cifar10_random_1.h5` | Yes |
| YAMNet initialization | `assets/init_models/yamnet_spcm_random_1.h5` | Yes |

Set `TFDS_DATA_DIR` to an absolute dataset-root path for the external loaders. The source does not supply the StyleGAN generation configuration, an exact external-data manifest, or the Speech Commands label-array preparation procedure. Merely placing arbitrary images/audio in these directories does not reproduce the paper's data.

The high-level launcher only accepts `cifar10`. Speech code and its initialization asset remain available for research inspection, but are not advertised as an independently verified command-line workflow.

The two `.h5` files are initialization assets, not published trained checkpoints. Their file format was inspected; their contents were not loaded by TensorFlow in this pass. `assets/saved_models/` and `assets/results/` originally contain directory placeholders. Preserve run arguments, random seed, data provenance and environment details alongside any new results.

## Preserved research behavior

- **Baseline epochs:** `run.py --methods fedavg` passes `method='fedavg'` to `Params`. The historical `Params.__post_init__` adjustment is keyed to `'standard'`, so the launcher retains five ordinary local epochs for FedAvg, while `client`/`fedcompress` retain five ordinary plus five compression epochs. The `'standard'` configuration would instead combine the epochs into ten ordinary epochs, but the launcher does not select it. This discrepancy is documented rather than silently changing baseline comparisons.
- **Compression schedule:** `set_compression_rnds` retains its original range calculation. For `compression_step=3, num_rounds=10`, it returns `[0, 3, 6, 10]`, skipping round 9. The default step of 1 is retained. A scheduling correction requires a separate change and experiment review.
- **Run defaults:** `run.py` explicitly sets 100 CIFAR-10 rounds, overriding the 30-round dataclass default. `src/main.py` has its own low-level defaults; use the high-level launcher for the commands in the README. Direct invocation requires reviewing those defaults and supplying appropriate output/initialization paths.
- **Size metric:** despite their `get_gzipped_*` names, the helpers retain the original ZIP/DEFLATE archive measurement in decimal kilobytes, including the original archive-member naming behavior. Cleanup now closes and removes temporary files; it does not replace the compression metric. Corrected key forwarding makes the existing mean and per-client size measurements appear in recorded metrics.

Architectures, clustering decisions, distillation objectives, dataset partitioning, and historical dependency pins are unchanged. No new accuracy, compression-ratio, communication-cost, or inference-speed claim is made by these maintenance changes.

## Citation metadata

The paper title, author order, venue, year, page range and DOI in `CITATION.cff` were checked against the [Crossref record](https://api.crossref.org/works/10.1109/ICASSP48485.2024.10447174) and the [arXiv record](https://arxiv.org/abs/2401.14211). The paper's publication metadata does not establish a software license; no license has been added.
