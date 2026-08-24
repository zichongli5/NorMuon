# NorMuon

Official implementation of **NorMuon: Making Muon more efficient and scalable** ([arXiv:2510.05491](https://arxiv.org/abs/2510.05491)).

> 🎉 **Accepted as a Spotlight at ICML 2026.**

NorMuon augments [Muon](https://github.com/KellerJordan/Muon) with a per-row second-moment normalizer (similar in spirit to Adam's second moment) applied to the orthogonalized update. The normalizer is rescaled so that the overall update norm matches Muon's, giving better-conditioned per-neuron step sizes without changing the effective learning rate.

NorMuon is also used in [karpathy/nanochat](https://github.com/karpathy/nanochat). For a fully distributed FSDP-style implementation, see our PR to the [Dion](https://github.com/microsoft/dion/pull/19) codebase. For a `modded-nanogpt` integration, see [this PR](https://github.com/KellerJordan/modded-nanogpt/pull/141).

## Installation

```
pip install git+https://github.com/zichongli5/NorMuon.git
```

`normuon.py` has no dependencies beyond PyTorch, so you can also just drop it into your project.
The optional M=512 Triton path can be installed with `pip install ".[triton]"`.

## What's in `normuon.py`

- `NorMuon` / `SingleDeviceNorMuon` — distributed (DDP) and single-GPU optimizers.
- `NorMuonWithAuxAdam` / `SingleDeviceNorMuonWithAuxAdam` — bundle NorMuon for hidden weights with AdamW for the rest. **Recommended.**

## Usage

Like Muon, NorMuon is meant for the hidden 2D weight matrices of the network. Embeddings, the classifier head, and gains/biases should be optimized with AdamW. The `WithAuxAdam` variants take care of routing for you:

```python
from normuon import NorMuonWithAuxAdam

hidden_weights      = [p for p in model.body.parameters() if p.ndim >= 2]
hidden_gains_biases = [p for p in model.body.parameters() if p.ndim < 2]
nonhidden_params    = [*model.head.parameters(), *model.embed.parameters()]

param_groups = [
    dict(params=hidden_weights, use_muon=True,
         lr=0.02, momentum=0.95, beta2=0.95, weight_decay=0.01),
    dict(params=hidden_gains_biases + nonhidden_params, use_muon=False,
         lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01),
]
optimizer = NorMuonWithAuxAdam(param_groups)
```

CUDA workloads whose hidden matrices have a 512-element short dimension can
opt into an all-Triton implementation of the same five-step Keller-Jordan
Newton-Schulz iteration:

```python
from normuon import SingleDeviceNorMuonWithAuxAdam

param_groups[0]["use_triton"] = True
optimizer = SingleDeviceNorMuonWithAuxAdam(param_groups)
```

The flag defaults to `False`. Shapes other than M=512 continue to use the
original Torch implementation; requesting Triton does not change their path.

The defaults (`lr=0.02`, `momentum=0.95`, `beta2=0.95`) match Muon's; in our experiments only `lr` and `weight_decay` typically need tuning, and the same values that work for Muon are a good starting point.

## Citation

```bibtex
@misc{li2025normuon,
  title         = {NorMuon: Making Muon more efficient and scalable},
  author        = {Li, Zichong and Liu, Liming and Liang, Chen and Chen, Weizhu and Zhao, Tuo},
  year          = {2025},
  eprint        = {2510.05491},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```

## Acknowledgements

Built directly on [Keller Jordan's Muon](https://github.com/KellerJordan/Muon); the Newton–Schulz iteration and the distributed update-sharding pattern are taken from there with minimal changes.
