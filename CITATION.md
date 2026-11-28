# Citing LOCOMgcm

The article that describes the model, **LOCOMgcm: A Differentiable,
GPU-Accelerated Ocean General Circulation Model in PyTorch** (Tang, Xu, Hong
and Zhou), has
not been published yet. Until it is, cite the software and state the Git
commit you used:

> Tang, H., Xu, H., Hong, B., and Zhou, W. (2026). LOCOMgcm: a differentiable
> ocean general circulation model in PyTorch (version 0.1.0) [Software].
> https://github.com/wojiaonioo/LOCOMgcm

```bibtex
@software{locomgcm,
  author  = {Tang, Haoyang and Xu, Hongzhou and Hong, Bo and Zhou, Wei},
  title   = {{LOCOMgcm}: a differentiable ocean general circulation model in {PyTorch}},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/wojiaonioo/LOCOMgcm},
  note    = {Git commit <hash of the commit you used>}
}
```

`git rev-parse HEAD` in your checkout prints the commit hash. The same
metadata is in [CITATION.cff](CITATION.cff), which GitHub shows under "Cite
this repository".

For MITgcm comparisons, also cite the upstream MITgcm revision used by the
case. Copyright and license notices for MITgcm-derived code are retained in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
