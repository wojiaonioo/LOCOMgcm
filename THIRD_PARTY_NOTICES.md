# Third-party notices

## MITgcm

The MITgcm-derived numerical methods, source overrides and case recipes in
this release originate from MITgcm and MITgcm-AD. The bundled overrides and
recipes are located in `verification/adjoint/*/mit_ad/` and
`experiments/mitcases/references/`.
Their original copyright and MIT permission notice is reproduced below.
Existing notices in individual files remain applicable.

The source for this notice is the upstream MITgcm `LICENSE.txt`, including
the MITgcm-AD and d861cd50 reference revisions.

```text
Copyright (c) 2018 MITgcm Developers and Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Runtime dependencies and data

The Python and native libraries named in
`requirements-runtime.txt` are installed separately and retain
their own licenses. Upstream MITgcm trees, Tapenade, compilers and input
datasets also retain their original terms. The LOCOMgcm license (GPL-3.0-only) covers
LOCOMgcm-owned software and documentation; data access and reuse terms are
recorded with the accompanying reproducibility materials.
