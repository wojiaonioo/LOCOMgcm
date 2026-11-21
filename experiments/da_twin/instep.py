"""Central differences of J with respect to an in-step momentum network."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch

from .momentum import MomentumMLP
from .run import specs, open_window, write_json


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--windows', type=Path, required=True)
    p.add_argument('--select', default='2021-12-27')
    p.add_argument('--bstd', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--eager', action='store_true')
    p.add_argument('--smoke', action='store_true', help='Check h=2^-17 only, retaining 144 model steps')
    p.add_argument('--device-gib', type=float, default=32)
    p.add_argument('--segment', type=int, default=1)
    args = p.parse_args(argv)
    args.kpp = 'lagged'
    row, = [r for r in specs(args.windows) if r['name'] == args.select]
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.out.mkdir(parents=True, exist_ok=True)
    result = dict(steps=144, h_exponents=[17] if args.smoke else list(range(31)), arms={})
    for arm in ('nonet', 'net'):
        law = MomentumMLP(20261025).cuda() if arm == 'net' else None
        w = open_window(row, args, law)
        try:
            zero = np.zeros(w.C.size)
            w.prob.value(zero, 36)
            w.prob.value_grad(zero, 36)
            timings = []
            for k in range(2 if args.smoke else 4):
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                J, _, _ = w.prob.value_grad(zero, 144)
                timings.append(dict(seconds=time.perf_counter()-start,
                    peak_bytes=torch.cuda.max_memory_allocated(), J=J))
            Jforward, _ = w.prob.value(zero, 144)
            record = dict(timings=timings[1:], J=J, J_forward=Jforward)
            if law is not None:
                theta = law.theta.detach().clone()
                gradient = law.theta.grad.detach().cpu().clone()
                np.save(args.out / 'gradient.npy', gradient.numpy())
                record['parameters'] = theta.numel()
                record['directions'] = []
                for seed in (20261101, 20261102):
                    d = torch.randn(theta.numel(), generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
                    d /= d.norm()
                    ad = float(gradient @ d)
                    if ad == 0 or not np.isfinite(ad):
                        raise FloatingPointError('Directional derivative is zero or non-finite')
                    ladder = []
                    try:
                        for k in result['h_exponents']:
                            h = 2.0**-k
                            values = []
                            for sign in (1, -1):
                                with torch.no_grad():
                                    law.theta.copy_(theta + sign * h * d.cuda())
                                values.append(w.prob.value(zero, 144)[0])
                            cd = (values[0] - values[1]) / (2*h)
                            ladder.append(dict(h=h, central=cd, ad=ad,
                                relative_difference=abs(cd-ad)/abs(ad),
                                rounding_ratio=abs(Jforward-J)/(h*abs(ad))))
                    finally:
                        with torch.no_grad():
                            law.theta.copy_(theta)
                    record['directions'].append(dict(seed=seed, ladder=ladder))
            result['arms'][arm] = record
            write_json(args.out / 'result.json', result)
        finally:
            w.prob.close()


if __name__ == '__main__':
    main()
