"""Prepare cases, twin initial states, trajectories, observations and manifests."""
import argparse
import datetime
import json
from pathlib import Path
import re
import shutil

import numpy as np
import yaml

from . import input_tools as it


def copy_preparation(src, dst):
    """Copy prepared inputs into a new output directory before replacing state."""
    if dst.exists():
        raise FileExistsError(dst)
    shutil.copytree(src, dst, symlinks=False)
    manifest = dst / 'manifest.yaml'
    if manifest.exists():
        data = yaml.safe_load(manifest.read_text())
        def relocate(value):
            if isinstance(value, dict):
                return {k: relocate(v) for k, v in value.items()}
            if isinstance(value, list):
                return [relocate(v) for v in value]
            if isinstance(value, str):
                return value.replace(str(src.resolve()), str(dst.resolve()))
            return value
        data = relocate(data)
        data['run_id'] = dst.name
        manifest.write_text(yaml.safe_dump(data, sort_keys=False))


def glorys(args):
    from locomgcm.io.runner.config import load_run_config
    from locomgcm.io.runner.model_input import build_model_state, write_model_input
    cfg, raw = it.load(args.case)
    bundles = it.prep_bundles(cfg, args.prepared_run, ('boundary', 'mesh', 'forcing', 'state'))
    original = yaml.safe_load(args.case.read_text())
    pipelines = original['pipelines']['stage1a_prepare_common']
    for pipeline in pipelines:
        if pipeline['name'] == 'state':
            pipeline['primary_source'] = 'glorys'
    overlay = {'sources': {'glorys': {'type': 'hycom_netcdf', 'path': str(args.glorys_dir.resolve()), 'steady': True}},
               'pipelines': {'stage1a_prepare_common': pipelines}}
    cfg_g = load_run_config(args.case, extra_yaml=overlay)
    st = it.state_bundle(cfg_g)
    grid = it.grid_tensors(args.prepared_run)
    wet = grid['hFacC'][0] > 0
    area = grid['cell_area']
    eta = np.asarray(st.data['eta'].values)
    b = np.asarray(bundles['state'].data['eta'].values).reshape(-1, *wet.shape)[0]
    t = eta.reshape(-1, *wet.shape)[0]
    ok = wet & np.isfinite(b) & np.isfinite(t)
    shift = float((area * (b-t))[ok].sum() / area[ok].sum())
    st = it.bundle_with(st, eta=(eta+shift).astype(eta.dtype))
    mi = build_model_state(cfg, raw, stage1a_bundles={**bundles, 'state': st})
    copy_preparation(args.prepared_run, args.out)
    write_model_input(mi, args.out / '_pipeline_meta/model_input')
    it.write_state_zarr(st, args.out / '_pipeline_meta/preprocessed_cf/prepared_state.zarr')
    (args.out / 'ssh_datum.json').write_text(json.dumps(dict(shift_m=shift))+'\n')


def initial(args):
    from locomgcm.io.runner.model_input import write_model_input
    g = it.grid_tensors(args.prepared_run)
    nz, ny, nx = g['hFacC'].shape
    f = it.mit_fields(args.mit_run, args.iteration, nz, ny, nx)
    args.state_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.state_out, **f)
    mi = it.build_native(args.case, args.prepared_run, f)
    copy_preparation(args.prepared_run, args.out)
    write_model_input(mi, args.out / '_pipeline_meta/model_input')


def trajectory(args):
    namelist = args.data.read_text()
    dt_match = re.search(r'(?i)\bdeltaT\s*=\s*([0-9.eEdD+-]+)', namelist)
    if dt_match is None or float(dt_match.group(1).lower().replace('d', 'e')) != 600:
        raise ValueError('Native trajectory namelist must specify deltaT=600')
    args.out.mkdir(parents=True, exist_ok=True)
    meta = (args.mit_run / 'T.0000000000.meta').read_text()
    dims = [int(x) for x in meta.split('dimList')[1].split('[')[1].split(']')[0].replace(',', ' ').split()[0::3]]
    nx, ny, nz = dims
    steps = np.arange(0, 1153, 36, dtype=np.int32)
    for name in it.MIT_FILES.values():
        shape = (ny, nx) if name == 'Eta' else (nz, ny, nx)
        array = np.lib.format.open_memmap(args.out / f'{name}.npy', mode='w+', dtype=np.float32, shape=(len(steps), *shape))
        for k, step in enumerate(steps):
            array[k] = it.mit_read(args.mit_run, step, name, shape)
            if not np.isfinite(array[k]).all():
                raise ValueError(f'Non-finite {name} at {step}')
        array.flush()
    np.save(args.out / 'steps.npy', steps)
    (args.out / 'README.txt').write_text('MITgcm snapshots; dt 600 s; step 0 = t0. Float32 arrays; U west faces, V south faces.\n')


def case(args):
    cfg = yaml.safe_load(args.template.read_text())
    cfg['run']['start_date'] = args.start
    cfg['sources']['hycom']['start_date'] = args.start
    cfg['time']['days'] = args.days
    if args.mesh:
        cfg['sources']['mesh_closed']['path'] = str(args.mesh.resolve())
    cfg['output']['root'] = str(args.output_root.resolve())
    cfg['cache']['root'] = str(args.output_root.resolve() / 'cache')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.safe_dump(cfg, sort_keys=False))


def manifest(args):
    dates = json.loads(args.dates.read_text())
    if args.select:
        dates = [r for r in dates if r['t0'] in args.select]
    if args.fourdvar_only:
        dates = [r for r in dates if r['fourdvar']]
    if not dates:
        raise ValueError('No dates selected')
    rows = []
    for r in dates:
        root = args.data_root.resolve() / r['t0']
        rows.append(dict(name=r['t0'], t0=r['t0'], split=r['split'],
            case=str(root / 't0.yaml'), run_dir=str(root / 'background'),
            grid=str(root / 'background/_pipeline_meta/model_input/grid/tensors.npz'),
            background=str(root / 'background.npz'), truth=str(root / 'truth.npz'),
            mit_truth=str(root / 'trajectory'), obs=str(root / 'obs/D2.npz')))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(windows=rows), indent=2)+'\n')


def reference(args):
    """Use one tRef/sRef in every native run, retaining other namelist values."""
    src, dst = args.reference_data.read_text(), args.data.read_text()
    for key in ('tRef', 'sRef'):
        pattern = rf'(?im)^\s*{key}\s*=.*?(?=^\s*[A-Za-z&/])'
        found = re.search(pattern, src, flags=re.S)
        if found is None:
            raise ValueError(f'{key} missing in reference namelist')
        dst, count = re.subn(pattern, lambda _: found.group(), dst, flags=re.S)
        if count != 1:
            raise ValueError(f'Expected one {key} in target namelist')
    args.data.write_text(dst)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('case')
    q.add_argument('--template', type=Path, required=True)
    q.add_argument('--start', required=True)
    q.add_argument('--days', type=int, choices=[5, 8], required=True)
    q.add_argument('--mesh', type=Path)
    q.add_argument('--output-root', type=Path, required=True)
    q.add_argument('--out', type=Path, required=True)
    q = sub.add_parser('convert-glorys')
    q.add_argument('--input', type=Path, required=True)
    q.add_argument('--out', type=Path, required=True)
    for name in ('glorys', 'initial'):
        q = sub.add_parser(name)
        q.add_argument('--case', type=Path, required=True)
        q.add_argument('--prepared-run', type=Path, required=True)
        q.add_argument('--out', type=Path, required=True)
        if name == 'glorys':
            q.add_argument('--glorys-dir', type=Path, required=True)
        else:
            q.add_argument('--mit-run', type=Path, required=True, help='Directory containing native .data/.meta snapshots')
            q.add_argument('--iteration', type=int, default=720)
            q.add_argument('--state-out', type=Path, required=True)
    q = sub.add_parser('trajectory')
    q.add_argument('--mit-run', type=Path, required=True)
    q.add_argument('--data', type=Path, required=True, help='Native data namelist for the trajectory clock')
    q.add_argument('--out', type=Path, required=True)
    q = sub.add_parser('observations')
    q.add_argument('--grid', type=Path, required=True)
    q.add_argument('--truth', type=Path, required=True)
    q.add_argument('--t0', required=True)
    q.add_argument('--out', type=Path, required=True)
    q = sub.add_parser('manifest')
    q.add_argument('--dates', type=Path, required=True)
    q.add_argument('--data-root', type=Path, required=True)
    q.add_argument('--out', type=Path, required=True)
    q.add_argument('--select', nargs='+')
    q.add_argument('--fourdvar-only', action='store_true')
    q = sub.add_parser('reference')
    q.add_argument('--reference-data', type=Path, required=True)
    q.add_argument('--data', type=Path, required=True)
    args = p.parse_args(argv)
    if args.command == 'convert-glorys':
        it.glorys_convert(args.input, args.out)
    elif args.command == 'observations':
        from . import generate_obs as og
        stamp = datetime.date.fromisoformat(args.t0)
        seed = int(stamp.strftime('%Y%m%d'))
        og.SEED_POS, og.SEED_NOISE = seed*100+1, seed*100+2
        og.generate(args.grid, args.truth, args.out, f'{args.t0}T00:00:00')
    else:
        globals()[args.command](args)


if __name__ == '__main__':
    main()
