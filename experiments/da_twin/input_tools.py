"""Prepared bundles, GLORYS conversion and native MITgcm snapshots."""
import datetime as dt
import logging
from pathlib import Path
import numpy as np
log = logging.getLogger(__name__)
GNAMES = {"thetao": "water_temp", "so": "salinity", "uo": "water_u", "vo": "water_v"}
MIT_FILES = {"temp": "T", "salt": "S", "u": "U", "v": "V", "eta": "Eta"}
def jprint(tag, obj):
    print(tag, obj, flush=True)
def load(case, source="hycom"):
    from defgcm.io.runner.config import load_run_config
    from defgcm.io.runner.config._loader import merge_yaml_layers
    return load_run_config(Path(case)), merge_yaml_layers(Path(case), (), ())

def prep_bundles(run_cfg, prep_run, names=('boundary', 'mesh', 'forcing')):
    from defgcm.io.runner.layer_b import DATA_SOURCES, DataQuery, SourceContext
    from defgcm.io.runner.stages._common import build_grid_from_config
    model_grid = build_grid_from_config(run_cfg)

    def resolver(grid_id):
        if grid_id in (model_grid.grid_id, 'model_grid'):
            return model_grid
        raise ValueError(grid_id)
    ctx = SourceContext(cache_root=run_cfg.cache.root, target_grid_resolver=resolver, logger=log)
    out = {}
    for name in names:
        src = DATA_SOURCES.instantiate('zarr_bundle', ctx, path=str(Path(prep_run).absolute() / f'_pipeline_meta/preprocessed_cf/prepared_{name}.zarr'))
        kind = tuple(src.capabilities.produces)[0]
        out[name] = src.fetch(DataQuery(location_kind=kind, variables=src.capabilities.variables, time_range=None, target_grid=model_grid, extras={}))
    return out

def state_bundle(run_cfg):
    from defgcm.io.runner import model_input as mi
    from defgcm.io.runner.layer_b import DataQuery
    from defgcm.io.runner.layer_c import OPERATORS
    tg = mi._target_grid(run_cfg)
    (pipeline, source) = mi._source_for_pipeline(run_cfg, 'state', target_grid=tg, logger=log)
    bundle = source.fetch(DataQuery(location_kind=mi._initial_kind_for_pipeline(pipeline, source), variables=source.capabilities.variables, time_range=mi._query_time_range(run_cfg), target_grid=tg, extras={'first_time_only': True}))
    for op_spec in pipeline.operators:
        config = dict(op_spec.config)
        if config.get('target_grid') == 'model_grid':
            config['target_grid'] = tg
        bundle = OPERATORS.instantiate(op_spec.op, None, **config)(bundle)
    return bundle
def glorys_convert(src_nc, out_dir):
    import netCDF4
    out_dir.mkdir(parents=True, exist_ok=True)
    src = netCDF4.Dataset(src_nc)
    src.set_auto_maskandscale(False)
    (lat, lon, dep) = (np.asarray(src[n][:], dtype=np.float64) for n in ('latitude', 'longitude', 'depth'))
    t = src['time']
    if not str(t.units).startswith('hours since 1950-01-01'):
        raise SystemExit(f'unexpected time units {t.units}')
    hours = np.asarray(t[:], dtype=np.float64)
    if hours.shape != (1,):
        raise SystemExit(f'expected one record, got {hours.shape}')
    when = dt.datetime(1950, 1, 1) + dt.timedelta(hours=float(hours[0]))
    if when.hour or when.minute:
        raise SystemExit(f'record at {when}, not 00:00')
    target = out_dir / f'hycom_uvts_{when:%Y%m%d}.nc'
    tmp = target.with_suffix('.nc.partial')
    d = netCDF4.Dataset(tmp, 'w', format='NETCDF4')
    d.createDimension('time', 1)
    d.createDimension('depth', len(dep))
    d.createDimension('lat', len(lat))
    d.createDimension('lon', len(lon))
    for (name, vals, unit) in (('time', [(when - dt.datetime(2000, 1, 1)).total_seconds() / 3600], 'hours since 2000-01-01 00:00:00'), ('depth', dep, 'm'), ('lat', lat, 'degrees_north'), ('lon', lon, 'degrees_east')):
        v = d.createVariable(name, 'f8', (name,))
        v.units = unit
        v[:] = vals
    for (gname, hname) in GNAMES.items():
        s = src[gname]
        v = d.createVariable(hname, 'i2', ('time', 'depth', 'lat', 'lon'), zlib=True, complevel=4, shuffle=True, chunksizes=(1, len(dep), len(lat), len(lon)), fill_value=s.getncattr('_FillValue'))
        v.set_auto_maskandscale(False)
        v.scale_factor = float(s.scale_factor)
        v.add_offset = float(s.add_offset)
        v.units = str(s.units)
        v.glorys_name = gname
        v[:] = np.asarray(s[0:1])
    zos = src['zos']
    (zfill, zsc, zoff) = (zos.getncattr('_FillValue'), float(zos.scale_factor), float(zos.add_offset))
    raw = np.asarray(zos[0]).astype(np.float64)
    eta = np.where(raw == zfill, np.nan, raw * zsc + zoff)
    v = d.createVariable('surf_el', 'f8', ('time', 'lat', 'lon'), zlib=True, complevel=4, fill_value=np.nan)
    v.units = 'm'
    v.long_name = 'GLORYS zos (no datum shift; prepare glorys shifts it on the model grid)'
    v[:] = eta[None]
    d.source = f'GLORYS12V1 daily mean of {when:%Y-%m-%d} from {Path(src_nc).name}; thetao is potential temperature'
    d.history = 'GLORYS state conversion for the South China Sea twin experiment'
    d.close()
    tmp.replace(target)
    src.close()
    jprint('GLORYS_CONVERTED', dict(target=str(target), time=when, nlat=len(lat), nlon=len(lon), ndepth=len(dep), lat=[lat[0], lat[-1]], lon=[lon[0], lon[-1]], depth=[dep[0], dep[-1]]))

def bundle_with(bundle, **arrays):
    from defgcm.io.runner.layer_a._types import DataBundle
    ds = bundle.data.copy()
    for (name, values) in arrays.items():
        old = ds[name]
        if values.shape != old.shape or values.dtype != old.dtype:
            raise ValueError(f'{name}: {values.shape} {values.dtype} != {old.shape} {old.dtype}')
        ds[name] = old.copy(data=values)
    return DataBundle(data=ds, meta=bundle.meta)

def grid_tensors(prep):
    with np.load(Path(prep) / '_pipeline_meta/model_input/grid/tensors.npz') as g:
        return {k: np.asarray(g[k]) for k in g.files}

def state_arrays(bundle):
    return {n: np.asarray(bundle.data[n].values) for n in bundle.data.data_vars}

def write_state_zarr(bundle, path):
    from defgcm.io.runner.layer_d._zarr_utils import atomic_write_zarr, serialize_bundle_meta
    atomic_write_zarr(bundle.data.assign_attrs(bundle_meta=serialize_bundle_meta(bundle.meta)), Path(path))

def mit_read(run_dir, it, name, shape):
    f = Path(run_dir) / f'{name}.{it:010d}.data'
    meta = f.with_suffix('.meta').read_text()
    if 'float64' not in meta:
        raise SystemExit(f'{f}: not float64 ({meta!r})')
    a = np.fromfile(f, dtype='>f8')
    if a.size != int(np.prod(shape)):
        raise SystemExit(f'{f}: {a.size} values, expected {shape}')
    return a.reshape(shape).astype(np.float64)

def mit_fields(run_dir, it, nz, ny, nx):
    return {k: mit_read(run_dir, it, f, (ny, nx) if k == 'eta' else (nz, ny, nx)) for (k, f) in MIT_FILES.items()}

def build_native(case, prep, fields):
    import torch
    from defgcm.io.runner import model_input as mi
    (run_cfg, raw) = load(case, 'hycom')
    bundles = prep_bundles(run_cfg, prep, ('boundary', 'mesh', 'forcing', 'state'))
    (dtype, device) = (torch.float64, 'cpu')
    ic = mi._ic_tensors_from_bundles(bundles, run_cfg, device=device, dtype=dtype)
    (u, v) = (fields['u'], fields['v'])
    native = dict(temperature=fields['temp'], salinity=fields['salt'], eta=fields['eta'], u=np.concatenate((u, u[..., :1]), axis=-1), v=np.concatenate((v, v[..., :1, :]), axis=-2))
    (nz, ny, nx) = (int(run_cfg.grid.nz), int(run_cfg.grid.ny), int(run_cfg.grid.nx))
    expect = dict(temperature=(nz, ny, nx), salinity=(nz, ny, nx), eta=(ny, nx), u=(nz, ny, nx + 1), v=(nz, ny + 1, nx))
    for (name, values) in native.items():
        if values.shape != expect[name]:
            raise ValueError(f'{name}: native {values.shape} != {expect[name]}')
        ic[name] = torch.as_tensor(np.ascontiguousarray(values), dtype=dtype, device=device)
    grid = mi._build_grid_from_tensors(run_cfg, ic['bathymetry'], del_x=ic.get('del_x'), del_y=ic.get('del_y'), device=device, dtype=dtype)
    mi._apply_canonical_transforms(ic, grid, dtype=dtype, device=device)
    (obcs, wind, sflux, bulk) = mi._build_runtime_sources(bundles, run_cfg, device=device, dtype=dtype)
    (model, state) = mi._assemble_model(run_cfg, ic_bundle=bundles['state'], ic_tensors=ic, grid=grid, obcs_source=obcs, wind_source=wind, surface_flux_source=sflux, bulk_source=bulk, device=device, tidal_bundle=bundles.get('tidal'))
    return mi.ModelInput(grid=model.grid, initial_state=state, boundary_source=obcs, era5_source=wind, barotropic_source=mi._BarotropicBoundaryBundleSource(obcs), run_params=mi._model_input_run_params(run_cfg, raw))
