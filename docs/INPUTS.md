# Simulation inputs

## Configure external inputs

Set `DEFGCM_PATHS` to the absolute path of a TOML file based on
[`external_paths.example.toml`](../external_paths.example.toml). The file uses
`schema_version = 1` and a `[paths]` table. Paths in the table must be absolute;
omit unused entries and source/pin pairs. Values are resolved when consumed,
so imports and CLI help do not require external inputs. An explicit CLI path
wins over environment settings; environment settings win over TOML. Missing
values name the required key and environment variable. There is no checkout
relative fallback for external inputs.

| TOML key | Environment names, including compatibility aliases |
|---|---|
| `mitgcm_root` | `MITGCM_ROOT`, `MITCASES_MIT_ROOT`, `RPAIR_MIT_ROOT` |
| `mitgcm_pin` | `MITGCM_PIN`, `MITCASES_MIT_PIN`, `RPAIR_MIT_PIN` |
| `mitgcm_ad_root` | `MITGCM_AD_ROOT`, `CS32_MIT_ROOT` |
| `mitgcm_ad_pin` | `MITGCM_AD_PIN`, `CS32_MIT_PIN` |
| `cs32_grid_dir` | `DEFGCM_CS32_GRID_DIR` |
| `cs32_input_dir` | `DEFGCM_CS32_INPUT_DIR` |
| `conda_root` | `MITGCM_CONDA_ROOT` |
| `mitgcm_optfile` | `MITGCM_OPTFILE`, `MITCASES_OPTFILE`, `RPAIR_OPTFILE` |
| `hycom_dir` | `DEFGCM_HYCOM_DIR` |
| `era5_dir` | `DEFGCM_ERA5_DIR` |
| `gebco_file` | `DEFGCM_GEBCO_FILE` |
| `gebco_scs_file` | `DEFGCM_GEBCO_SCS_FILE` |
| `topo_file` | `DEFGCM_TOPO_FILE` |
| `tpxo9_dir` | `DEFGCM_TPXO9_DIR` |
| `global_data_dir` | `DEFGCM_GLOBAL_DATA_DIR` |
| `global_forecast_dir` | `DEFGCM_GLOBAL_FORECAST_DIR` |
| `scs_clm_dir` | `DEFGCM_SCS_CLM_DIR` |
| `cube_cs192_dir` | `DEFGCM_CUBE_CS192_DIR` |
| `cube_cs1024_dir` | `DEFGCM_CUBE_CS1024_DIR` |
| `synthetic_pureocean_dir` | `DEFGCM_SYNTHETIC_PUREOCEAN_DIR` |

## Case data locations

Case YAML uses `external:KEY` or `external:KEY/child` for raw inputs. For
example, `sources.hycom.path: external:hycom_dir` and
`grid.cube_grid_dir: external:cube_cs192_dir/grid` select named locations.
Append the entries needed by your case to `[paths]`, for example:

```toml
hycom_dir = "/absolute/path/to/hycom"
era5_dir = "/absolute/path/to/era5"
gebco_file = "/absolute/path/to/gebco_2024_westpac.nc"
tpxo9_dir = "/absolute/path/to/TPXO9_atlas_v5_nc"
```

`gebco_file`, `gebco_scs_file` and `topo_file` each name one file; the other
data keys name directories. The historical SCS configurations select their
original SCS topography through `gebco_scs_file`; the standalone HYCOM
comparison's default topography uses `topo_file`.
`global_data_dir` contains the global HYCOM files, `_raw_by_variable/hycom`,
`era5`, `hycom_topo_as_gebco_elevation.nc` and `tpxo9_atlas_v5_nc` as required
by the selected global case. `global_forecast_dir` contains the forecast
HYCOM files. `scs_clm_dir` contains the state, mesh, boundary and forcing
Zarr bundles and the selected hot-start directory. Each cube directory
contains `grid/`, its bathymetry and initial-state files, and any configured
forcing Zarr. `synthetic_pureocean_dir` contains the example generator's
`hycom/`, `era5/` and `gebco.nc`.

Resolution runs after all overlays and `--set` values have been merged, before
model construction. Missing keys and missing paths raise `ExternalPathError`;
the message names the key, its environment variable or the missing input.
An absolute `sources.<name>.path` override remains supported. Relative raw
input paths, relative data-key environment values and `..` in named children
are rejected. Direct data-source constructors use the same resolver. Directory
and file existence are checked here; readers validate file contents and time
coverage when used. A prepared run explicitly selected with `--run-dir` or
`--prepared-run` retains its existing path semantics.

## Source and tool settings

Conflicting environment aliases are rejected. `MITCASES_VERIFICATION_ROOT`
remains an alias for the selected source's `verification` directory.
`MITCASES_MIT_ENV` can explicitly select the compiler environment; otherwise it
is derived as `conda_root/envs/${MITGCM_CONDA_ENV_NAME:-mitgcm-intel}`.
`CONDA_EXE` can explicitly select conda; otherwise it is `conda_root/bin/conda`.
These derived tool settings do not add TOML keys.

A source root and its full 40-character Git pin must be configured together.
The source must carry a matching `PIN` file or its own Git checkout. If both
are present, both must match. An enclosing Git repository is not a source
identity. Local patches remain allowed and are recorded by the build workflow.
CLI source overrides still require the configured pin. `verification`, default
`CPP_OPTIONS.h`, and KPP package headers are derived from the verified source.
A complete case-local CPP/KPP header remains self-contained.

The mitref patch bundle additionally requires the revision declared by
`experiments.mitcases.mitref.contract.PIN`; changing a configured pin does not
make its patches compatible with another revision. The common path resolver
performs all source-identity checks.

## Supplied simulation archives

The accompanying input archives contain the prepared arrays actually used by
the regional and global experiments. Extract the required archive at the root
of the extracted source project. Each archive includes `INPUT_FILES.json`, which lists its relative input
paths, source runs and file sizes.
The archives store input files directly. Prepared Zarr chunks are already compressed; the container is TAR.

| Archive | Contents | Use |
|---|---|---|
| `torchocean_inputs_regional_h160.tar` | Prepared low-resolution forward and sensitivity inputs | The 29-day low-resolution forward comparison and the 100-step regional sensitivity experiments |
| `torchocean_inputs_regional_h960.tar` | Prepared high-resolution forward inputs | The 29-day high-resolution forward comparison |
| `torchocean_inputs_global.tar` | Prepared EXF inputs and selected MITgcm CS32 verification inputs | EXF and CS32 initial conditions, geometry and prescribed forcing |

The regional archives include the prepared state, mesh, forcing and boundary
stores; model-input and open-boundary exports; configuration snapshots; and
configuration metadata. The global archive includes
the EXF prepared-input run and the eleven CS32 grid, hydrography, bathymetry and
wind files plus the native adjoint seed file used by `setup_run.py`.

```sh
tar -xf torchocean_inputs_regional_h160.tar
```

Input preparation is stage 1a,
LOCOMgcm input export is 1b, native input export is 1c, and forward execution
is 2 or 3. Prepared inputs can be reused when selecting later stages. Select the desired configuration and run directory explicitly, and
use the compute-allocation instructions in the repository README. The evidence
archive provides a separate route for checking the reported numbers and
redrawing the figures without rerunning these integrations.

## External data and acquisition

The regional runs start on 15 July 2019 and cover 29 days. The source archive's
case configurations specify each model grid and its spatial interpolation.
The downloaded HYCOM daily bundles cover 0--45 degrees north and
94.9599609375--170 degrees east, with eight three-hourly records and 40 depth
levels per day. Their variables are `water_temp`, `salinity`, `water_u`,
`water_v` and `surf_el`. Temperature and salinity use a scale factor of 0.001
and an offset of 20; velocities and elevation use a scale factor of 0.001 and
zero offset. NetCDF readers must apply those packing attributes.

The official HYCOM acquisition endpoint is the Naval Research Laboratory's
GOFS 3.1 analysis on GLBy0.08, experiment 93.0:

- [Product description and conditions](https://www.hycom.org/dataserver/gofs-3pt1/analysis)
- [Dataset and access services](https://tds.hycom.org/thredds/catalogs/GLBy0.08/expt_93.0.html?dataset=GLBy0.08-expt_93.0)
- [NetCDF subset service](https://ncss.hycom.org/thredds/ncss/grid/GLBy0.08/expt_93.0)

Select 15 July through 13 August 2019, three-hourly times, the five variables
above, all 40 published depth levels, and the source bounding box. The source
tree's `scripts/redownload_hycom.py` and `scripts/configs/hycom_redownload.yaml`
record this product, grid and cadence for the four temperature, salinity and
velocity variables. Obtain `surf_el` from the same catalog when reproducing all
five variables in the retained bundles. The official
catalog covers the requested dates and marks the data freely available; the
product page states public release and unlimited distribution. Access was
checked on 1 October 2026.

The retained daily files identify themselves as a NAS aggregation and do not
store the upstream experiment identifier. The product association above is
supported by the retained acquisition configuration and matching grid and
cadence. The accompanying prepared-input archives preserve the exact arrays
used in this study independently of changes to the public service.

ERA5 10-metre wind components are available from the
[ERA5 hourly single-level dataset](https://doi.org/10.24381/cds.adbb2d47).
Select the same regional dates and the spatial bounds required by the case;
the preparation code converts the winds to the six-hourly surface stresses
used by the model. The source file selection is retained in each configuration
snapshot. Bathymetry is the
[GEBCO 2024 grid](https://doi.org/10.5285/1c44ce99-0a0d-5f4f-e063-7086abc0ea0f),
retained locally as `data/gebco_2024_westpac.nc`. Follow each provider's stated
access and reuse conditions. The software's license does not replace
third-party data conditions.

## Reference model inputs

The repository README and case records identify the upstream MITgcm revision for
each comparison. The global native adjoint uses checkpoint68u; the regional
forward comparison uses revision `d861cd50`. Native build tools and the
upstream source tree are obtained as described in the repository README. The
global input archive preserves the selected verification inputs and their
upstream license notices. It can be extracted alongside a separately obtained
upstream tree without replacing its model source.
