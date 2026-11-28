# South China Sea twin experiment

4D-Var, an analysis network trained online, and a network evaluated at every
time step in South China Sea twin experiments (160 × 152 × 41 grid, 600 s step,
three-day windows, forecasts to day eight). MITgcm runs provide the truth.
Commands run from the repository root.

## Inputs

For each date in [windows.json](windows.json): HYCOM state, open boundaries
and ERA5 winds from five days before to eight days after the window start, a
GLORYS12V1 daily state five days before the start, the prepared mesh of this
grid, and a MITgcm `data` namelist with the reference profiles `tRef` and
`sRef`. External paths are configured as in [docs/INPUTS.md](../../docs/INPUTS.md).
[b_std.json](b_std.json) holds the background error standard deviations.

## Truth, background and observations for one window

```sh
export LOCOMGCM_PATHS=/absolute/path/to/external_paths.toml
D="$PWD/outputs/da_twin/2021-12-27"
MESH=/absolute/path/to/prepared_mesh.zarr
GLORYS=/absolute/path/to/glorys_20211222.nc
REFERENCE=/absolute/path/to/reference/data

python -m experiments.da_twin.prepare case \
  --template experiments/configs/cases/da_twin/scs_twin.yaml \
  --start 2021-12-22 --days 5 --mesh "$MESH" --output-root "$D" --out "$D/adjust.yaml"
python -m experiments.da_twin.prepare case \
  --template experiments/configs/cases/da_twin/scs_twin.yaml \
  --start 2021-12-27 --days 8 --mesh "$MESH" --output-root "$D" --out "$D/t0.yaml"
python -m experiments.pipeline_v2.run_pipeline --config "$D/adjust.yaml" \
  --run-id adjust-background --set run.experiment_name=adjust-background --output "$D" --stages 1a,1b
python -m experiments.da_twin.prepare convert-glorys --input "$GLORYS" --out "$D/glorys-input"
python -m experiments.da_twin.prepare glorys --case "$D/adjust.yaml" \
  --prepared-run "$D/adjust-background" --glorys-dir "$D/glorys-input" --out "$D/adjust-truth"

for branch in background truth; do
  python -m experiments.pipeline_v2.run_pipeline --config "$D/adjust.yaml" \
    --run-id "adjust-$branch" --set "run.experiment_name=adjust-$branch" --output "$D" --stages 1c --force
  python -m experiments.da_twin.prepare reference --reference-data "$REFERENCE" \
    --data "$D/adjust-$branch/mitgcm/input/data"
  python -m experiments.pipeline_v2.run_pipeline --config "$D/adjust.yaml" \
    --run-id "adjust-$branch" --set "run.experiment_name=adjust-$branch" --output "$D" --stages 3
done

python -m experiments.pipeline_v2.run_pipeline --config "$D/t0.yaml" \
  --run-id t0-input --set run.experiment_name=t0-input --output "$D" --stages 1a,1b
for branch in background truth; do
  python -m experiments.da_twin.prepare initial --case "$D/t0.yaml" \
    --prepared-run "$D/t0-input" --mit-run "$D/adjust-$branch/mitgcm/run" \
    --iteration 720 --state-out "$D/$branch.npz" --out "$D/$branch"
done
python -m experiments.pipeline_v2.run_pipeline --config "$D/t0.yaml" \
  --run-id truth --set run.experiment_name=truth --output "$D" --stages 1c --force
python -m experiments.da_twin.prepare reference --reference-data "$REFERENCE" \
  --data "$D/truth/mitgcm/input/data"
python -m experiments.pipeline_v2.run_pipeline --config "$D/t0.yaml" \
  --run-id truth --set run.experiment_name=truth --output "$D" --stages 3
python -m experiments.da_twin.prepare trajectory --mit-run "$D/truth/mitgcm/run" \
  --data "$D/truth/mitgcm/input/data" --out "$D/trajectory"
python -m experiments.da_twin.prepare observations \
  --grid "$D/background/_pipeline_meta/model_input/grid/tensors.npz" \
  --truth "$D/trajectory" --t0 2021-12-27 --out "$D/obs"
```

## 4D-Var

```sh
python -m experiments.da_twin.prepare manifest --dates experiments/da_twin/windows.json \
  --data-root outputs/da_twin --out outputs/da_twin/all.json
python -m experiments.da_twin.prepare manifest --dates experiments/da_twin/windows.json \
  --data-root outputs/da_twin --fourdvar-only --out outputs/da_twin/var-windows.json
. experiments/da_twin/accelerators.sh
python -m experiments.da_twin.run var --windows outputs/da_twin/var-windows.json \
  --bstd experiments/da_twin/b_std.json --out outputs/da_twin/var
```

## Analysis network

```sh
python -m experiments.da_twin.run train --phase 1 --windows outputs/da_twin/all.json \
  --bstd experiments/da_twin/b_std.json --out outputs/da_twin/train-2019
python -m experiments.da_twin.run train --phase 2 --windows outputs/da_twin/all.json \
  --init-net outputs/da_twin/train-2019/best_s432.pt \
  --bstd experiments/da_twin/b_std.json --out outputs/da_twin/train-full
python -m experiments.da_twin.run analyze --windows outputs/da_twin/all.json --split test \
  --checkpoint outputs/da_twin/train-full/best.pt --norm outputs/da_twin/train-full/norm.json \
  --bstd experiments/da_twin/b_std.json --out outputs/da_twin/network
```

## Network evaluated at every time step, and evaluation

```sh
python -m experiments.da_twin.instep --windows outputs/da_twin/all.json \
  --select 2021-12-27 --bstd experiments/da_twin/b_std.json --out outputs/da_twin/instep
python -m experiments.da_twin.run evaluate --windows outputs/da_twin/all.json --split test \
  --scale-windows outputs/da_twin/all.json --var-dir outputs/da_twin/var \
  --network-dir outputs/da_twin/network --bstd experiments/da_twin/b_std.json \
  --out outputs/da_twin/evaluation
```

`--smoke` on `run var`, `run train`, `run analyze` and `instep` runs one
window with a few iterations.
