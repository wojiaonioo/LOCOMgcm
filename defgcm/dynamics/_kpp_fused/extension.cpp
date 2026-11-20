#include <torch/extension.h>

#include <string>
#include <vector>

void statekpp_fused_launch(
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&,
    torch::Tensor&, double, double, const std::string&);

void kpp_bldepth_launch(
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&, torch::Tensor&,
    torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&,
    torch::Tensor&, double, double, double, double, double, double, double,
    double, bool, int, bool, bool);

void kpp_blmix_launch(
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    torch::Tensor&, torch::Tensor&, torch::Tensor&, double, double, double);

void kpp_preprocess_launch(
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&,
    torch::Tensor&, torch::Tensor&,
    torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&,
    torch::Tensor&, torch::Tensor&, double, double, double, double);

void kpp_ri_launch(
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, torch::Tensor&, double, double,
    double, double, double, double, double, double, double, double,
    double, double, torch::Tensor&);

void kpp_wscale_launch(
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&,
    torch::Tensor&, torch::Tensor&, double, double, bool);

void kpp_finalize_launch(
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&, torch::Tensor&, torch::Tensor&,
    torch::Tensor&, torch::Tensor&, double);

namespace {

void check_float(const torch::Tensor& value, const torch::Tensor& reference,
                 const char* name) {
  TORCH_CHECK(value.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(value.device() == reference.device(), name, " device mismatch");
  TORCH_CHECK(value.scalar_type() == reference.scalar_type(), name,
              " dtype mismatch");
  TORCH_CHECK(value.scalar_type() == at::kFloat ||
                  value.scalar_type() == at::kDouble,
              name, " must be float32 or float64");
}

// Slot names for the debug vector ``regional_kpp_fused_all`` returns.
// The production entry returns its five outputs directly by name.
// Keep in lockstep with ``dispatcher.STAGE_NAMES``.
enum StageSlot : size_t {
  kDbloc = 0, kDbsfc, kRho, kAlpha, kBeta,
  kDblocSm, kDblocMasked, kRitop, kDvsq, kShsq, kUstar, kBo, kBosol,
  kHbl, kKbl, kBfsfc, kStable, kCasea, kRib,
  kDiffus,
  kWmHbl, kWsHbl, kWmLevels, kWsLevels, kWmD, kWsD,
  kBlmc, kGhat, kDkm1,
  kViscOut, kKzsOut, kKztOut, kGhatOut,
  kRiTap,
  kStageCount,
};

// Stage-resolved core.  ``regional_kpp_fused_fwd`` (the production entry)
// publishes only the five model-facing outputs; ``regional_kpp_fused_stages``
// (debug only, no production caller) publishes every intermediate in launch
// order so the eager reference can be diffed stage by stage and the FIRST
// unequal stage identified. Both call THIS body with the same kernel sequence
// and arithmetic. Production releases scratch after its last consumer launch;
// debug retains it for the stage vector. All launches use the current stream,
// so the CUDA allocator can safely reuse the released storage on that stream.
// Debug order must stay in lockstep with ``dispatcher.STAGE_NAMES``.
std::vector<torch::Tensor> regional_kpp_fused_all(
    torch::Tensor theta, torch::Tensor salt, torch::Tensor u_vel,
    torch::Tensor v_vel, torch::Tensor surf_forc_t, torch::Tensor surf_forc_s,
    torch::Tensor surf_forc_u, torch::Tensor surf_forc_v,
    torch::Tensor coriol, torch::Tensor mask_c, torch::Tensor kmtj,
    torch::Tensor layer_thickness, torch::Tensor zgrid, torch::Tensor hwide,
    torch::Tensor wmt, torch::Tensor wst, torch::Tensor qsw,
    torch::Tensor pressure, torch::Tensor bg_kzs, torch::Tensor bg_kzt,
    std::vector<double> scalars, const std::string& eos, bool penetrating_sw,
    bool sign_zero_positive, bool tap) {
  TORCH_CHECK(theta.is_cuda() && theta.dim() == 3,
              "theta must be a rank-3 CUDA tensor");
  TORCH_CHECK(kmtj.is_cuda() && kmtj.device() == theta.device() &&
                  kmtj.scalar_type() == theta.scalar_type(),
              "kmtj must share theta's CUDA device and storage dtype");
  TORCH_CHECK(theta.size(0) == 90 && theta.size(1) >= 3 && theta.size(2) >= 3,
              "regional KPP fused supports non-empty nz=90 slabs");
  TORCH_CHECK(scalars.size() == 22, "regional KPP fused expects 22 scalars");
  TORCH_CHECK(eos == "jmd95" || eos == "mdjwf", "unsupported KPP EOS");
  for (const auto* pair : {&salt, &u_vel, &v_vel, &surf_forc_t, &surf_forc_s,
                           &surf_forc_u, &surf_forc_v, &coriol, &mask_c,
                           &layer_thickness, &zgrid, &hwide, &wmt, &wst, &qsw,
                           &pressure, &bg_kzs, &bg_kzt}) {
    check_float(*pair, theta, "floating KPP input");
  }
  const auto nz = theta.size(0);
  const auto ny = theta.size(1);
  const auto nx = theta.size(2);
  TORCH_CHECK(salt.sizes() == theta.sizes() && u_vel.sizes() == theta.sizes() &&
                  v_vel.sizes() == theta.sizes(),
              "state volumes must have identical shapes");
  TORCH_CHECK(mask_c.sizes() == theta.sizes() &&
                  bg_kzs.sizes() == theta.sizes() &&
                  bg_kzt.sizes() == theta.sizes(),
              "mask/background volumes must match theta");

  auto theta4 = theta.unsqueeze(0);
  auto salt4 = salt.unsqueeze(0);
  auto pressure4 = pressure.dim() == 1
      ? pressure.reshape({1, nz, 1, 1})
      : pressure.unsqueeze(0);
  auto dbloc4 = torch::empty_like(theta4);
  auto dbsfc4 = torch::empty_like(theta4);
  auto rho4 = torch::empty({1, ny, nx}, theta.options());
  auto alpha4 = torch::empty({1, nz + 1, ny, nx}, theta.options());
  auto beta4 = torch::empty_like(alpha4);
  statekpp_fused_launch(theta4, salt4, pressure4, dbloc4, dbsfc4, rho4,
                        alpha4, beta4, scalars[0], scalars[1], eos);
  auto dbloc = dbloc4.squeeze(0);
  auto dbsfc = dbsfc4.squeeze(0);
  auto rho = rho4.squeeze(0);
  auto alpha = alpha4.squeeze(0);
  auto beta = beta4.squeeze(0);
  // Keep only the 3-D views; the 4-D aliases must not extend scratch lifetime.
  dbloc4.reset();
  dbsfc4.reset();
  rho4.reset();
  alpha4.reset();
  beta4.reset();

  auto dbloc_sm = torch::empty_like(theta);
  auto dbloc_masked = torch::empty_like(theta);
  auto ritop = torch::empty_like(theta);
  auto dvsq = torch::empty_like(theta);
  auto shsq = torch::empty_like(theta);
  auto ustar = torch::empty({ny, nx}, theta.options());
  auto bo = torch::empty_like(ustar);
  auto bosol = torch::empty_like(ustar);
  kpp_preprocess_launch(
      dbloc, dbsfc, rho, alpha, beta, u_vel, v_vel, surf_forc_t,
      surf_forc_s, surf_forc_u, surf_forc_v, qsw, mask_c, kmtj, zgrid,
      layer_thickness, dbloc_sm,
      dbloc_masked, ritop, dvsq, shsq, ustar, bo, bosol, scalars[0],
      scalars[1], scalars[20], scalars[21]);
  if (!tap) {
    dbloc.reset();
    dbsfc.reset();
    rho.reset();
    alpha.reset();
    beta.reset();
  }

  auto kmtj_long = kmtj.scalar_type() == at::kLong ? kmtj : kmtj.to(at::kLong);
  auto hbl = torch::empty_like(ustar);
  auto kbl = torch::empty({ny, nx}, kmtj_long.options());
  auto bfsfc = torch::empty_like(ustar);
  auto stable = torch::empty_like(ustar);
  auto casea = torch::empty_like(ustar);
  // bldepth writes Rib[1:nz] but may interpolate against Rib[0], which is
  // defined as zero by the eager implementation.
  auto rib = torch::zeros_like(theta);
  kpp_bldepth_launch(
      kmtj_long, dvsq, dbloc_masked, ritop, ustar, bo, bosol, coriol, zgrid,
      hwide, wst, hbl, kbl, bfsfc, stable, casea, rib, scalars[2], scalars[3],
      scalars[4], scalars[5], scalars[6], scalars[7], scalars[8], scalars[20],
      true, 2, penetrating_sw, sign_zero_positive);
  if (!tap) {
    dvsq.reset();
    ritop.reset();
    bo.reset();
    bosol.reset();
    rib.reset();
  }

  auto diffus = torch::empty({3, nz + 2, ny, nx}, theta.options());
  // Sub-expression tap: undefined tensors -> the kernel receives nullptr and
  // writes no tap.  When requested, the tap holds ri_kernel's own
  // intermediates (later stages overwrite diffus in place).
  torch::Tensor ri_tap;
  if (tap) {
    ri_tap = torch::empty({9, nz + 2, ny, nx}, theta.options());
  }
  // Reciprocals of the two Python-scalar divisors, computed HERE on the host in
  // double -- the same place and the same type ATen's div.Scalar computes them
  // before launching its multiply.  Doing it on the device instead would round
  // differently.
  const double inv_riinfty = 1.0 / scalars[9];
  const double inv_bvsqcon = 1.0 / scalars[10];
  kpp_ri_launch(kmtj_long, shsq, dbloc_masked, dbloc_sm, bg_kzs, bg_kzt,
                zgrid, diffus, scalars[9], scalars[10], scalars[11],
                scalars[12], scalars[13], scalars[14], scalars[15], scalars[16],
                scalars[17], scalars[20], inv_riinfty, inv_bvsqcon, ri_tap);
  kmtj_long.reset();
  if (!tap) {
    shsq.reset();
    dbloc_masked.reset();
    dbloc_sm.reset();
  }

  auto wm_hbl = torch::empty_like(ustar);
  auto ws_hbl = torch::empty_like(ustar);
  auto wm_levels = torch::empty_like(theta);
  auto ws_levels = torch::empty_like(theta);
  auto wm_d = torch::empty_like(ustar);
  auto ws_d = torch::empty_like(ustar);
  kpp_wscale_launch(ustar, bfsfc, hbl, stable, kbl, zgrid, hwide, wmt, wst,
                    wm_hbl, ws_hbl, wm_levels, ws_levels, wm_d, ws_d,
                    scalars[3], scalars[20], sign_zero_positive);

  auto blmc = torch::empty({3, nz, ny, nx}, theta.options());
  auto ghat = torch::empty_like(theta);
  auto dkm1 = torch::empty({3, ny, nx}, theta.options());
  kpp_blmix_launch(ustar, bfsfc, hbl, stable, casea, kbl, diffus, zgrid,
                   hwide, wm_hbl, ws_hbl, wm_levels, ws_levels, wm_d, ws_d,
                   blmc, ghat, dkm1, scalars[19], scalars[18], scalars[20]);
  if (!tap) {
    ustar.reset();
    bfsfc.reset();
    stable.reset();
    wm_hbl.reset();
    ws_hbl.reset();
    wm_levels.reset();
    ws_levels.reset();
    wm_d.reset();
    ws_d.reset();
  }

  auto visc_out = torch::empty_like(theta);
  auto kzs_out = torch::empty_like(theta);
  auto kzt_out = torch::empty_like(theta);
  auto ghat_out = torch::empty_like(theta);
  kpp_finalize_launch(dkm1, hbl, kbl, casea, diffus, blmc, ghat, zgrid,
                      mask_c, bg_kzs, bg_kzt, visc_out, kzs_out, kzt_out,
                      ghat_out, scalars[17]);
  if (!tap) {
    // Returning here drops the remaining scratch at finalize's last read.
    return {visc_out, kzs_out, kzt_out, ghat_out, hbl};
  }
  std::vector<torch::Tensor> stages = {
      dbloc,   dbsfc,     rho,        alpha,      beta,
      dbloc_sm, dbloc_masked, ritop,  dvsq,       shsq,
      ustar,    bo,        bosol,
      hbl,      kbl,       bfsfc,     stable,     casea,     rib,
      diffus,
      wm_hbl,   ws_hbl,    wm_levels, ws_levels,  wm_d,      ws_d,
      blmc,     ghat,      dkm1,
      visc_out, kzs_out,   kzt_out,   ghat_out,
      tap ? ri_tap : diffus};
  TORCH_CHECK(stages.size() == kStageCount,
              "stage vector has ", stages.size(), " entries but StageSlot "
              "declares ", static_cast<size_t>(kStageCount));
  return stages;
}

}  // namespace

std::vector<torch::Tensor> regional_kpp_fused_fwd(
    torch::Tensor theta, torch::Tensor salt, torch::Tensor u_vel,
    torch::Tensor v_vel, torch::Tensor surf_forc_t, torch::Tensor surf_forc_s,
    torch::Tensor surf_forc_u, torch::Tensor surf_forc_v,
    torch::Tensor coriol, torch::Tensor mask_c, torch::Tensor kmtj,
    torch::Tensor layer_thickness, torch::Tensor zgrid, torch::Tensor hwide,
    torch::Tensor wmt, torch::Tensor wst, torch::Tensor qsw,
    torch::Tensor pressure, torch::Tensor bg_kzs, torch::Tensor bg_kzt,
    std::vector<double> scalars, const std::string& eos, bool penetrating_sw,
    bool sign_zero_positive) {
  return regional_kpp_fused_all(
      theta, salt, u_vel, v_vel, surf_forc_t, surf_forc_s, surf_forc_u,
      surf_forc_v, coriol, mask_c, kmtj, layer_thickness, zgrid, hwide, wmt,
      wst, qsw, pressure, bg_kzs, bg_kzt, scalars, eos, penetrating_sw,
      sign_zero_positive, /*tap=*/false);
}

std::vector<torch::Tensor> regional_kpp_fused_stages(
    torch::Tensor theta, torch::Tensor salt, torch::Tensor u_vel,
    torch::Tensor v_vel, torch::Tensor surf_forc_t, torch::Tensor surf_forc_s,
    torch::Tensor surf_forc_u, torch::Tensor surf_forc_v,
    torch::Tensor coriol, torch::Tensor mask_c, torch::Tensor kmtj,
    torch::Tensor layer_thickness, torch::Tensor zgrid, torch::Tensor hwide,
    torch::Tensor wmt, torch::Tensor wst, torch::Tensor qsw,
    torch::Tensor pressure, torch::Tensor bg_kzs, torch::Tensor bg_kzt,
    std::vector<double> scalars, const std::string& eos, bool penetrating_sw,
    bool sign_zero_positive) {
  return regional_kpp_fused_all(
      theta, salt, u_vel, v_vel, surf_forc_t, surf_forc_s, surf_forc_u,
      surf_forc_v, coriol, mask_c, kmtj, layer_thickness, zgrid, hwide, wmt,
      wst, qsw, pressure, bg_kzs, bg_kzt, scalars, eos, penetrating_sw,
      sign_zero_positive, /*tap=*/true);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("regional_kpp_fused_fwd", &regional_kpp_fused_fwd,
             "Full regional KPP fused scope (CUDA)");
  module.def("regional_kpp_fused_stages", &regional_kpp_fused_stages,
             "Full regional KPP fused scope, every intermediate (debug)");
}
