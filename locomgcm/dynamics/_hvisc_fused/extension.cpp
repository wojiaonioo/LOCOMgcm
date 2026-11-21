#include <torch/extension.h>

#include <optional>
#include <vector>

void hvisc_stage1_launch(
    const torch::Tensor &u, const torch::Tensor &v, const torch::Tensor &u_ext,
    const torch::Tensor &v_dyC_ext, const torch::Tensor &dxC_ext,
    const torch::Tensor &dxG_ext, const torch::Tensor &dyG,
    const torch::Tensor &dyF, const torch::Tensor &recip_rA,
    const torch::Tensor &recip_rAz, const torch::Tensor &hFacW,
    const torch::Tensor &hFacS, const torch::Tensor &recip_hFacC,
    const torch::Tensor &maskW, const torch::Tensor &maskS,
    const torch::Tensor &hfacz, const std::optional<torch::Tensor> &maskInC,
    const std::optional<torch::Tensor> &maskInW,
    const std::optional<torch::Tensor> &maskInS, const torch::Tensor &l2_d,
    const torch::Tensor &l3_d, const torch::Tensor &l2_z,
    const torch::Tensor &l3_z, const torch::Tensor &leith2fac,
    const torch::Tensor &smag2fac, const torch::Tensor &base,
    double side_mask_fac, torch::Tensor &viscAh_D, torch::Tensor &hDiv_flux,
    torch::Tensor &hz_v3);

void hvisc_stage2_launch(
    const torch::Tensor &hDiv_flux, const std::optional<torch::Tensor> &hDiv_x,
    const std::optional<torch::Tensor> &hDiv_y, const torch::Tensor &hz_v3,
    const torch::Tensor &dxC_row, const torch::Tensor &dyC,
    const torch::Tensor &dxG_ext, const torch::Tensor &dyG,
    const torch::Tensor &recip_hFacW, const torch::Tensor &recip_hFacS,
    const torch::Tensor &maskW, const torch::Tensor &maskS, bool u_all_faces,
    bool v_all_faces, bool keep_west, bool keep_east, bool keep_south,
    bool keep_north, torch::Tensor &u_out, torch::Tensor &v_out);

void hvisc_mr_pre_launch(
    const torch::Tensor &u, const torch::Tensor &v, const torch::Tensor &u_ext,
    const torch::Tensor &v_dyC_ext, const torch::Tensor &dxC_ext,
    const torch::Tensor &dxG, const torch::Tensor &dyG, const torch::Tensor &dyF,
    const torch::Tensor &recip_rA, const torch::Tensor &recip_rAz,
    const torch::Tensor &hFacW, const torch::Tensor &hFacS,
    const torch::Tensor &recip_hFacC, const torch::Tensor &maskW,
    const torch::Tensor &maskS, const torch::Tensor &hfacz,
    const std::optional<torch::Tensor> &maskInC,
    const std::optional<torch::Tensor> &maskInW,
    const std::optional<torch::Tensor> &maskInS, double side_factor,
    torch::Tensor &hdiv, torch::Tensor &vort, torch::Tensor &strain,
    torch::Tensor &tension, torch::Tensor &vrtDx, torch::Tensor &vrtDy);

void hvisc_mr_coeff_launch(
    const torch::Tensor &hdiv, const torch::Tensor &vort,
    const torch::Tensor &strain, const torch::Tensor &vrtDx_x,
    const torch::Tensor &vrtDy_y, const torch::Tensor &tension_xy,
    const torch::Tensor &hfacz, const torch::Tensor &l2_d,
    const torch::Tensor &l3_d, const torch::Tensor &l2_z,
    const torch::Tensor &l3_z, const torch::Tensor &leith,
    const torch::Tensor &smag, const torch::Tensor &base,
    torch::Tensor &visc_d, torch::Tensor &flux, torch::Tensor &hz);

namespace {

void check_float(const torch::Tensor &value, const char *name,
                 at::ScalarType dtype, const torch::Device &device) {
  TORCH_CHECK(value.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(value.device() == device, name, " must share one CUDA device");
  TORCH_CHECK(value.scalar_type() == dtype, name,
              " must share the reference dtype");
  for (int64_t dim = 0; dim < value.dim(); ++dim) {
    TORCH_CHECK(value.stride(dim) >= 0, name,
                " must not have negative strides");
  }
}

// The 2-D metric accessors are read through their LAST TWO strides, so a
// leading broadcast axis (``l2_d`` arrives as ``[1, 1, ny, nx]``) is fine as
// long as those axes are singleton -- anything else would silently drop data.
void want2(const torch::Tensor &value, const char *name, at::ScalarType dtype,
           const torch::Device &device, int64_t d0, int64_t d1) {
  check_float(value, name, dtype, device);
  TORCH_CHECK(value.dim() >= 2, name, " must have at least two dimensions");
  const int64_t dim = value.dim();
  for (int64_t axis = 0; axis < dim - 2; ++axis) {
    TORCH_CHECK(value.size(axis) == 1, name,
                " may only carry singleton leading dimensions");
  }
  TORCH_CHECK(value.size(dim - 2) == d0 && value.size(dim - 1) == d1, name,
              " has an unexpected shape");
}

void want2_opt(const std::optional<torch::Tensor> &value, const char *name,
               at::ScalarType dtype, const torch::Device &device, int64_t d0,
               int64_t d1) {
  if (!value.has_value()) {
    return;
  }
  want2(*value, name, dtype, device, d0, d1);
}

// Accessor precondition.  The kernel-side accessors (kernel.cu A4/A3/A2/M2)
// index their innermost axis with an IMPLICIT stride of 1 -- dropping that
// per-tensor multiply is what took hvisc_stage1_kernel<double> from 170 to 96
// registers, i.e. from one resident block per SM to five.  Every tensor that
// reaches an
// accessor is therefore routed through here first.  Production inputs are
// already contiguous, so this is a refcount bump and nothing is copied; the
// copy exists so an input that is not cannot silently be read at the wrong
// addresses.  The copy holds identical values, so results are unchanged, and
// the caching allocator keeps the temporary alive on this stream until the
// asynchronous kernel that reads it has run.
torch::Tensor unit_row(const torch::Tensor &value) {
  return value.stride(value.dim() - 1) == 1 ? value : value.contiguous();
}

std::optional<torch::Tensor> unit_row_opt(
    const std::optional<torch::Tensor> &value) {
  if (!value.has_value()) {
    return value;
  }
  return unit_row(*value);
}

} // namespace

std::vector<torch::Tensor> hvisc_fused_stage1(
    torch::Tensor u, torch::Tensor v, torch::Tensor u_ext,
    torch::Tensor v_dyC_ext, torch::Tensor dxC_ext, torch::Tensor dxG_ext,
    torch::Tensor dyG, torch::Tensor dyF, torch::Tensor recip_rA,
    torch::Tensor recip_rAz, torch::Tensor hFacW, torch::Tensor hFacS,
    torch::Tensor recip_hFacC, torch::Tensor maskW, torch::Tensor maskS,
    torch::Tensor hfacz, std::optional<torch::Tensor> maskInC,
    std::optional<torch::Tensor> maskInW, std::optional<torch::Tensor> maskInS,
    torch::Tensor l2_d, torch::Tensor l3_d, torch::Tensor l2_z,
    torch::Tensor l3_z, torch::Tensor leith2fac, torch::Tensor smag2fac,
    torch::Tensor base, double side_mask_fac) {
  const auto dtype = u.scalar_type();
  const auto device = u.device();
  TORCH_CHECK(dtype == at::kFloat || dtype == at::kDouble,
              "hvisc fused inputs must be float32 or float64");
  TORCH_CHECK(u.dim() == 4, "u must be four-dimensional");

  const auto batch = u.size(0);
  const auto nz = u.size(1);
  const auto ny = u.size(2);
  const auto nx = u.size(3) - 1;
  TORCH_CHECK(nx >= 2 && ny >= 2, "hvisc fused requires ny >= 2 and nx >= 2");
  TORCH_CHECK(batch > 0 && nz > 0, "hvisc fused extents must be positive");

  const auto want4 = [&](const torch::Tensor &value, const char *name,
                         int64_t d2, int64_t d3) {
    check_float(value, name, dtype, device);
    TORCH_CHECK(value.dim() == 4, name, " must be four-dimensional");
    TORCH_CHECK(value.size(0) == batch && value.size(1) == nz &&
                    value.size(2) == d2 && value.size(3) == d3,
                name, " has an unexpected shape");
  };
  const auto want3 = [&](const torch::Tensor &value, const char *name,
                         int64_t d1, int64_t d2) {
    check_float(value, name, dtype, device);
    TORCH_CHECK(value.dim() == 3, name, " must be three-dimensional");
    TORCH_CHECK(value.size(0) == nz && value.size(1) == d1 &&
                    value.size(2) == d2,
                name, " has an unexpected shape");
  };
  const auto want0 = [&](const torch::Tensor &value, const char *name) {
    check_float(value, name, dtype, device);
    TORCH_CHECK(value.dim() == 0, name, " must be a 0-d tensor");
  };

  want4(u, "u", ny, nx + 1);
  want4(v, "v", ny + 1, nx);
  want4(u_ext, "u_ext", ny + 2, nx + 1);
  want4(v_dyC_ext, "v_dyC_ext", ny + 1, nx + 2);

  want2(dxC_ext, "dxC_ext", dtype, device, ny + 2, nx + 1);
  want2(dxG_ext, "dxG_ext", dtype, device, ny + 1, nx);
  want2(dyG, "dyG", dtype, device, ny, nx + 1);
  want2(dyF, "dyF", dtype, device, ny, nx);
  want2(recip_rA, "recip_rA", dtype, device, ny, nx);
  want2(recip_rAz, "recip_rAz", dtype, device, ny + 1, nx + 1);

  want3(hFacW, "hFacW", ny, nx + 1);
  want3(hFacS, "hFacS", ny + 1, nx);
  want3(recip_hFacC, "recip_hFacC", ny, nx);
  want3(maskW, "maskW", ny, nx + 1);
  want3(maskS, "maskS", ny + 1, nx);
  want3(hfacz, "hFacZ", ny + 1, nx + 1);

  want2_opt(maskInC, "maskInC", dtype, device, ny, nx);
  want2_opt(maskInW, "maskInW", dtype, device, ny, nx + 1);
  want2_opt(maskInS, "maskInS", dtype, device, ny + 1, nx);

  want2(l2_d, "l2_d", dtype, device, ny, nx);
  want2(l3_d, "l3_d", dtype, device, ny, nx);
  want2(l2_z, "l2_z", dtype, device, ny + 1, nx + 1);
  want2(l3_z, "l3_z", dtype, device, ny + 1, nx + 1);

  want0(leith2fac, "leith2fac");
  want0(smag2fac, "smag2fac");
  want0(base, "base");

  // Accessor precondition (see ``unit_row``): every tensor below is read
  // through an A4/A3/A2/M2 accessor, which assumes a unit innermost stride.
  // The three 0-d factors are read through a bare pointer, so they are not
  // listed.  Keep this list consistent with the accessor call sites in
  // kernel.cu when adding a tensor to the launch.
  u = unit_row(u);
  v = unit_row(v);
  u_ext = unit_row(u_ext);
  v_dyC_ext = unit_row(v_dyC_ext);
  dxC_ext = unit_row(dxC_ext);
  dxG_ext = unit_row(dxG_ext);
  dyG = unit_row(dyG);
  dyF = unit_row(dyF);
  recip_rA = unit_row(recip_rA);
  recip_rAz = unit_row(recip_rAz);
  hFacW = unit_row(hFacW);
  hFacS = unit_row(hFacS);
  recip_hFacC = unit_row(recip_hFacC);
  maskW = unit_row(maskW);
  maskS = unit_row(maskS);
  hfacz = unit_row(hfacz);
  maskInC = unit_row_opt(maskInC);
  maskInW = unit_row_opt(maskInW);
  maskInS = unit_row_opt(maskInS);
  l2_d = unit_row(l2_d);
  l3_d = unit_row(l3_d);
  l2_z = unit_row(l2_z);
  l3_z = unit_row(l3_z);

  auto viscAh_D = torch::empty({batch, nz, ny, nx}, u.options());
  auto hDiv_flux = torch::empty({batch, nz, ny, nx}, u.options());
  auto hz_v3 = torch::empty({batch, nz, ny + 1, nx + 1}, u.options());
  hvisc_stage1_launch(u, v, u_ext, v_dyC_ext, dxC_ext, dxG_ext, dyG, dyF,
                      recip_rA, recip_rAz, hFacW, hFacS, recip_hFacC, maskW,
                      maskS, hfacz, maskInC, maskInW, maskInS, l2_d, l3_d, l2_z,
                      l3_z, leith2fac, smag2fac, base, side_mask_fac, viscAh_D,
                      hDiv_flux, hz_v3);
  return {viscAh_D, hDiv_flux, hz_v3};
}

std::vector<torch::Tensor> hvisc_fused_stage2(
    torch::Tensor hDiv_flux, std::optional<torch::Tensor> hDiv_x,
    std::optional<torch::Tensor> hDiv_y, torch::Tensor hz_v3,
    torch::Tensor dxC_row, torch::Tensor dyC, torch::Tensor dxG_ext,
    torch::Tensor dyG, torch::Tensor recip_hFacW, torch::Tensor recip_hFacS,
    torch::Tensor maskW, torch::Tensor maskS, bool u_all_faces,
    bool v_all_faces, bool keep_west, bool keep_east, bool keep_south,
    bool keep_north) {
  const auto dtype = hDiv_flux.scalar_type();
  const auto device = hDiv_flux.device();
  TORCH_CHECK(dtype == at::kFloat || dtype == at::kDouble,
              "hvisc fused inputs must be float32 or float64");
  TORCH_CHECK(hDiv_flux.dim() == 4, "hDiv_flux must be four-dimensional");

  const auto batch = hDiv_flux.size(0);
  const auto nz = hDiv_flux.size(1);
  const auto ny = hDiv_flux.size(2);
  const auto nx = hDiv_flux.size(3);
  TORCH_CHECK(nx >= 2 && ny >= 2, "hvisc fused requires ny >= 2 and nx >= 2");
  TORCH_CHECK(batch > 0 && nz > 0, "hvisc fused extents must be positive");

  const auto want4 = [&](const torch::Tensor &value, const char *name,
                         int64_t d2, int64_t d3) {
    check_float(value, name, dtype, device);
    TORCH_CHECK(value.dim() == 4, name, " must be four-dimensional");
    TORCH_CHECK(value.size(0) == batch && value.size(1) == nz &&
                    value.size(2) == d2 && value.size(3) == d3,
                name, " has an unexpected shape");
  };
  const auto want3 = [&](const torch::Tensor &value, const char *name,
                         int64_t d1, int64_t d2) {
    check_float(value, name, dtype, device);
    TORCH_CHECK(value.dim() == 3, name, " must be three-dimensional");
    TORCH_CHECK(value.size(0) == nz && value.size(1) == d1 &&
                    value.size(2) == d2,
                name, " has an unexpected shape");
  };

  want4(hDiv_flux, "hDiv_flux", ny, nx);
  want4(hz_v3, "hz_v3", ny + 1, nx + 1);
  // The halo tensors exist iff the matching all-faces branch is taken -- the
  // eager reference builds each one INSIDE that branch, so a present-but-unused
  // halo would mean the adapter ran an exchange the eager path never runs.
  TORCH_CHECK(hDiv_x.has_value() == u_all_faces,
              "hDiv_x must be supplied exactly when u_all_faces is set");
  TORCH_CHECK(hDiv_y.has_value() == v_all_faces,
              "hDiv_y must be supplied exactly when v_all_faces is set");
  if (hDiv_x.has_value()) {
    want4(*hDiv_x, "hDiv_x", ny, nx + 2);
  }
  if (hDiv_y.has_value()) {
    want4(*hDiv_y, "hDiv_y", ny + 2, nx);
  }
  if (!u_all_faces) {
    TORCH_CHECK(!keep_west && !keep_east,
                "west/east seam faces require the u all-faces branch");
  }
  if (!v_all_faces) {
    TORCH_CHECK(!keep_south && !keep_north,
                "south/north seam faces require the v all-faces branch");
  }

  want2(dxC_row, "dxC_row", dtype, device, ny, nx + 1);
  want2(dyC, "dyC", dtype, device, ny + 1, nx);
  want2(dxG_ext, "dxG_ext", dtype, device, ny + 1, nx);
  want2(dyG, "dyG", dtype, device, ny, nx + 1);
  want3(recip_hFacW, "recip_hFacW", ny, nx + 1);
  want3(recip_hFacS, "recip_hFacS", ny + 1, nx);
  want3(maskW, "maskW", ny, nx + 1);
  want3(maskS, "maskS", ny + 1, nx);

  // Accessor precondition (see ``unit_row``): stage 2 shares kernel.cu's
  // accessor structs with stage 1, so it carries the same requirement.
  hDiv_flux = unit_row(hDiv_flux);
  hDiv_x = unit_row_opt(hDiv_x);
  hDiv_y = unit_row_opt(hDiv_y);
  hz_v3 = unit_row(hz_v3);
  dxC_row = unit_row(dxC_row);
  dyC = unit_row(dyC);
  dxG_ext = unit_row(dxG_ext);
  dyG = unit_row(dyG);
  recip_hFacW = unit_row(recip_hFacW);
  recip_hFacS = unit_row(recip_hFacS);
  maskW = unit_row(maskW);
  maskS = unit_row(maskS);

  auto u_out = torch::empty({batch, nz, ny, nx + 1}, hDiv_flux.options());
  auto v_out = torch::empty({batch, nz, ny + 1, nx}, hDiv_flux.options());
  hvisc_stage2_launch(hDiv_flux, hDiv_x, hDiv_y, hz_v3, dxC_row, dyC, dxG_ext,
                      dyG, recip_hFacW, recip_hFacS, maskW, maskS, u_all_faces,
                      v_all_faces, keep_west, keep_east, keep_south, keep_north,
                      u_out, v_out);
  return {u_out, v_out};
}

std::vector<torch::Tensor> hvisc_fused_mr_pre(
    torch::Tensor u, torch::Tensor v, torch::Tensor u_ext,
    torch::Tensor v_dyC_ext, torch::Tensor dxC_ext, torch::Tensor dxG,
    torch::Tensor dyG, torch::Tensor dyF, torch::Tensor recip_rA,
    torch::Tensor recip_rAz, torch::Tensor hFacW, torch::Tensor hFacS,
    torch::Tensor recip_hFacC, torch::Tensor maskW, torch::Tensor maskS,
    torch::Tensor hfacz, std::optional<torch::Tensor> maskInC,
    std::optional<torch::Tensor> maskInW, std::optional<torch::Tensor> maskInS,
    double side_factor) {
  const auto dtype = u.scalar_type();
  const auto device = u.device();
  TORCH_CHECK(dtype == at::kFloat || dtype == at::kDouble,
              "hvisc fused inputs must be float32 or float64");
  TORCH_CHECK(u.dim() == 4, "u must be four-dimensional");

  const auto batch = u.size(0), nz = u.size(1), ny = u.size(2);
  const auto nx = u.size(3) - 1;
  TORCH_CHECK(nx >= 2 && ny >= 2, "hvisc fused requires ny >= 2 and nx >= 2");
  TORCH_CHECK(batch > 0 && nz > 0, "hvisc fused extents must be positive");

  const auto want4 = [&](const torch::Tensor &value, const char *name,
                         int64_t d2, int64_t d3) {
    check_float(value, name, dtype, device);
    TORCH_CHECK(value.dim() == 4, name, " must be four-dimensional");
    TORCH_CHECK(value.size(0) == batch && value.size(1) == nz &&
                    value.size(2) == d2 && value.size(3) == d3,
                name, " has an unexpected shape");
  };
  const auto want3 = [&](const torch::Tensor &value, const char *name,
                         int64_t d1, int64_t d2) {
    check_float(value, name, dtype, device);
    TORCH_CHECK(value.dim() == 3, name, " must be three-dimensional");
    TORCH_CHECK(value.size(0) == nz && value.size(1) == d1 &&
                    value.size(2) == d2,
                name, " has an unexpected shape");
  };

  want4(u, "u", ny, nx + 1);
  want4(v, "v", ny + 1, nx);
  want4(u_ext, "u_ext", ny + 2, nx + 1);
  want4(v_dyC_ext, "v_dyC_ext", ny + 1, nx + 2);
  want2(dxC_ext, "dxC_ext", dtype, device, ny + 2, nx + 1);
  want2(dxG, "dxG", dtype, device, ny + 1, nx);
  want2(dyG, "dyG", dtype, device, ny, nx + 1);
  want2(dyF, "dyF", dtype, device, ny, nx);
  want2(recip_rA, "recip_rA", dtype, device, ny, nx);
  want2(recip_rAz, "recip_rAz", dtype, device, ny + 1, nx + 1);
  want3(hFacW, "hFacW", ny, nx + 1);
  want3(hFacS, "hFacS", ny + 1, nx);
  want3(recip_hFacC, "recip_hFacC", ny, nx);
  want3(maskW, "maskW", ny, nx + 1);
  want3(maskS, "maskS", ny + 1, nx);
  want3(hfacz, "hFacZ", ny + 1, nx + 1);
  want2_opt(maskInC, "maskInC", dtype, device, ny, nx);
  want2_opt(maskInW, "maskInW", dtype, device, ny, nx + 1);
  want2_opt(maskInS, "maskInS", dtype, device, ny + 1, nx);

  auto hdiv = torch::empty({batch, nz, ny, nx}, u.options());
  auto vort = torch::empty({batch, nz, ny + 1, nx + 1}, u.options());
  auto strain = torch::empty_like(vort);
  auto tension = torch::empty_like(hdiv);
  auto vrtDx = torch::empty_like(vort);
  auto vrtDy = torch::empty_like(vort);
  hvisc_mr_pre_launch(u, v, u_ext, v_dyC_ext, dxC_ext, dxG, dyG, dyF,
                      recip_rA, recip_rAz, hFacW, hFacS, recip_hFacC, maskW,
                      maskS, hfacz, maskInC, maskInW, maskInS, side_factor,
                      hdiv, vort, strain, tension, vrtDx, vrtDy);
  return {hdiv, vort, strain, tension, vrtDx, vrtDy};
}

std::vector<torch::Tensor> hvisc_fused_mr_coeff(
    torch::Tensor hdiv, torch::Tensor vort, torch::Tensor strain,
    torch::Tensor vrtDx_x, torch::Tensor vrtDy_y, torch::Tensor tension_xy,
    torch::Tensor hfacz, torch::Tensor l2_d, torch::Tensor l3_d,
    torch::Tensor l2_z, torch::Tensor l3_z, torch::Tensor leith,
    torch::Tensor smag, torch::Tensor base) {
  const auto dtype = hdiv.scalar_type();
  const auto device = hdiv.device();
  TORCH_CHECK(dtype == at::kFloat || dtype == at::kDouble,
              "hvisc fused inputs must be float32 or float64");
  TORCH_CHECK(hdiv.dim() == 4, "hdiv must be four-dimensional");

  const auto batch = hdiv.size(0), nz = hdiv.size(1), ny = hdiv.size(2);
  const auto nx = hdiv.size(3);
  TORCH_CHECK(nx >= 2 && ny >= 2, "hvisc fused requires ny >= 2 and nx >= 2");
  TORCH_CHECK(batch > 0 && nz > 0, "hvisc fused extents must be positive");

  const auto want4 = [&](const torch::Tensor &value, const char *name,
                         int64_t d2, int64_t d3) {
    check_float(value, name, dtype, device);
    TORCH_CHECK(value.dim() == 4, name, " must be four-dimensional");
    TORCH_CHECK(value.size(0) == batch && value.size(1) == nz &&
                    value.size(2) == d2 && value.size(3) == d3,
                name, " has an unexpected shape");
  };
  const auto want3 = [&](const torch::Tensor &value, const char *name,
                         int64_t d1, int64_t d2) {
    check_float(value, name, dtype, device);
    TORCH_CHECK(value.dim() == 3, name, " must be three-dimensional");
    TORCH_CHECK(value.size(0) == nz && value.size(1) == d1 &&
                    value.size(2) == d2,
                name, " has an unexpected shape");
  };
  const auto want0 = [&](const torch::Tensor &value, const char *name) {
    check_float(value, name, dtype, device);
    TORCH_CHECK(value.dim() == 0, name, " must be a 0-d tensor");
  };

  want4(hdiv, "hdiv", ny, nx);
  want4(vort, "vort3", ny + 1, nx + 1);
  want4(strain, "strain_bc", ny + 1, nx + 1);
  want4(vrtDx_x, "vrtDx_x", ny + 1, nx + 2);
  want4(vrtDy_y, "vrtDy_y", ny + 2, nx + 1);
  want4(tension_xy, "tension_xy", ny + 2, nx + 2);
  want3(hfacz, "hFacZ", ny + 1, nx + 1);
  want2(l2_d, "l2_d", dtype, device, ny, nx);
  want2(l3_d, "l3_d", dtype, device, ny, nx);
  want2(l2_z, "l2_z", dtype, device, ny + 1, nx + 1);
  want2(l3_z, "l3_z", dtype, device, ny + 1, nx + 1);
  want0(leith, "leith2fac");
  want0(smag, "smag2fac");
  want0(base, "base");

  auto visc_d = torch::empty_like(hdiv);
  auto flux = torch::empty_like(hdiv);
  auto hz = torch::empty({batch, nz, ny + 1, nx + 1}, hdiv.options());
  hvisc_mr_coeff_launch(hdiv, vort, strain, vrtDx_x, vrtDy_y, tension_xy,
                        hfacz, l2_d, l3_d, l2_z, l3_z, leith, smag, base,
                        visc_d, flux, hz);
  return {visc_d, flux, hz};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("hvisc_fused_stage1", &hvisc_fused_stage1,
             "Leith/Smag harmonic viscosity coefficient assembly (CUDA)");
  module.def("hvisc_fused_stage2", &hvisc_fused_stage2,
             "MIT div-vort dissipation stencil (CUDA)");
  module.def("hvisc_fused_mr_pre", &hvisc_fused_mr_pre,
             "Multi-rank pre-exchange intermediates (CUDA)");
  module.def("hvisc_fused_mr_coeff", &hvisc_fused_mr_coeff,
             "Multi-rank exchanged coefficient assembly (CUDA)");
}
