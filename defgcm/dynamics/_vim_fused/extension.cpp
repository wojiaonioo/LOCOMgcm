#include <torch/extension.h>

#include <optional>
#include <vector>

void vim_fused_launch(const torch::Tensor &ke_x_ext,
                      const torch::Tensor &ke_y_ext,
                      const torch::Tensor &u_corner,
                      const torch::Tensor &v_corner,
                      const torch::Tensor &v_x_halo,
                      const torch::Tensor &u_y_halo, const torch::Tensor &hfacz,
                      const torch::Tensor &fg, const torch::Tensor &dxC_ext,
                      const torch::Tensor &dyC, const torch::Tensor &rAz,
                      const torch::Tensor &dxG_south,
                      const torch::Tensor &dxG_north,
                      const torch::Tensor &recip_dxC_u, const torch::Tensor &dyG,
                      const torch::Tensor &recip_dyC, const torch::Tensor &maskW,
                      const torch::Tensor &maskS,
                      const torch::Tensor &x_corner_mask,
                      const torch::Tensor &y_corner_mask,
                      const std::optional<torch::Tensor> &edge_pair_mask_x,
                      const std::optional<torch::Tensor> &edge_pair_mask_y,
                      bool periodic_x_copy, bool periodic_y_copy,
                      torch::Tensor &tend_u, torch::Tensor &tend_v);

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

void check_bool(const torch::Tensor &value, const char *name, int64_t length,
                const torch::Device &device) {
  TORCH_CHECK(value.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(value.device() == device, name, " must share one CUDA device");
  TORCH_CHECK(value.scalar_type() == at::kBool, name, " must be bool");
  TORCH_CHECK(value.dim() == 1 && value.size(0) == length, name,
              " must be a 1-D mask of the corner length");
  TORCH_CHECK(value.stride(0) >= 0, name, " must not have negative strides");
}

} // namespace

std::vector<torch::Tensor> vim_fused_fwd(
    torch::Tensor ke_x_ext, torch::Tensor ke_y_ext, torch::Tensor u_corner,
    torch::Tensor v_corner, torch::Tensor v_x_halo, torch::Tensor u_y_halo,
    torch::Tensor hfacz, torch::Tensor fg, torch::Tensor dxC_ext,
    torch::Tensor dyC, torch::Tensor rAz, torch::Tensor dxG_south,
    torch::Tensor dxG_north, torch::Tensor recip_dxC_u, torch::Tensor dyG,
    torch::Tensor recip_dyC, torch::Tensor maskW, torch::Tensor maskS,
    torch::Tensor x_corner_mask, torch::Tensor y_corner_mask,
    std::optional<torch::Tensor> edge_pair_mask_x,
    std::optional<torch::Tensor> edge_pair_mask_y, bool periodic_x_copy,
    bool periodic_y_copy) {
  const auto dtype = ke_x_ext.scalar_type();
  const auto device = ke_x_ext.device();
  TORCH_CHECK(dtype == at::kFloat || dtype == at::kDouble,
              "VIM fused inputs must be float32 or float64");
  TORCH_CHECK(ke_x_ext.dim() == 4, "ke_x_ext must be four-dimensional");

  const auto batch = ke_x_ext.size(0);
  const auto nz = ke_x_ext.size(1);
  const auto ny = ke_x_ext.size(2);
  const auto nx = ke_x_ext.size(3) - 2;
  TORCH_CHECK(nx >= 2 && ny >= 2, "VIM fused requires ny >= 2 and nx >= 2");
  TORCH_CHECK(batch > 0 && nz > 0, "VIM fused extents must be positive");

  const auto want4 = [&](const torch::Tensor &value, const char *name,
                         int64_t d2, int64_t d3) {
    check_float(value, name, dtype, device);
    TORCH_CHECK(value.dim() == 4, name, " must be four-dimensional");
    TORCH_CHECK(value.size(0) == batch && value.size(1) == nz &&
                    value.size(2) == d2 && value.size(3) == d3,
                name, " has an unexpected shape");
  };

  want4(ke_x_ext, "ke_x_ext", ny, nx + 2);
  want4(ke_y_ext, "ke_y_ext", ny + 2, nx);
  want4(u_corner, "u_corner", ny + 2, nx + 3);
  want4(v_corner, "v_corner", ny + 3, nx + 2);
  want4(v_x_halo, "v_x_halo", ny + 1, nx + 2);
  want4(u_y_halo, "u_y_halo", ny + 2, nx + 1);

  const auto want = [&](const torch::Tensor &value, const char *name,
                        std::vector<int64_t> shape) {
    check_float(value, name, dtype, device);
    TORCH_CHECK(value.sizes() == at::IntArrayRef(shape), name,
                " has an unexpected shape");
  };

  want(hfacz, "hFacZ", {nz, ny + 1, nx + 1});
  want(fg, "fCoriG", {ny + 1, nx + 1});
  want(dxC_ext, "dxC_ext", {ny + 2, nx + 1});
  want(dyC, "dyC", {ny + 1, nx});
  want(rAz, "rAz", {ny + 1, nx + 1});
  want(dxG_south, "dxG_south", {ny, 1});
  want(dxG_north, "dxG_north", {ny, 1});
  want(recip_dxC_u, "recip_dxC_u", {ny, 1});
  want(dyG, "dyG", {});
  want(recip_dyC, "recip_dyC", {});
  want(maskW, "maskW", {nz, ny, nx + 1});
  want(maskS, "maskS", {nz, ny + 1, nx});

  check_bool(x_corner_mask, "x_corner_mask", nx + 1, device);
  check_bool(y_corner_mask, "y_corner_mask", ny + 1, device);
  if (edge_pair_mask_x.has_value()) {
    check_bool(*edge_pair_mask_x, "edge_pair_mask_x", nx + 1, device);
  }
  if (edge_pair_mask_y.has_value()) {
    check_bool(*edge_pair_mask_y, "edge_pair_mask_y", ny + 1, device);
  }

  auto tend_u = torch::empty({batch, nz, ny, nx + 1}, ke_x_ext.options());
  auto tend_v = torch::empty({batch, nz, ny + 1, nx}, ke_x_ext.options());
  vim_fused_launch(ke_x_ext, ke_y_ext, u_corner, v_corner, v_x_halo, u_y_halo,
                   hfacz, fg, dxC_ext, dyC, rAz, dxG_south, dxG_north,
                   recip_dxC_u, dyG, recip_dyC, maskW, maskS, x_corner_mask,
                   y_corner_mask, edge_pair_mask_x, edge_pair_mask_y,
                   periodic_x_copy, periodic_y_copy, tend_u, tend_v);
  return {tend_u, tend_v};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("vim_fused_fwd", &vim_fused_fwd,
             "Horizontal-phase fused vector-invariant momentum (CUDA)");
}
