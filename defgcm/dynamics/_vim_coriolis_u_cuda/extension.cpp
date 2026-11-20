// vim coriolis u -- pybind entry point and input validation.
//
// The Python decline predicate selects dispatch; TORCH_CHECK validates
// the same input requirements again at the native entry point.
// The launcher and arithmetic are implemented in kernel.cu.

#include <torch/extension.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <vector>

#include "../../solvers/_cuda_launch/pybind.h"

// Defined in kernel.cu.
void vim_coriolis_u_launch(
    const torch::Tensor &v_ext,
    const torch::Tensor &corner_value,
    const torch::Tensor &dxg_south,
    const torch::Tensor &dxg_north,
    const torch::Tensor &recip_dxc_u,
    const torch::Tensor &hfacz,
    const torch::Tensor &face_mask,
    const torch::Tensor &edge_pair_mask,
    torch::Tensor &u_face_coriolis,
    bool edge_face_couples
);

const defgcm::cuda_launch::LaunchConfig &vim_coriolis_u_launch_config(int device);

namespace {

std::atomic<int64_t> native_launch_count{0};

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_FLOATING(x)                                                      \
  TORCH_CHECK((x).scalar_type() == at::kDouble,                                               \
              #x " must be float64")
#define CHECK_RANK(x)                                                          \
  TORCH_CHECK((x).dim() == 0 || (x).dim() == 1 || (x).dim() == 2 || (x).dim() == 3 || (x).dim() == 4,                                                \
              #x " must have ndim in [0, 1, 2, 3, 4]")
#define CHECK_STRIDED(x)                                                       \
  TORCH_CHECK((x).layout() == c10::kStrided, #x " must use strided layout")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_FLOATING(x);                                                           \
  CHECK_RANK(x);                                                               \
  CHECK_STRIDED(x)
// An input carrying its own dtype -- a bool mask beside float data -- is
// checked against that dtype, never against the kernel's floating one.
#define CHECK_TYPED_INPUT(x, ty)                                               \
  CHECK_CUDA(x);                                                               \
  TORCH_CHECK((x).scalar_type() == ty, #x " must be " #ty);                    \
  CHECK_RANK(x);                                                               \
  CHECK_STRIDED(x)

void check_non_negative_strides(const torch::Tensor &value, const char *name) {
  for (int dimension = 0; dimension < value.dim(); ++dimension) {
    TORCH_CHECK(value.stride(dimension) >= 0, name,
                " must not have negative strides");
  }
}

// Device and extents are shared by every input; dtype is NOT, so it is checked
// per input by the macros above rather than against the reference here.
void check_companion(const torch::Tensor &value, const torch::Tensor &reference,
                     const char *name) {
  TORCH_CHECK(value.device() == reference.device(), name,
              " must be on the same CUDA device as v_ext");
  TORCH_CHECK(value.numel() > 0, name, " must have positive extents");
  check_non_negative_strides(value, name);
}

} // namespace

torch::Tensor vim_coriolis_u_fwd(
    torch::Tensor v_ext,
    torch::Tensor corner_value,
    torch::Tensor dxg_south,
    torch::Tensor dxg_north,
    torch::Tensor recip_dxc_u,
    c10::optional<torch::Tensor> hfacz_arg,
    c10::optional<torch::Tensor> face_mask_arg,
    c10::optional<torch::Tensor> edge_pair_mask_arg,
    bool edge_face_couples) {
  torch::Tensor hfacz = hfacz_arg.value_or(torch::Tensor());
  torch::Tensor face_mask = face_mask_arg.value_or(torch::Tensor());
  torch::Tensor edge_pair_mask = edge_pair_mask_arg.value_or(torch::Tensor());
  CHECK_INPUT(v_ext);
  CHECK_INPUT(corner_value);
  CHECK_INPUT(dxg_south);
  CHECK_INPUT(dxg_north);
  CHECK_INPUT(recip_dxc_u);
  check_companion(corner_value, v_ext, "corner_value");
  check_companion(dxg_south, v_ext, "dxG_south");
  check_companion(dxg_north, v_ext, "dxG_north");
  check_companion(recip_dxc_u, v_ext, "recip_dxC_u");
  TORCH_CHECK(corner_value.scalar_type() == v_ext.scalar_type(), "corner_value must have the same dtype as v_ext");
  TORCH_CHECK(dxg_south.scalar_type() == v_ext.scalar_type(), "dxG_south must have the same dtype as v_ext");
  TORCH_CHECK(dxg_north.scalar_type() == v_ext.scalar_type(), "dxG_north must have the same dtype as v_ext");
  TORCH_CHECK(recip_dxc_u.scalar_type() == v_ext.scalar_type(), "recip_dxC_u must have the same dtype as v_ext");
  if (hfacz.defined()) {
    CHECK_INPUT(hfacz);
    check_companion(hfacz, v_ext, "hFacZ");
  }
  if (face_mask.defined()) {
    CHECK_INPUT(face_mask);
    check_companion(face_mask, v_ext, "face_mask");
  }
  if (edge_pair_mask.defined()) {
    CHECK_TYPED_INPUT(edge_pair_mask, at::kBool);
    check_companion(edge_pair_mask, v_ext, "edge_pair_mask");
  }

  // Second line of defence for the leading batch: the output below is
  // sized from the LARGER leading extent and kernel.cu's broadcast_strides4
  // zeroes an input's batch stride only when that input's own extent is 1, so
  // two different non-unit extents walk the shorter input past its end.  The
  // Python predicate declines this before dispatch; this refuses to run on it.
  TORCH_CHECK(v_ext.dim() != 4 || corner_value.dim() != 4 ||
                  v_ext.size(0) == 1 || corner_value.size(0) == 1 ||
                  v_ext.size(0) == corner_value.size(0),
              "vim_coriolis_u leading batch dimensions are not "
              "broadcast-compatible");

  // Output extents and the launch call.
  std::vector<int64_t> sizes = v_ext.sizes().vec();
  if (corner_value.dim() == 4 && corner_value.size(0) > sizes[0]) {
    sizes[0] = corner_value.size(0);
  }
  sizes[sizes.size() - 2] = recip_dxc_u.size(-2);
  sizes[sizes.size() - 1] = corner_value.size(-1);
  auto u_face_coriolis = torch::empty(sizes, v_ext.options());
  vim_coriolis_u_launch(v_ext, corner_value, dxg_south, dxg_north,
                        recip_dxc_u, hfacz, face_mask, edge_pair_mask,
                        u_face_coriolis, edge_face_couples);
  native_launch_count.fetch_add(1, std::memory_order_relaxed);
  return u_face_coriolis;
}

pybind11::dict launch_config(int device) {
  return defgcm::cuda_launch::to_python(vim_coriolis_u_launch_config(device));
}

int64_t get_native_launch_count() {
  return native_launch_count.load(std::memory_order_relaxed);
}

void reset_native_launch_count() {
  native_launch_count.store(0, std::memory_order_relaxed);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("launch_config", &launch_config,
        "Cached vim coriolis u CUDA launch configuration");
  m.def("vim_coriolis_u_fwd", &vim_coriolis_u_fwd,
        "Fused vim coriolis u numerical core (CUDA)");
  m.def("native_launch_count", &get_native_launch_count,
        "Number of native vim_coriolis_u CUDA launches requested");
  m.def("reset_native_launch_count", &reset_native_launch_count,
        "Reset the native vim_coriolis_u CUDA launch counter");
}
