#include <torch/extension.h>

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "../../solvers/_cuda_launch/pybind.h"

void kpp_blmix_launch(
    const torch::Tensor& ustar,
    const torch::Tensor& bfsfc,
    const torch::Tensor& hbl,
    const torch::Tensor& stable,
    const torch::Tensor& casea,
    const torch::Tensor& kbl,
    const torch::Tensor& diffus,
    const torch::Tensor& zgrid,
    const torch::Tensor& hwide,
    const torch::Tensor& wm_hbl,
    const torch::Tensor& ws_hbl,
    const torch::Tensor& wm_levels,
    const torch::Tensor& ws_levels,
    const torch::Tensor& wm_d,
    const torch::Tensor& ws_d,
    torch::Tensor& blmc,
    torch::Tensor& ghat,
    torch::Tensor& dkm1,
    double conc1,
    double cg,
    double phepsi);

void kpp_blmix_backward_launch(
    const std::vector<torch::Tensor>& values,
    const std::vector<c10::optional<torch::Tensor>>& seeds,
    std::vector<c10::optional<torch::Tensor>>& bars,
    double conc1, double cg, double phepsi);

const defgcm::cuda_launch::LaunchConfig&
kpp_blmix_cuda_launch_config(int device, at::ScalarType dtype);

namespace {

at::ScalarType parse_scalar_type(const std::string& name) {
  if (name == "float64") {
    return at::kDouble;
  }
  if (name == "float32") {
    return at::kFloat;
  }
  TORCH_CHECK(false, "kpp_blmix CUDA dtype must be float64 or float32");
  return at::kDouble;
}

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
// The kernel is instantiated for both floating dtypes; this predicate names
// exactly the set AT_DISPATCH_FLOATING_TYPES in kernel.cu covers.
#define CHECK_FLOATING(x) \
  TORCH_CHECK( \
      (x).scalar_type() == at::kDouble || (x).scalar_type() == at::kFloat, \
      #x " must be float64 or float32")
#define CHECK_LONG(x) \
  TORCH_CHECK((x).scalar_type() == at::kLong, #x " must be int64")

void check_tensor(
    const torch::Tensor& value,
    const torch::Device& device,
    const int64_t dimensions,
    const char* name,
    const bool integer = false) {
  CHECK_CUDA(value);
  if (integer) {
    CHECK_LONG(value);
  } else {
    CHECK_FLOATING(value);
  }
  TORCH_CHECK(value.device() == device, name, " must be on the ustar device");
  TORCH_CHECK(value.dim() == dimensions, name, " has an invalid rank");
  for (int dimension = 0; dimension < value.dim(); ++dimension) {
    TORCH_CHECK(
        value.stride(dimension) >= 0,
        name,
        " must not have negative strides");
  }
}

void check_shape2(
    const torch::Tensor& value,
    const int64_t ny,
    const int64_t nx,
    const char* name) {
  TORCH_CHECK(
      value.size(0) == ny && value.size(1) == nx,
      name,
      " has an invalid shape");
}

void check_shape3(
    const torch::Tensor& value,
    const int64_t levels,
    const int64_t ny,
    const int64_t nx,
    const char* name) {
  TORCH_CHECK(
      value.size(0) == levels && value.size(1) == ny && value.size(2) == nx,
      name,
      " has an invalid shape");
}

void check_profile_inputs(const torch::Tensor& ustar,
    const torch::Tensor& bfsfc,
    const torch::Tensor& hbl,
    const torch::Tensor& stable,
    const torch::Tensor& casea,
    const torch::Tensor& kbl,
    const torch::Tensor& diffus,
    const torch::Tensor& zgrid,
    const torch::Tensor& hwide,
    const torch::Tensor& wm_hbl,
    const torch::Tensor& ws_hbl,
    const torch::Tensor& wm_levels,
    const torch::Tensor& ws_levels,
    const torch::Tensor& wm_d,
    const torch::Tensor& ws_d) {
  CHECK_CUDA(ustar);
  CHECK_FLOATING(ustar);
  TORCH_CHECK(ustar.dim() == 2, "ustar must be two-dimensional");
  TORCH_CHECK(
      ustar.size(0) > 0 && ustar.size(1) > 0,
      "ustar extents must be positive");
  for (int dimension = 0; dimension < ustar.dim(); ++dimension) {
    TORCH_CHECK(
        ustar.stride(dimension) >= 0,
        "ustar must not have negative strides");
  }
  const auto device = ustar.device();
  const auto ny = ustar.size(0);
  const auto nx = ustar.size(1);

  for (const auto* value : {
           &bfsfc, &hbl, &stable, &casea, &wm_hbl, &ws_hbl, &wm_d, &ws_d}) {
    check_tensor(*value, device, 2, "column input");
    check_shape2(*value, ny, nx, "column input");
  }
  check_tensor(kbl, device, 2, "kbl", true);
  check_shape2(kbl, ny, nx, "kbl");
  check_tensor(diffus, device, 4, "diffus");
  TORCH_CHECK(diffus.size(0) == 3, "diffus must have three components");
  TORCH_CHECK(diffus.size(1) >= 3, "diffus must contain at least one KPP level");
  TORCH_CHECK(
      diffus.size(2) == ny && diffus.size(3) == nx,
      "diffus has an invalid horizontal shape");
  const auto nz = diffus.size(1) - 2;
  TORCH_CHECK(
      nz <= std::numeric_limits<int>::max(),
      "KPP level count exceeds CUDA kernel integer capacity");

  for (const auto* value : {&zgrid, &hwide}) {
    check_tensor(*value, device, 1, "vertical profile");
    TORCH_CHECK(value->size(0) == nz + 2, "vertical profile has an invalid shape");
  }
  for (const auto* value : {&wm_levels, &ws_levels}) {
    check_tensor(*value, device, 3, "level velocity scale");
    check_shape3(*value, nz, ny, nx, "level velocity scale");
  }
  // Every floating input feeds one instantiation, so they share one dtype.
  for (const auto* value : {
           &bfsfc, &hbl, &stable, &casea, &wm_hbl, &ws_hbl, &wm_d, &ws_d,
           &diffus, &zgrid, &hwide, &wm_levels, &ws_levels}) {
    TORCH_CHECK(
        value->scalar_type() == ustar.scalar_type(),
        "every floating KPP blmix input must share the ustar dtype");
  }
}

}  // namespace

std::vector<torch::Tensor> kpp_blmix_fwd(
    torch::Tensor ustar,
    torch::Tensor bfsfc,
    torch::Tensor hbl,
    torch::Tensor stable,
    torch::Tensor casea,
    torch::Tensor kbl,
    torch::Tensor diffus,
    torch::Tensor zgrid,
    torch::Tensor hwide,
    torch::Tensor wm_hbl,
    torch::Tensor ws_hbl,
    torch::Tensor wm_levels,
    torch::Tensor ws_levels,
    torch::Tensor wm_d,
    torch::Tensor ws_d,
    double conc1,
    double cg,
    double phepsi) {
  check_profile_inputs(ustar, bfsfc, hbl, stable, casea, kbl, diffus, zgrid, hwide, wm_hbl, ws_hbl, wm_levels, ws_levels, wm_d, ws_d);
  const auto ny=ustar.size(0), nx=ustar.size(1), nz=diffus.size(1)-2;
  auto blmc = torch::empty({3, nz, ny, nx}, ustar.options());
  auto ghat = torch::empty({nz, ny, nx}, ustar.options());
  auto dkm1 = torch::empty({3, ny, nx}, ustar.options());
  kpp_blmix_launch(
      ustar,
      bfsfc,
      hbl,
      stable,
      casea,
      kbl,
      diffus,
      zgrid,
      hwide,
      wm_hbl,
      ws_hbl,
      wm_levels,
      ws_levels,
      wm_d,
      ws_d,
      blmc,
      ghat,
      dkm1,
      conc1,
      cg,
      phepsi);
  return {blmc, ghat, dkm1};
}

std::vector<c10::optional<torch::Tensor>> kpp_blmix_bwd(
    std::vector<torch::Tensor> values, std::vector<c10::optional<torch::Tensor>> seeds,
    std::vector<bool> needs, double conc1, double cg, double phepsi) {
  TORCH_CHECK(values.size()==15 && needs.size()==15 && seeds.size()==3,
              "BLMIX backward expects fifteen inputs/needs and three seeds");
  check_profile_inputs(values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7], values[8], values[9], values[10], values[11], values[12], values[13], values[14]);
  TORCH_CHECK(!needs[5] && !needs[7] && !needs[8],
              "BLMIX native profile does not differentiate kbl/zgrid/hwide; use Torch");
  const auto ny=values[0].size(0), nx=values[0].size(1), nz=values[6].size(1)-2;
  const std::vector<std::vector<int64_t>> shapes={{3,nz,ny,nx},{nz,ny,nx},{3,ny,nx}};
  bool seeded=false, active=false;
  for(int i=0;i<3;++i) if(seeds[i]) {
    check_tensor(*seeds[i],values[0].device(),i==0?4:3,"output seed");
    TORCH_CHECK(seeds[i]->sizes()==at::IntArrayRef(shapes[i]),"BLMIX seed shape mismatch");
    TORCH_CHECK(seeds[i]->scalar_type()==values[0].scalar_type(),"BLMIX seed dtype mismatch");
    seeded=true;
  }
  std::vector<c10::optional<torch::Tensor>> bars(15);
  for(int i=0;i<15;++i) if(needs[i]) {
    bars[i]=torch::empty(values[i].sizes(),values[i].options()); active=true;
  }
  if(active && seeded) kpp_blmix_backward_launch(values,seeds,bars,conc1,cg,phepsi);
  else if(active) for(auto& bar:bars) if(bar) bar->zero_();
  return bars;
}

pybind11::dict launch_config(int device, const std::string& dtype) {
  pybind11::dict result = defgcm::cuda_launch::to_python(
      kpp_blmix_cuda_launch_config(device, parse_scalar_type(dtype)));
  result["dtype"] = dtype;
  return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "launch_config",
      &launch_config,
      "Cached KPP blmix CUDA launch configuration",
      pybind11::arg("device"),
      pybind11::arg("dtype") = "float64");
  m.def("kpp_blmix_bwd", &kpp_blmix_bwd, "BLMIX native profile VJP (CUDA)");
  m.def(
      "kpp_blmix_fwd",
      &kpp_blmix_fwd,
      "Fused KPP boundary-layer mixing profile (CUDA)");
}
