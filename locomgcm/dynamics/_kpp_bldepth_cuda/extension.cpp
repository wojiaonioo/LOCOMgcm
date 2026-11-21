#include <torch/extension.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "../../solvers/_cuda_launch/pybind.h"

void kpp_bldepth_launch(
    const torch::Tensor& kmtj,
    const torch::Tensor& dvsq,
    const torch::Tensor& dbloc,
    const torch::Tensor& ritop,
    const torch::Tensor& ustar,
    const torch::Tensor& bo,
    const torch::Tensor& bosol,
    const torch::Tensor& coriol,
    const torch::Tensor& zgrid,
    const torch::Tensor& hwide,
    const torch::Tensor& wst,
    torch::Tensor& hbl,
    torch::Tensor& kbl,
    torch::Tensor& bfsfc,
    torch::Tensor& stable,
    torch::Tensor& casea,
    torch::Tensor& rib,
    double ricr,
    double epsilon,
    double vtc,
    double cekman,
    double cmonob,
    double min_kpp_hbl,
    double vonk,
    double phepsi,
    bool limit_hbl_stable,
    int jwtype,
    bool penetrating_sw,
    bool sign_zero_positive);

const locomgcm::cuda_launch::LaunchConfig&
kpp_bldepth_cuda_launch_config(int device, int nz, at::ScalarType dtype);

namespace {

at::ScalarType parse_scalar_type(const std::string& name) {
  if (name == "float64") {
    return at::kDouble;
  }
  if (name == "float32") {
    return at::kFloat;
  }
  TORCH_CHECK(false, "kpp_bldepth CUDA dtype must be float64 or float32");
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
#define CHECK_DIM(x, n) \
  TORCH_CHECK((x).dim() == (n), #x " must be " #n "-dimensional")

void check_nonnegative_strides(const torch::Tensor& value, const char* name) {
  for (int dimension = 0; dimension < value.dim(); ++dimension) {
    TORCH_CHECK(
        value.stride(dimension) >= 0,
        name,
        " must not have negative strides");
  }
}

void check_same_device(
    const torch::Tensor& value,
    const torch::Tensor& reference,
    const char* name) {
  TORCH_CHECK(value.device() == reference.device(), name, " is on the wrong device");
}

void check_horizontal_shape(
    const torch::Tensor& value,
    int64_t ny,
    int64_t nx,
    const char* name) {
  TORCH_CHECK(
      value.size(0) == ny && value.size(1) == nx,
      name,
      " has an invalid horizontal shape");
}

}  // namespace

std::vector<torch::Tensor> kpp_bldepth_fwd(
    torch::Tensor kmtj,
    torch::Tensor dvsq,
    torch::Tensor dbloc,
    torch::Tensor ritop,
    torch::Tensor ustar,
    torch::Tensor bo,
    torch::Tensor bosol,
    torch::Tensor coriol,
    torch::Tensor zgrid,
    torch::Tensor hwide,
    torch::Tensor wmt,
    torch::Tensor wst,
    double ricr,
    double epsilon,
    double vtc,
    double cekman,
    double cmonob,
    double min_kpp_hbl,
    double vonk,
    double phepsi,
    bool limit_hbl_stable,
    int64_t jwtype,
    bool penetrating_sw,
    bool sign_zero_positive) {
  CHECK_CUDA(kmtj);
  CHECK_LONG(kmtj);
  CHECK_DIM(kmtj, 2);
  CHECK_CUDA(dvsq);
  CHECK_FLOATING(dvsq);
  CHECK_DIM(dvsq, 3);

  const auto nz = dvsq.size(0);
  const auto ny = dvsq.size(1);
  const auto nx = dvsq.size(2);
  TORCH_CHECK(
      nz == 90 && ny > 0 && nx > 0,
      "kpp_bldepth CUDA supports non-empty (90, NY, NX) columns only");
  check_same_device(kmtj, dvsq, "kmtj");
  check_horizontal_shape(kmtj, ny, nx, "kmtj");

  const std::vector<std::pair<torch::Tensor*, const char*>> floating_inputs = {
      {&dvsq, "dvsq"},
      {&dbloc, "dbloc"},
      {&ritop, "Ritop"},
      {&ustar, "ustar"},
      {&bo, "bo"},
      {&bosol, "bosol"},
      {&coriol, "coriol"},
      {&zgrid, "zgrid"},
      {&hwide, "hwide"},
      {&wmt, "wmt"},
      {&wst, "wst"},
  };
  check_nonnegative_strides(kmtj, "kmtj");
  for (const auto& entry : floating_inputs) {
    const auto& value = *entry.first;
    CHECK_CUDA(value);
    CHECK_FLOATING(value);
    // Every floating input feeds one instantiation, so they share one dtype.
    TORCH_CHECK(
        value.scalar_type() == dvsq.scalar_type(),
        entry.second,
        " must have the same dtype as dvsq");
    check_same_device(value, dvsq, entry.second);
    check_nonnegative_strides(value, entry.second);
  }

  CHECK_DIM(dbloc, 3);
  CHECK_DIM(ritop, 3);
  TORCH_CHECK(dbloc.sizes() == dvsq.sizes(), "dbloc must match dvsq shape");
  TORCH_CHECK(ritop.sizes() == dvsq.sizes(), "Ritop must match dvsq shape");
  CHECK_DIM(ustar, 2);
  CHECK_DIM(bo, 2);
  CHECK_DIM(bosol, 2);
  CHECK_DIM(coriol, 2);
  check_horizontal_shape(ustar, ny, nx, "ustar");
  check_horizontal_shape(bo, ny, nx, "bo");
  check_horizontal_shape(bosol, ny, nx, "bosol");
  check_horizontal_shape(coriol, ny, nx, "coriol");
  CHECK_DIM(zgrid, 1);
  CHECK_DIM(hwide, 1);
  TORCH_CHECK(zgrid.size(0) == nz + 2, "zgrid must have nz+2 entries");
  TORCH_CHECK(hwide.size(0) == nz + 2, "hwide must have nz+2 entries");
  CHECK_DIM(wmt, 2);
  CHECK_DIM(wst, 2);
  TORCH_CHECK(
      wmt.size(0) == 892 && wmt.size(1) == 482,
      "wmt must have the default KPP table shape (892, 482)");
  TORCH_CHECK(
      wst.sizes() == wmt.sizes(),
      "wst must match the wmt table shape");
  TORCH_CHECK(jwtype >= 1 && jwtype <= 5, "jwtype must be in [1, 5]");
  auto hbl = torch::empty(ustar.sizes(), dvsq.options());
  auto kbl = torch::empty(kmtj.sizes(), kmtj.options());
  auto bfsfc = torch::empty(ustar.sizes(), dvsq.options());
  auto stable = torch::empty(ustar.sizes(), dvsq.options());
  auto casea = torch::empty(ustar.sizes(), dvsq.options());
  auto rib = torch::zeros(dvsq.sizes(), dvsq.options());
  kpp_bldepth_launch(
      kmtj,
      dvsq,
      dbloc,
      ritop,
      ustar,
      bo,
      bosol,
      coriol,
      zgrid,
      hwide,
      wst,
      hbl,
      kbl,
      bfsfc,
      stable,
      casea,
      rib,
      ricr,
      epsilon,
      vtc,
      cekman,
      cmonob,
      min_kpp_hbl,
      vonk,
      phepsi,
      limit_hbl_stable,
      static_cast<int>(jwtype),
      penetrating_sw,
      sign_zero_positive);
  return {hbl, kbl, bfsfc, stable, casea, rib};
}

pybind11::dict launch_config(int device, int nz, const std::string& dtype) {
  pybind11::dict result = locomgcm::cuda_launch::to_python(
      kpp_bldepth_cuda_launch_config(device, nz, parse_scalar_type(dtype)));
  result["dtype"] = dtype;
  return result;
}

void kpp_bldepth_backward_launch(const std::vector<torch::Tensor>&,
    const torch::Tensor&,const std::vector<torch::Tensor>&,
    std::vector<torch::Tensor>&,const std::vector<double>&,bool,int,bool,bool);

std::vector<c10::optional<torch::Tensor>> kpp_bldepth_bwd(
    const std::vector<torch::Tensor>& values,const torch::Tensor& rib,
    const std::vector<c10::optional<torch::Tensor>>& seeds,
    const std::vector<double>& parameters,bool limit,int64_t jwtype,
    bool penetrating_sw,bool sign_zero_positive,const std::vector<bool>& needs) {
  TORCH_CHECK(values.size()==12 && seeds.size()==3 && parameters.size()==8
      && needs.size()==7,"invalid BLDEPTH backward argument counts");
  const auto& ref=values[1];
  CHECK_CUDA(ref);CHECK_FLOATING(ref);CHECK_DIM(ref,3);
  TORCH_CHECK(ref.size(0)==90 && ref.size(1)>0 && ref.size(2)>0,
      "BLDEPTH backward requires nonempty 90-level columns");
  TORCH_CHECK(jwtype>=1 && jwtype<=5,"invalid BLDEPTH jwtype");
  for(int i=0;i<12;++i) {
    const auto& v=values[i];CHECK_CUDA(v);check_same_device(v,ref,"input");
    check_nonnegative_strides(v,"input");
    TORCH_CHECK(v.scalar_type()==(i==0?at::kLong:ref.scalar_type()),"input dtype");
    const int rank=i>=10?2:(i>=8?1:(i>=1 && i<=3?3:2));
    TORCH_CHECK(v.dim()==rank,"input rank");
    if(i>=1 && i<=3) {
      TORCH_CHECK(v.sizes()==ref.sizes(),"volume shape");
    } else if(i<8) {
      check_horizontal_shape(v,ref.size(1),ref.size(2),"input");
    } else if(i<10) {
      TORCH_CHECK(v.size(0)==92,"grid shape");
    } else {
      TORCH_CHECK(v.size(0)==892 && v.size(1)==482,"table shape");
    }
  }
  TORCH_CHECK(rib.is_cuda() && rib.device()==ref.device() &&
      rib.scalar_type()==ref.scalar_type() && rib.sizes()==ref.sizes(),"invalid Rib");
  check_nonnegative_strides(rib,"Rib");
  std::vector<torch::Tensor> seed_values(3),outputs(7);
  for(int i=0;i<3;++i) if(seeds[i].has_value()) {
    const auto& g=seeds[i].value();
    TORCH_CHECK(g.is_cuda() && g.device()==ref.device() &&
        g.scalar_type()==ref.scalar_type() &&
        g.sizes()==(i==2?ref.sizes():values[4].sizes()),"invalid BLDEPTH cotangent");
    check_nonnegative_strides(g,"cotangent");seed_values[i]=g;
  }
  std::vector<c10::optional<torch::Tensor>> result(7);
  for(int i=0;i<7;++i) if(needs[i]) {
    outputs[i]=torch::empty(values[i+1].sizes(),ref.options());result[i]=outputs[i];
  }
  kpp_bldepth_backward_launch(values,rib,seed_values,outputs,parameters,limit,jwtype,
      penetrating_sw,sign_zero_positive);
  return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("kpp_bldepth_bwd", &kpp_bldepth_bwd, "BLDEPTH fixed-parameter native VJP");
  module.def(
      "launch_config",
      &launch_config,
      "Cached KPP bldepth CUDA launch configuration",
      pybind11::arg("device"),
      pybind11::arg("nz"),
      pybind11::arg("dtype") = "float64");
  module.def(
      "kpp_bldepth_fwd",
      &kpp_bldepth_fwd,
      "Fused KPP boundary-layer depth calculation (CUDA)");
}
