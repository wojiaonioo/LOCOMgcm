#include <torch/extension.h>

#include <vector>

torch::Tensor seaice_lsr_factor(torch::Tensor a, torch::Tensor b, torch::Tensor c, bool transpose);
std::vector<torch::Tensor> seaice_lsr_sweep(torch::Tensor coef_u, torch::Tensor factors_u,
                                            torch::Tensor old_u, torch::Tensor relax_u,
                                            torch::Tensor coef_v, torch::Tensor factors_v,
                                            torch::Tensor old_v, torch::Tensor relax_v,
                                            bool swept_u, bool swept_v);
torch::Tensor seaice_ordered_sum(torch::Tensor block);
torch::Tensor seaice_ordered_sum_pair(torch::Tensor first, torch::Tensor second);
void seaice_lsr_sweep_state(torch::Tensor coef_u, torch::Tensor factors_u, torch::Tensor old_u,
                            torch::Tensor coef_v, torch::Tensor factors_v, torch::Tensor old_v,
                            torch::Tensor reals, torch::Tensor state, torch::Tensor out_u, torch::Tensor out_v);
void seaice_lsr_advance(torch::Tensor state, torch::Tensor reals, torch::Tensor log_ints,
                        torch::Tensor log_reals, torch::Tensor pre_u, torch::Tensor pre_v,
                        torch::Tensor u, torch::Tensor v, torch::Tensor exchange,
                        torch::Tensor mask_u, torch::Tensor mask_v,
                        c10::optional<torch::Tensor> flex_u, c10::optional<torch::Tensor> flex_v,
                        int64_t position, bool check, bool flex);

namespace {

void check_field(const torch::Tensor& x, const char* name, int dim) {
  TORCH_CHECK(x.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(x.scalar_type() == at::kDouble, name, " must be float64");
  TORCH_CHECK(x.dim() == dim, name, " must have ", dim, " dimensions");
}

void check_contiguous(const torch::Tensor& x, const char* name, int dim) {
  check_field(x, name, dim);
  TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
}

torch::Tensor factor(torch::Tensor a, torch::Tensor b, torch::Tensor c, bool transpose) {
  check_contiguous(a, "a", 4);
  check_contiguous(b, "b", 4);
  check_contiguous(c, "c", 4);
  TORCH_CHECK(a.sizes() == b.sizes() && a.sizes() == c.sizes(), "a, b, c must share a shape");
  TORCH_CHECK(a.size(-1) >= 7 && a.size(-2) >= 7, "LSR lines need at least three points");
  return seaice_lsr_factor(a, b, c, transpose);
}

std::vector<torch::Tensor> sweep(torch::Tensor coef_u, torch::Tensor factors_u, torch::Tensor old_u,
                                 torch::Tensor relax_u, torch::Tensor coef_v, torch::Tensor factors_v,
                                 torch::Tensor old_v, torch::Tensor relax_v, bool swept_u, bool swept_v) {
  check_contiguous(old_u, "old_u", 4);
  check_contiguous(old_v, "old_v", 4);
  check_contiguous(coef_u, "coef_u", 5);
  check_contiguous(coef_v, "coef_v", 5);
  check_contiguous(factors_u, "factors_u", 5);
  check_contiguous(factors_v, "factors_v", 5);
  check_field(relax_u, "relax_u", 0);
  check_field(relax_v, "relax_v", 0);
  TORCH_CHECK(old_u.sizes() == old_v.sizes(), "U and V storage must agree");
  TORCH_CHECK(coef_u.size(0) == 7 && coef_v.size(0) == 7, "coefficients stack seven fields");
  const long rows = old_u.size(-2), cols = old_u.size(-1);
  TORCH_CHECK(factors_u.size(-2) == rows - 4 && factors_u.size(-1) == cols - 4, "U factors shape");
  TORCH_CHECK(factors_v.size(-2) == cols - 4 && factors_v.size(-1) == rows - 4, "V factors shape");
  return seaice_lsr_sweep(coef_u, factors_u, old_u, relax_u, coef_v, factors_v, old_v, relax_v,
                          swept_u, swept_v);
}

torch::Tensor ordered_sum(torch::Tensor block) {
  check_field(block, "block", 4);
  for (int d = 0; d < 4; ++d) TORCH_CHECK(block.stride(d) >= 0, "block strides must be non-negative");
  return seaice_ordered_sum(block);
}

torch::Tensor ordered_sum_pair(torch::Tensor first, torch::Tensor second) {
  check_field(first, "first", 4);
  check_field(second, "second", 4);
  TORCH_CHECK(first.device() == second.device(), "both blocks must share a device");
  TORCH_CHECK(first.sizes() == second.sizes() && first.strides() == second.strides(),
              "both blocks must share a shape and strides");
  for (int d = 0; d < 4; ++d) TORCH_CHECK(first.stride(d) >= 0, "block strides must be non-negative");
  return seaice_ordered_sum_pair(first, second);
}

void check_state(const torch::Tensor& state, const torch::Tensor& reals) {
  TORCH_CHECK(state.is_cuda() && state.scalar_type() == at::kInt && state.dim() == 1 && state.is_contiguous(),
              "state must be a contiguous CUDA int32 vector");
  TORCH_CHECK(reals.is_cuda() && reals.scalar_type() == at::kDouble && reals.dim() == 1 && reals.is_contiguous(),
              "reals must be a contiguous CUDA float64 vector");
}

void sweep_state(torch::Tensor coef_u, torch::Tensor factors_u, torch::Tensor old_u,
                 torch::Tensor coef_v, torch::Tensor factors_v, torch::Tensor old_v,
                 torch::Tensor reals, torch::Tensor state, torch::Tensor out_u, torch::Tensor out_v) {
  for (auto* x : {&old_u, &old_v, &out_u, &out_v}) check_contiguous(*x, "velocity", 4);
  check_contiguous(coef_u, "coef_u", 5);
  check_contiguous(coef_v, "coef_v", 5);
  check_contiguous(factors_u, "factors_u", 5);
  check_contiguous(factors_v, "factors_v", 5);
  check_state(state, reals);
  TORCH_CHECK(old_u.sizes() == old_v.sizes() && out_u.sizes() == old_u.sizes() && out_v.sizes() == old_u.sizes(),
              "U and V storage must agree");
  seaice_lsr_sweep_state(coef_u, factors_u, old_u, coef_v, factors_v, old_v, reals, state, out_u, out_v);
}

void advance(torch::Tensor state, torch::Tensor reals, torch::Tensor log_ints, torch::Tensor log_reals,
             torch::Tensor pre_u, torch::Tensor pre_v, torch::Tensor u, torch::Tensor v,
             torch::Tensor exchange, torch::Tensor mask_u, torch::Tensor mask_v,
             c10::optional<torch::Tensor> flex_u, c10::optional<torch::Tensor> flex_v,
             int64_t position, bool check, bool flex) {
  check_state(state, reals);
  for (auto* x : {&pre_u, &pre_v, &u, &v, &mask_u, &mask_v}) check_contiguous(*x, "field", 4);
  TORCH_CHECK(exchange.is_cuda() && exchange.scalar_type() == at::kInt && exchange.numel() == u.numel(),
              "exchange must map every native cell");
  TORCH_CHECK(log_ints.scalar_type() == at::kInt && log_ints.dim() == 2 && log_ints.is_contiguous() &&
              position < log_ints.size(0), "log row out of range");
  TORCH_CHECK(log_reals.scalar_type() == at::kDouble && log_reals.dim() == 2 && log_reals.is_contiguous() &&
              position < log_reals.size(0), "log row out of range");
  TORCH_CHECK(!(check && flex) || (flex_u.has_value() && flex_v.has_value()), "a flex check needs both means");
  seaice_lsr_advance(state, reals, log_ints, log_reals, pre_u, pre_v, u, v, exchange, mask_u, mask_v,
                     flex_u, flex_v, position, check, flex);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("factor", &factor, "LSR line factors (bet, CUU)");
  m.def("sweep", &sweep, "ordered LSR sweep of U and V");
  m.def("ordered_sum", &ordered_sum, "left fold in logical order");
  m.def("ordered_sum_pair", &ordered_sum_pair, "two left folds in one launch");
  m.def("sweep_state", &sweep_state, "ordered LSR sweep with device-resident iteration state");
  m.def("advance", &advance, "check, record row and X6 of one replayed iteration");
}
