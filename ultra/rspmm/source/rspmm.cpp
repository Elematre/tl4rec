#include <mutex>

#include <ATen/Parallel.h>

#include "operator.cuh"
#include "rspmm.h"

namespace at {

// In PyTorch 1.4.0, parallel_for depends on some functions from at::internal in ATen/Parallel.h
// which are not explicitly included
// This is fixed in some new PyTorch release
using namespace at::internal;

void rspmm_forward_check(CheckedFrom c, const TensorArg &edge_index_arg, const TensorArg &edge_type_arg,
                         const TensorArg &edge_weight_arg, const TensorArg &edge_attr_arg, const TensorArg &input_arg) {
    checkDim(c, edge_index_arg, 2);
    checkDim(c, edge_type_arg, 1);
    checkDim(c, edge_weight_arg, 1);
    checkDim(c, input_arg, 2);
    checkSameType(c, edge_index_arg, edge_type_arg);
    checkAllSameType(c, {edge_weight_arg, edge_attr_arg, input_arg});
    checkSize(c, edge_index_arg, 0, 2);
    checkSize(c, edge_type_arg, {edge_index_arg->size(1)});
    checkSize(c, edge_weight_arg, {edge_index_arg->size(1)});
    checkSize(c, edge_attr_arg, 0, edge_index_arg->size(1));
}

void rspmm_backward_check(CheckedFrom c, const TensorArg &edge_index_arg, const TensorArg &edge_type_arg,
                          const TensorArg &edge_weight_arg, const TensorArg &edge_attr_arg, const TensorArg &input_arg,
                          const TensorArg &output_arg, const TensorArg &output_grad_arg) {
    rspmm_forward_check(c, edge_index_arg, edge_type_arg, edge_weight_arg, edge_attr_arg, input_arg);
    checkDim(c, output_arg, 2);
    checkSameSize(c, output_arg, output_grad_arg);
    checkAllSameType(c, {input_arg, output_arg, output_grad_arg});
    checkSize(c, output_arg, 1, input_arg->size(1));
}

Tensor ind2ptr(const Tensor &index, int size) {
    // scatter_add is super slow for int64, due to non-hardware atomic operations
    // use int32 instead
    Tensor num_per_index = at::zeros({size}, index.options().dtype(at::ScalarType::Int));
    num_per_index.scatter_add_(0, index, at::ones(index.sizes(), num_per_index.options()));
    num_per_index = num_per_index.toType(at::ScalarType::Long);
    Tensor pointer = num_per_index.cumsum(0) - num_per_index;
    return pointer;
}

template <class scalar_t, class NaryOp, class BinaryOp>
void rspmm_forward_out_cpu(const int64_t *row_ptr, const int64_t *col_ind,
                           const scalar_t *weight, const scalar_t *edge_attr, const scalar_t *input,
                           scalar_t *output,
                           int64_t num_row, int64_t nnz, int64_t dim, int64_t edge_attr_dim) {
    parallel_for(0, num_row, 0, [&](int64_t row_start, int64_t row_end) {
        for (int64_t row = row_start; row < row_end; row++) {
            for (int64_t d = 0; d < dim; d++)
                output[row * dim + d] = NaryOp::zero;

            int64_t ptr_start = row_ptr[row];
            int64_t ptr_end = row + 1 < num_row ? row_ptr[row + 1] : nnz;
            for (int64_t ptr = ptr_start; ptr < ptr_end; ptr++) {
                int64_t col = col_ind[ptr];
                scalar_t w = weight[ptr];
                
                const scalar_t *attr_ptr = edge_attr + ptr * edge_attr_dim;
                for (int64_t d = 0; d < dim; d++) {
                    scalar_t edge_attr_value = attr_ptr[d % edge_attr_dim]; // Map dim to edge_attr_dim
                    scalar_t x = BinaryOp::forward(edge_attr_value, input[col * dim + d]);
                    scalar_t y = w * x;
                    scalar_t &out = output[row * dim + d];
                    out = NaryOp::forward(out, y);
                }
            }
        }
    });
}

template <class scalar_t, class NaryOp, class BinaryOp>
void rspmm_backward_out_cpu(const int64_t *row_ptr, const int64_t *col_ind,
                            const scalar_t *weight, const scalar_t *edge_attr, const scalar_t *input,
                            const scalar_t *output, const scalar_t *output_grad,
                            scalar_t *weight_grad, scalar_t *edge_attr_grad,
                            scalar_t *input_grad,
                            int64_t num_row, int64_t nnz, int64_t dim, int64_t edge_attr_dim,
                            std::vector<std::mutex> &edge_attr_mutex, std::vector<std::mutex> &input_mutex) {
    parallel_for(0, num_row, 0, [&](int64_t row_start, int64_t row_end) {
        for (int64_t row = row_start; row < row_end; row++) {
            int64_t ptr_start = row_ptr[row];
            int64_t ptr_end = row + 1 < num_row ? row_ptr[row + 1] : nnz;
            for (int64_t ptr = ptr_start; ptr < ptr_end; ptr++) {
                int64_t col = col_ind[ptr];
                scalar_t w = weight[ptr];
                scalar_t w_grad = 0;
                const scalar_t *attr_ptr = edge_attr + ptr * edge_attr_dim;
                scalar_t *attr_grad_ptr = edge_attr_grad + ptr * edge_attr_dim;
                for (int64_t d = 0; d < dim; d++) {
                    scalar_t attr_value = attr_ptr[d % edge_attr_dim];
                    scalar_t in = input[col * dim + d];
                    scalar_t out = output[row * dim + d];
                    scalar_t out_grad = output_grad[row * dim + d];
                    scalar_t x = BinaryOp::forward(attr_value, in);
                    scalar_t y = w * x;
                    scalar_t dx_dattr = BinaryOp::backward_lhs(attr_value, in);
                    scalar_t dx_din = BinaryOp::backward_rhs(attr_value, in);
                    scalar_t dout_dy = NaryOp::backward(out, y);
                    scalar_t dy_dw = x;
                    scalar_t dy_dx = w;
                    w_grad += out_grad * dout_dy * dy_dw;
                    {
                        std::lock_guard<std::mutex> lock(edge_attr_mutex[ptr * edge_attr_dim + d % edge_attr_dim]);
                        attr_grad_ptr[d % edge_attr_dim] += out_grad * dout_dy * dy_dx * dx_dattr;
                    }
                    {
                        std::lock_guard<std::mutex> lock(input_mutex[col * dim + d]);
                        input_grad[col * dim + d] += out_grad * dout_dy * dy_dx * dx_din;
                    }
                }
                weight_grad[ptr] = w_grad;
            }
        }
    });
}

template <template<class> class NaryOp, template<class> class BinaryOp>
Tensor rspmm_forward_cpu(const Tensor &edge_index_, const Tensor &edge_type_, const Tensor &edge_weight_,
                         const Tensor &edge_attr_, const Tensor &input_) {
    constexpr const char *fn_name = "rspmm_forward_cpu";
    TensorArg edge_index_arg(edge_index_, "edge_index", 1), edge_type_arg(edge_type_, "edge_type", 2),
              edge_weight_arg(edge_weight_, "edge_weight", 3), edge_attr_arg(edge_attr_, "edge_attr", 4),
              input_arg(input_, "input", 5);

    rspmm_forward_check(fn_name, edge_index_arg, edge_type_arg, edge_weight_arg, edge_attr_arg, input_arg);
    checkDeviceType(fn_name, {edge_index_, edge_type_, edge_weight_, edge_attr_, input_}, kCPU);

    const Tensor edge_index = edge_index_.contiguous();
    const Tensor edge_type = edge_type_.contiguous();
    const Tensor edge_weight = edge_weight_.contiguous();
    const Tensor input = input_.contiguous();
    const Tensor edge_attr = edge_attr_.contiguous();


    int64_t nnz = edge_index.size(1);
    int64_t num_row = input.size(0);
    int64_t dim = input.size(1);
    int64_t edge_attr_dim = edge_attr.size(1);
    Tensor output = at::empty({num_row, dim}, input.options());

    Tensor row_ind = edge_index.select(0, 0);
    Tensor row_ptr = ind2ptr(row_ind, num_row);
    Tensor col_ind = edge_index.select(0, 1);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "rspmm_forward_cpu", [&] {
        rspmm_forward_out_cpu<scalar_t, NaryOp<scalar_t>, BinaryOp<scalar_t>>(
            row_ptr.data_ptr<int64_t>(),
            col_ind.data_ptr<int64_t>(),
            edge_weight.data_ptr<scalar_t>(),
            edge_attr.data_ptr<scalar_t>(), // Pass edge_attr
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            num_row, nnz, dim, edge_attr_dim
        );
    });

    return output;
}

template <template<class> class NaryOp, template<class> class BinaryOp>
std::tuple<Tensor, Tensor, Tensor> rspmm_backward_cpu(
        const Tensor &edge_index_, const Tensor &edge_type_, const Tensor &edge_weight_, const Tensor &edge_attr_,
        const Tensor &input_, const Tensor &output_, const Tensor &output_grad_) {
    constexpr const char *fn_name = "rspmm_backward_cpu";
    TensorArg edge_index_arg(edge_index_, "edge_index", 1), edge_type_arg(edge_type_, "edge_type", 2),
              edge_weight_arg(edge_weight_, "edge_weight", 3), edge_attr_arg(edge_attr_, "edge_attr", 4),
              input_arg(input_, "input", 5), output_arg(output_, "output", 6),
              output_grad_arg(output_grad_, "output_grad", 7);

    rspmm_backward_check(fn_name, edge_index_arg, edge_type_arg, edge_weight_arg, edge_attr_arg, input_arg,
                         output_arg, output_grad_arg);
    checkDeviceType(fn_name, {edge_index_, edge_type_, edge_weight_, edge_attr_, input_, output_, output_grad_}, kCPU);

    const Tensor edge_index = edge_index_.contiguous();
    const Tensor edge_type = edge_type_.contiguous();
    const Tensor edge_weight = edge_weight_.contiguous();
    const Tensor edge_attr = edge_attr_.contiguous();
    const Tensor input = input_.contiguous();
    const Tensor output = output_.contiguous();
    const Tensor output_grad = output_grad_.contiguous();

    int64_t nnz = edge_index.size(1);
    int64_t num_row = input.size(0);
    int64_t dim = input.size(1);
    int64_t edge_attr_dim = edge_attr.size(1);
    Tensor weight_grad = at::zeros_like(edge_weight);
    Tensor edge_attr_grad = at::zeros_like(edge_attr);
    Tensor input_grad = at::zeros_like(input);

    Tensor row_ind = edge_index.select(0, 0);
    Tensor row_ptr = ind2ptr(row_ind, num_row);
    Tensor col_ind = edge_index.select(0, 1);
    std::vector<std::mutex> edge_attr_mutex(edge_attr.numel());
    std::vector<std::mutex> input_mutex(input.numel());
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "rspmm_backward_cpu", [&] {
        rspmm_backward_out_cpu<scalar_t, NaryOp<scalar_t>, BinaryOp<scalar_t>>(
            row_ptr.data_ptr<int64_t>(),
            col_ind.data_ptr<int64_t>(),
            edge_weight.data_ptr<scalar_t>(),
            edge_attr.data_ptr<scalar_t>(), 
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            output_grad.data_ptr<scalar_t>(),
            weight_grad.data_ptr<scalar_t>(),
            edge_attr_grad.data_ptr<scalar_t>(),
            input_grad.data_ptr<scalar_t>(),
            num_row, nnz, dim, edge_attr_dim,
            edge_attr_mutex, input_mutex
        );
    });

    return std::make_tuple(weight_grad, edge_attr_grad, input_grad);
}

#define DECLARE_FORWARD_IMPL(ADD, MUL, NARYOP, BINARYOP) \
    Tensor rspmm_##ADD##_##MUL##_forward_cpu(                                                            \
            const Tensor &edge_index, const Tensor &edge_type, const Tensor &edge_weight,                \
            const Tensor &edge_attr, const Tensor &input) {                                               \
        return rspmm_forward_cpu<NARYOP, BINARYOP>(edge_index, edge_type, edge_weight, edge_attr, input); \
    }

#define DECLARE_BACKWARD_IMPL(ADD, MUL, NARYOP, BINARYOP) \
    std::tuple<Tensor, Tensor, Tensor> rspmm_##ADD##_##MUL##_backward_cpu(                                  \
            const Tensor &edge_index, const Tensor &edge_type, const Tensor &edge_weight,                   \
            const Tensor &edge_attr, const Tensor &input, const Tensor &output, const Tensor &output_grad) { \
        return rspmm_backward_cpu<NARYOP, BINARYOP>(edge_index, edge_type, edge_weight, edge_attr, input,    \
                                                     output, output_grad);                                  \
    }

DECLARE_FORWARD_IMPL(add, mul, NaryAdd, BinaryMul)
DECLARE_BACKWARD_IMPL(add, mul, NaryAdd, BinaryMul)

DECLARE_FORWARD_IMPL(min, mul, NaryMin, BinaryMul)
DECLARE_BACKWARD_IMPL(min, mul, NaryMin, BinaryMul)

DECLARE_FORWARD_IMPL(max, mul, NaryMax, BinaryMul)
DECLARE_BACKWARD_IMPL(max, mul, NaryMax, BinaryMul)

DECLARE_FORWARD_IMPL(add, add, NaryAdd, BinaryAdd)
DECLARE_BACKWARD_IMPL(add, add, NaryAdd, BinaryAdd)

DECLARE_FORWARD_IMPL(min, add, NaryMin, BinaryAdd)
DECLARE_BACKWARD_IMPL(min, add, NaryMin, BinaryAdd)

DECLARE_FORWARD_IMPL(max, add, NaryMax, BinaryAdd)
DECLARE_BACKWARD_IMPL(max, add, NaryMax, BinaryAdd)

} // namespace at

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rspmm_add_mul_forward_cpu", &at::rspmm_add_mul_forward_cpu);
    m.def("rspmm_add_mul_backward_cpu", &at::rspmm_add_mul_backward_cpu);
    m.def("rspmm_min_mul_forward_cpu", &at::rspmm_min_mul_forward_cpu);
    m.def("rspmm_min_mul_backward_cpu", &at::rspmm_min_mul_backward_cpu);
    m.def("rspmm_max_mul_forward_cpu", &at::rspmm_max_mul_forward_cpu);
    m.def("rspmm_max_mul_backward_cpu", &at::rspmm_max_mul_backward_cpu);
    m.def("rspmm_add_add_forward_cpu", &at::rspmm_add_add_forward_cpu);
    m.def("rspmm_add_add_backward_cpu", &at::rspmm_add_add_backward_cpu);
    m.def("rspmm_min_add_forward_cpu", &at::rspmm_min_add_forward_cpu);
    m.def("rspmm_min_add_backward_cpu", &at::rspmm_min_add_backward_cpu);
    m.def("rspmm_max_add_forward_cpu", &at::rspmm_max_add_forward_cpu);
    m.def("rspmm_max_add_backward_cpu", &at::rspmm_max_add_backward_cpu);
#ifdef CUDA_OP
    m.def("rspmm_add_mul_forward_cuda", &at::rspmm_add_mul_forward_cuda);
    m.def("rspmm_add_mul_backward_cuda", &at::rspmm_add_mul_backward_cuda);
    m.def("rspmm_min_mul_forward_cuda", &at::rspmm_min_mul_forward_cuda);
    m.def("rspmm_min_mul_backward_cuda", &at::rspmm_min_mul_backward_cuda);
    m.def("rspmm_max_mul_forward_cuda", &at::rspmm_max_mul_forward_cuda);
    m.def("rspmm_max_mul_backward_cuda", &at::rspmm_max_mul_backward_cuda);
    m.def("rspmm_add_add_forward_cuda", &at::rspmm_add_add_forward_cuda);
    m.def("rspmm_add_add_backward_cuda", &at::rspmm_add_add_backward_cuda);
    m.def("rspmm_min_add_forward_cuda", &at::rspmm_min_add_forward_cuda);
    m.def("rspmm_min_add_backward_cuda", &at::rspmm_min_add_backward_cuda);
    m.def("rspmm_max_add_forward_cuda", &at::rspmm_max_add_forward_cuda);
    m.def("rspmm_max_add_backward_cuda", &at::rspmm_max_add_backward_cuda);
#endif
}