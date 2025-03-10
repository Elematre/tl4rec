#include <ATen/cuda/CUDAContext.h>
#include <THC/THCAtomics.cuh>

#include "util.cuh"
#include "operator.cuh"
#include "rspmm.h"

namespace at {

// Memory & time efficient implementation of generalized spmm
// Much of the code is inspired by GE-SpMM
// https://github.com/hgyhungry/ge-spmm

namespace {

const int kCoarseningFactor = 2;
const int kThreadPerBlock = 256;

} // namespace anonymous

template <class scalar_t, class NaryOp, class BinaryOp>
__global__
void rspmm_forward_out_cuda(const int64_t *row_ptr, const int64_t *col_ind,
                            const scalar_t *weight, const scalar_t *edge_attr, 
                            const scalar_t *input,
                            scalar_t *output,
                            int64_t num_row, int64_t nnz, int64_t dim, int64_t edge_attr_dim) {
    // for best optimization, the following code is compiled with constant warpSize
    assert(blockDim.x == warpSize);

    extern __shared__ int64_t buffer[];
    // each fit an entire block
    int64_t *col_ind_buf = buffer;
    scalar_t *weight_buf = reinterpret_cast<scalar_t *>(buffer + blockDim.y * warpSize);
    scalar_t *edge_attr_buf = reinterpret_cast<scalar_t *>(weight_buf + blockDim.y * warpSize);
    // used to get the base address per thread
    col_ind_buf += threadIdx.y * warpSize;
    weight_buf += threadIdx.y * warpSize;
    edge_attr_buf += threadIdx.y * warpSize * edge_attr_dim;
    

    int64_t row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= num_row)
        return;
    int64_t d_start = blockIdx.y * warpSize * kCoarseningFactor + threadIdx.x;
    int64_t ptr_start = row_ptr[row];
    int64_t ptr_end = row + 1 < num_row ? row_ptr[row + 1] : nnz;
    scalar_t out[kCoarseningFactor];
#pragma unroll
    for (int64_t i = 0; i < kCoarseningFactor; i++)
        out[i] = NaryOp::zero;

    for (int64_t block_ptr = ptr_start; block_ptr < ptr_end; block_ptr += warpSize) {
        int64_t ptr = block_ptr + threadIdx.x;
        if (ptr < ptr_end) {
            col_ind_buf[threadIdx.x] = col_ind[ptr];
            weight_buf[threadIdx.x] = weight[ptr];
            for (int d = 0; d < edge_attr_dim; ++d) {
                edge_attr_buf[threadIdx.x * edge_attr_dim + d] = edge_attr[ptr * edge_attr_dim + d];
            }

        }

        __syncwarp();

        int64_t max_offset = warpSize < ptr_end - block_ptr ? warpSize : ptr_end - block_ptr;
        for (int64_t offset_ptr = 0; offset_ptr < max_offset; offset_ptr++) {
            int64_t col = col_ind_buf[offset_ptr];
            scalar_t w = weight_buf[offset_ptr];


            // weight_buf[offset_ptr] = weight[ptr]; where ptr=  block_ptr + offset_ptr;
            //--> weight_buf[offset_ptr] = weight[block_ptr + offset_ptr];
            // Pointer to the edge attributes for this edge
            // this is correct if we dont want to write edge_attr into shared memory
            //const scalar_t *attr_ptr = edge_attr + (block_ptr + offset_ptr) * edge_attr_dim;
           
            scalar_t *edge_attr_base = edge_attr_buf + offset_ptr * edge_attr_dim;

#pragma unroll
            for (int64_t i = 0; i < kCoarseningFactor; i++) {
                int64_t d = d_start + i * warpSize;
                if (d >= dim)
                    break;

                 // Use edge_attr directly in the distmult computation
                scalar_t edge_attr_value = edge_attr_base[d % edge_attr_dim];
                //scalar_t edge_attr_value = attr_ptr[d % edge_attr_dim]; // Use modulo to map dim to edge_attr features
                scalar_t x = BinaryOp::forward(edge_attr_value, input[col * dim + d]);
                scalar_t y = w * x;

                out[i] = NaryOp::forward(out[i], y);
            }
        }
        __syncwarp();
    }

#pragma unroll
    for (int64_t i = 0; i < kCoarseningFactor; i++) {
        int64_t d = d_start + i * warpSize;
        if (d >= dim)
            break;
        output[row * dim + d] = out[i];
    }
}

template <class scalar_t, class NaryOp, class BinaryOp>
__global__
void rspmm_backward_out_cuda(const int64_t *row_ptr, const int64_t *col_ind,
                             const scalar_t *weight, const scalar_t *edge_attr, const scalar_t *input,
                             const scalar_t *output, const scalar_t *output_grad,
                             scalar_t *weight_grad, scalar_t *edge_attr_grad, scalar_t *input_grad,
                             int64_t num_row, int64_t nnz, int64_t dim, int64_t edge_attr_dim) {
    // for best optimization, the following code is compiled with constant warpSize
    assert(blockDim.x == warpSize);

    extern __shared__ int64_t buffer[];
    int64_t *col_ind_buf = buffer;
    scalar_t *weight_buf = reinterpret_cast<scalar_t *>(buffer + blockDim.y * warpSize);
    scalar_t *edge_attr_buf = reinterpret_cast<scalar_t *>(weight_buf + blockDim.y * warpSize);
    col_ind_buf += threadIdx.y * warpSize;
    weight_buf += threadIdx.y * warpSize;
    edge_attr_buf += threadIdx.y * warpSize * edge_attr_dim;

    int64_t row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= num_row)
        return;
    int64_t d_start = blockIdx.y * warpSize * kCoarseningFactor + threadIdx.x;
    int64_t ptr_start = row_ptr[row];
    int64_t ptr_end = row + 1 < num_row ? row_ptr[row + 1] : nnz;

    for (int64_t block_ptr = ptr_start; block_ptr < ptr_end; block_ptr += warpSize) {
        int64_t ptr = block_ptr + threadIdx.x;
        if (ptr < ptr_end) {
            col_ind_buf[threadIdx.x] = col_ind[ptr];
            weight_buf[threadIdx.x] = weight[ptr];
            for (int d = 0; d < edge_attr_dim; ++d) {
                edge_attr_buf[threadIdx.x * edge_attr_dim + d] = edge_attr[ptr * edge_attr_dim + d];
            }
        }
        __syncwarp();

        int64_t max_offset = warpSize < ptr_end - block_ptr ? warpSize : ptr_end - block_ptr;
        for (int64_t offset_ptr = 0; offset_ptr < max_offset; offset_ptr++) {
            int64_t col = col_ind_buf[offset_ptr];
            scalar_t w = weight_buf[offset_ptr];
            scalar_t *edge_attr_base = edge_attr_buf + offset_ptr * edge_attr_dim;

            scalar_t w_grad = 0;
            scalar_t *attr_grad_ptr = edge_attr_grad + (block_ptr + offset_ptr) * edge_attr_dim;
#pragma unroll
            for (int64_t i = 0; i < kCoarseningFactor; i++) {
                int64_t d = d_start + i * warpSize;
                if (d >= dim)
                    break;

                scalar_t attr_value = edge_attr_base[d % edge_attr_dim];
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
                atomicAdd(&attr_grad_ptr[d % edge_attr_dim], out_grad * dout_dy * dy_dx * dx_dattr);
                atomicAdd(&input_grad[col * dim + d], out_grad * dout_dy * dy_dx * dx_din);
            }
            w_grad = warp_reduce(w_grad);
            if (threadIdx.x == 0)
                atomicAdd(&weight_grad[block_ptr + offset_ptr], w_grad);
        }
        __syncwarp();
    }
}

// only edge_attr & input require gradients
template <class scalar_t, class NaryOp, class BinaryOp>
__global__
void rspmm_backward_out_cuda(const int64_t *row_ptr, const int64_t *col_ind,
                             const scalar_t *weight, const scalar_t *edge_attr , const scalar_t *input,
                             const scalar_t *output, const scalar_t *output_grad, scalar_t *edge_attr_grad,
                             scalar_t *input_grad,
                             int64_t num_row, int64_t nnz, int64_t dim, int64_t edge_attr_dim) {
    // for best optimization, the following code is compiled with constant warpSize
    assert(blockDim.x == warpSize);

    extern __shared__ int64_t buffer[];
    int64_t *col_ind_buf = buffer;
    scalar_t *weight_buf = reinterpret_cast<scalar_t *>(buffer + blockDim.y * warpSize);
    scalar_t *edge_attr_buf = reinterpret_cast<scalar_t *>(weight_buf + blockDim.y * warpSize);
    col_ind_buf += threadIdx.y * warpSize;
    weight_buf += threadIdx.y * warpSize;
    edge_attr_buf += threadIdx.y * warpSize * edge_attr_dim;

    int64_t row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= num_row)
        return;
    int64_t d_start = blockIdx.y * warpSize * kCoarseningFactor + threadIdx.x;
    int64_t ptr_start = row_ptr[row];
    int64_t ptr_end = row + 1 < num_row ? row_ptr[row + 1] : nnz;

    for (int64_t block_ptr = ptr_start; block_ptr < ptr_end; block_ptr += warpSize) {
        int64_t ptr = block_ptr + threadIdx.x;
        if (ptr < ptr_end) {
            col_ind_buf[threadIdx.x] = col_ind[ptr];
            weight_buf[threadIdx.x] = weight[ptr];
            for (int d = 0; d < edge_attr_dim; ++d) {
                edge_attr_buf[threadIdx.x * edge_attr_dim + d] = edge_attr[ptr * edge_attr_dim + d];
            }
        }
        __syncwarp();

        int64_t max_offset = warpSize < ptr_end - block_ptr ? warpSize : ptr_end - block_ptr;
        for (int64_t offset_ptr = 0; offset_ptr < max_offset; offset_ptr++) {
            int64_t col = col_ind_buf[offset_ptr];
            scalar_t w = weight_buf[offset_ptr];
            scalar_t *edge_attr_base = edge_attr_buf + offset_ptr * edge_attr_dim;

            
            scalar_t *attr_grad_ptr = edge_attr_grad + (block_ptr + offset_ptr) * edge_attr_dim;
#pragma unroll
            for (int64_t i = 0; i < kCoarseningFactor; i++) {
                int64_t d = d_start + i * warpSize;
                if (d >= dim)
                    break;
                scalar_t attr_value = edge_attr_base[d % edge_attr_dim];
                scalar_t in = input[col * dim + d];
                scalar_t out = output[row * dim + d];
                scalar_t out_grad = output_grad[row * dim + d];
                scalar_t x = BinaryOp::forward(attr_value, in);
                scalar_t y = w * x;
                scalar_t dx_dattr = BinaryOp::backward_lhs(attr_value, in);
                scalar_t dx_din = BinaryOp::backward_rhs(attr_value, in);
                scalar_t dout_dy = NaryOp::backward(out, y);
                scalar_t dy_dx = w;
                atomicAdd(&attr_grad_ptr[d % edge_attr_dim], out_grad * dout_dy * dy_dx * dx_dattr);
                atomicAdd(&input_grad[col * dim + d], out_grad * dout_dy * dy_dx * dx_din);
            }
        }
        __syncwarp();
    }
}

template <template<class> class NaryOp, template<class> class BinaryOp>
Tensor rspmm_forward_cuda(const Tensor &edge_index_, const Tensor &edge_type_, const Tensor &edge_weight_,
                          const Tensor &edge_attr_, const Tensor &input_) {
    constexpr const char *fn_name = "rspmm_forward_cuda";
    TensorArg edge_index_arg(edge_index_, "edge_index", 1), edge_type_arg(edge_type_, "edge_type", 2),
              edge_weight_arg(edge_weight_, "edge_weight", 3), edge_attr_arg(edge_attr_, "edge_attr", 4),
              input_arg(input_, "input", 5);

    rspmm_forward_check(fn_name, edge_index_arg, edge_type_arg, edge_weight_arg, edge_attr_arg, input_arg);
    checkAllSameGPU(fn_name, {edge_index_arg, edge_type_arg, edge_weight_arg, edge_attr_arg, input_arg});

    const Tensor edge_index = edge_index_.contiguous();
    const Tensor edge_type = edge_type_.contiguous();
    const Tensor edge_weight = edge_weight_.contiguous();
    const Tensor edge_attr = edge_attr_.contiguous(); 
    const Tensor input = input_.contiguous();

    int64_t nnz = edge_index.size(1);
    int64_t num_row = input.size(0);
    int64_t dim = input.size(1);
    int64_t edge_attr_dim = edge_attr.size(1); 
    Tensor output = at::empty({num_row, dim}, input.options());

    Tensor row_ind = edge_index.select(0, 0);
    Tensor row_ptr = ind2ptr(row_ind, num_row);
    Tensor col_ind = edge_index.select(0, 1);

    cudaSetDevice(input.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();

    const int dim_per_block = 32; // warpSize
    const int num_dim_block = (dim + dim_per_block * kCoarseningFactor - 1) / (dim_per_block * kCoarseningFactor);
    const int row_per_block = kThreadPerBlock / dim_per_block;
    const int num_row_block = (num_row + row_per_block - 1) / row_per_block;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "rspmm_forward_cuda", [&] {
        const int memory_size = kThreadPerBlock * (sizeof(int64_t) + sizeof(scalar_t) + edge_attr_dim * sizeof(scalar_t));
        rspmm_forward_out_cuda<scalar_t, NaryOp<scalar_t>, BinaryOp<scalar_t>>
            <<<dim3(num_row_block, num_dim_block), dim3(dim_per_block, row_per_block), memory_size, stream>>>(
            row_ptr.data_ptr<int64_t>(),
            col_ind.data_ptr<int64_t>(),
            edge_weight.data_ptr<scalar_t>(),
            edge_attr.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            num_row, nnz, dim,
            edge_attr_dim
        );
    });

    return output;
}

template <template<class> class NaryOp, template<class> class BinaryOp>
std::tuple<Tensor, Tensor, Tensor> rspmm_backward_cuda(
        const Tensor &edge_index_, const Tensor &edge_type_, const Tensor &edge_weight_,const Tensor &edge_attr_,
        const Tensor &input_, const Tensor &output_, const Tensor &output_grad_) {
    constexpr const char *fn_name = "rspmm_backward_cuda";
    TensorArg edge_index_arg(edge_index_, "edge_index", 1), edge_type_arg(edge_type_, "edge_type", 2),
              edge_weight_arg(edge_weight_, "edge_weight", 3), edge_attr_arg(edge_attr_, "edge_attr", 4),
              input_arg(input_, "input", 5), output_arg(output_, "output", 6),
              output_grad_arg(output_grad_, "output_grad", 7);

    rspmm_backward_check(fn_name, edge_index_arg, edge_type_arg, edge_weight_arg, edge_attr_arg, input_arg,
                         output_arg, output_grad_arg);
    checkAllSameGPU(fn_name, {edge_index_arg, edge_type_arg, edge_weight_arg, edge_attr_arg, input_arg, output_arg,
                              output_grad_arg});

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

    cudaSetDevice(input.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();

    const int dim_per_block = 32; // warpSize
    const int num_dim_block = (dim + dim_per_block * kCoarseningFactor - 1) / (dim_per_block * kCoarseningFactor);
    const int row_per_block = kThreadPerBlock / dim_per_block;
    const int num_row_block = (num_row + row_per_block - 1) / row_per_block;

    if (edge_weight.requires_grad())
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "rspmm_backward_cuda", [&] {
            const int memory_size = kThreadPerBlock * (sizeof(int64_t) + sizeof(scalar_t) + edge_attr_dim * sizeof(scalar_t));
            rspmm_backward_out_cuda<scalar_t, NaryOp<scalar_t>, BinaryOp<scalar_t>>
                <<<dim3(num_row_block, num_dim_block), dim3(dim_per_block, row_per_block), memory_size, stream>>>(
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
                num_row, nnz, dim, edge_attr_dim
            );
        });
    else
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "rspmm_backward_cuda", [&] {
            const int memory_size = kThreadPerBlock * (sizeof(int64_t) + sizeof(scalar_t) + edge_attr_dim * sizeof(scalar_t));
            rspmm_backward_out_cuda<scalar_t, NaryOp<scalar_t>, BinaryOp<scalar_t>>
                <<<dim3(num_row_block, num_dim_block), dim3(dim_per_block, row_per_block), memory_size, stream>>>(
                row_ptr.data_ptr<int64_t>(),
                col_ind.data_ptr<int64_t>(),
                edge_weight.data_ptr<scalar_t>(),
                edge_attr.data_ptr<scalar_t>(),
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                output_grad.data_ptr<scalar_t>(),
                edge_attr_grad.data_ptr<scalar_t>(),
                input_grad.data_ptr<scalar_t>(),
                num_row, nnz, dim, edge_attr_dim
            );
        });

    return std::make_tuple(weight_grad, edge_attr_grad, input_grad);
}

#define DECLARE_FORWARD_IMPL(ADD, MUL, NARYOP, BINARYOP) \
    Tensor rspmm_##ADD##_##MUL##_forward_cuda(                                                            \
            const Tensor &edge_index, const Tensor &edge_type, const Tensor &edge_weight,                 \
            const Tensor &edge_attr, const Tensor &input) {                                                \
        return rspmm_forward_cuda<NARYOP, BINARYOP>(edge_index, edge_type, edge_weight,edge_attr, input); \
    }

#define DECLARE_BACKWARD_IMPL(ADD, MUL, NARYOP, BINARYOP) \
    std::tuple<Tensor, Tensor, Tensor> rspmm_##ADD##_##MUL##_backward_cuda(                                 \
            const Tensor &edge_index, const Tensor &edge_type, const Tensor &edge_weight, const Tensor &edge_attr,                   \
            const Tensor &input, const Tensor &output, const Tensor &output_grad) { \
        return rspmm_backward_cuda<NARYOP, BINARYOP>(edge_index, edge_type, edge_weight, edge_attr, input,   \
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