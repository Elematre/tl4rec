import os
import sys

import torch.backends.openmp
from torch import autograd
from torch.utils import cpp_extension

module = sys.modules[__name__]

# This is the class we use with the current hyper parameters
class RSPMMAddMulFunction(autograd.Function):

    @staticmethod
    def forward(ctx, edge_index, edge_type, edge_weight, edge_attr, relation, input):
        #print (f"edge_index.shape: {edge_index.shape}")
        #print (f"edge_attr.shape: {edge_attr.shape}")
        #print (f"relation.shape() : {relation.shape}")
        #print (f"edge_index.shape() : {edge_index.shape}")
        #print (f"edge_type.shape() : {edge_type.shape}")
        #print (f"input.shape() : {input.shape}")
        #raise ValueError("until here") 
        print(edge_index.dtype, edge_type.dtype, edge_weight.dtype, edge_attr.dtype,relation.dtype, input.dtype)
        raise ValueError("until here") 
        node_in, node_out = edge_index
        key = node_in * (node_out.max() + 1) + node_out
        assert (key.diff() >= 0).all(), "Expect sorted `edge_index`"
        #print(input.device.type)
        if input.device.type == "cuda":
            forward = rspmm.rspmm_add_mul_forward_cuda
        else:
            forward = rspmm.rspmm_add_mul_forward_cpu
        
        output = forward(edge_index, edge_type, edge_weight, edge_attr, relation, input)
        #raise ValueError("until here") 
        ctx.save_for_backward(edge_index, edge_type, edge_weight,edge_attr, relation, input, output)
        #ctx.save_for_backward(edge_index, edge_type, edge_weight, relation, input, output)
        return output
        
    # calculates gradient
    @staticmethod
    def backward(ctx, output_grad):
        if output_grad.device.type == "cuda":
            backward = rspmm.rspmm_add_mul_backward_cuda
        else:
            backward = rspmm.rspmm_add_mul_backward_cpu
        weight_grad, edge_attr_grad, relation_grad, input_grad = backward(*ctx.saved_tensors, output_grad)
        return None, None, weight_grad, edge_attr_grad, relation_grad, input_grad


class RSPMMMinMulFunction(autograd.Function):

    @staticmethod
    def forward(ctx, edge_index, edge_type, edge_weight, relation, input):
        node_in, node_out = edge_index
        key = node_in * (node_out.max() + 1) + node_out
        assert (key.diff() >= 0).all(), "Expect sorted `edge_index`"

        if input.device.type == "cuda":
            forward = rspmm.rspmm_min_mul_forward_cuda
        else:
            forward = rspmm.rspmm_min_mul_forward_cpu
        output = forward(edge_index, edge_type, edge_weight, relation, input)
        ctx.save_for_backward(edge_index, edge_type, edge_weight, relation, input, output)
        return output

    @staticmethod
    def backward(ctx, output_grad):
        if output_grad.device.type == "cuda":
            backward = rspmm.rspmm_min_mul_backward_cuda
        else:
            backward = rspmm.rspmm_min_mul_backward_cpu
        weight_grad, relation_grad, input_grad = backward(*ctx.saved_tensors, output_grad)
        return None, None, weight_grad, relation_grad, input_grad


class RSPMMMaxMulFunction(autograd.Function):

    @staticmethod
    def forward(ctx, edge_index, edge_type, edge_weight, relation, input):
        node_in, node_out = edge_index
        key = node_in * (node_out.max() + 1) + node_out
        assert (key.diff() >= 0).all(), "Expect sorted `edge_index`"

        if input.device.type == "cuda":
            forward = rspmm.rspmm_max_mul_forward_cuda
        else:
            forward = rspmm.rspmm_max_mul_forward_cpu
        output = forward(edge_index, edge_type, edge_weight, relation, input)
        ctx.save_for_backward(edge_index, edge_type, edge_weight, relation, input, output)
        return output

    @staticmethod
    def backward(ctx, output_grad):
        if output_grad.device.type == "cuda":
            backward = rspmm.rspmm_max_mul_backward_cuda
        else:
            backward = rspmm.rspmm_max_mul_backward_cpu
        weight_grad, relation_grad, input_grad = backward(*ctx.saved_tensors, output_grad)
        return None, None, weight_grad, relation_grad, input_grad


class RSPMMAddAddFunction(autograd.Function):

    @staticmethod
    def forward(ctx, edge_index, edge_type, edge_weight, relation, input):
        print ("hey we are using RSPMMAddAddFunction ")
        node_in, node_out = edge_index
        key = node_in * (node_out.max() + 1) + node_out
        assert (key.diff() >= 0).all(), "Expect sorted `edge_index`"

        if input.device.type == "cuda":
            forward = rspmm.rspmm_add_add_forward_cuda
        else:
            forward = rspmm.rspmm_add_add_forward_cpu
        output = forward(edge_index, edge_type, edge_weight, relation, input)
        ctx.save_for_backward(edge_index, edge_type, edge_weight, relation, input, output)
        return output

    @staticmethod
    def backward(ctx, output_grad):
        if output_grad.device.type == "cuda":
            backward = rspmm.rspmm_add_add_backward_cuda
        else:
            backward = rspmm.rspmm_add_add_backward_cpu
        weight_grad, relation_grad, input_grad = backward(*ctx.saved_tensors, output_grad)
        return None, None, weight_grad, relation_grad, input_grad


class RSPMMMinAddFunction(autograd.Function):

    @staticmethod
    def forward(ctx, edge_index, edge_type, edge_weight, relation, input):
        node_in, node_out = edge_index
        key = node_in * (node_out.max() + 1) + node_out
        assert (key.diff() >= 0).all(), "Expect sorted `edge_index`"

        if input.device.type == "cuda":
            forward = rspmm.rspmm_min_add_forward_cuda
        else:
            forward = rspmm.rspmm_min_add_forward_cpu
        output = forward(edge_index, edge_type, edge_weight, relation, input)
        ctx.save_for_backward(edge_index, edge_type, edge_weight, relation, input, output)
        return output

    @staticmethod
    def backward(ctx, output_grad):
        if output_grad.device.type == "cuda":
            backward = rspmm.rspmm_min_add_backward_cuda
        else:
            backward = rspmm.rspmm_min_add_backward_cpu
        weight_grad, relation_grad, input_grad = backward(*ctx.saved_tensors, output_grad)
        return None, None, weight_grad, relation_grad, input_grad


class RSPMMMaxAddFunction(autograd.Function):

    @staticmethod
    def forward(ctx, edge_index, edge_type, edge_weight, relation, input):
        node_in, node_out = edge_index
        key = node_in * (node_out.max() + 1) + node_out
        assert (key.diff() >= 0).all(), "Expect sorted `edge_index`"

        if input.device.type == "cuda":
            forward = rspmm.rspmm_max_add_forward_cuda
        else:
            forward = rspmm.rspmm_max_add_forward_cpu
        output = forward(edge_index, edge_type, edge_weight, relation, input)
        ctx.save_for_backward(edge_index, edge_type, edge_weight, relation, input, output)
        return output

    @staticmethod
    def backward(ctx, output_grad):
        if output_grad.device.type == "cuda":
            backward = rspmm.rspmm_max_add_backward_cuda
        else:
            backward = rspmm.rspmm_max_add_backward_cpu
        weight_grad, relation_grad, input_grad = backward(*ctx.saved_tensors, output_grad)
        return None, None, weight_grad, relation_grad, input_grad


def generalized_rspmm(edge_index, edge_type, edge_weight, edge_attr, relation, input, sum="add", mul="mul"):
    name = "RSPMM%s%sFunction" % (sum.capitalize(), mul.capitalize())
    if not hasattr(module, name):
        raise ValueError("No generalized rspmm implementation found for summation `%s` and multiplication `%s`"
                         % (sum, mul))
    Function = getattr(module, name)

    # Create a unique key for every edge and sorts according to source and target node
    node_in, node_out = edge_index
    key = node_in * (node_out.max() + 1) + node_out
    order = key.argsort()

    return Function.apply(edge_index[:, order], edge_type[order], edge_weight[order], edge_attr [order, :], relation, input)


def load_extension(name, sources, extra_cflags=None, extra_cuda_cflags=None, **kwargs):
    if extra_cflags is None:
        extra_cflags = ["-Ofast"]
        # PyTorch 2.2.1+ on Apple Silicon is now compiled by default with OpenMP
        # However, installing OpenMP on macs properly and wiring it together to the compiler is tedious
        # So on macs we turn off OpenMP (as the default behavior in all torch < 2.2.1 versions)
        if torch.backends.openmp.is_available() and not sys.platform.startswith('darwin'):
            extra_cflags += ["-fopenmp", "-DAT_PARALLEL_OPENMP"]
        else:
            extra_cflags.append("-DAT_PARALLEL_NATIVE")
    if extra_cuda_cflags is None:
        if torch.cuda.is_available():
            extra_cuda_cflags = ["-O3"]
            extra_cflags.append("-DCUDA_OP")
        else:
            new_sources = []
            for source in sources:
                if not cpp_extension._is_cuda_file(source):
                    new_sources.append(source)
            sources = new_sources

    return cpp_extension.load(name, sources, extra_cflags, extra_cuda_cflags, **kwargs)


print("Load rspmm extension. This may take a while...")
path = os.path.join(os.path.dirname(__file__), "source")
rspmm = load_extension("rspmm", [os.path.join(path, "rspmm.cpp"), os.path.join(path, "rspmm.cu")])