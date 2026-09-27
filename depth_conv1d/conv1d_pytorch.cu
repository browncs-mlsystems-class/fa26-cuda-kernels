#include <torch/extension.h>

#include "conv1d_kernel.cuh"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x) \
    TORCH_CHECK(x.dtype() == torch::kFloat32, #x "must be float32 tensor")
    
torch::Tensor conv1d(torch::Tensor u, torch::Tensor filter, torch::Tensor bias) {
    // Check properties of input
    CHECK_INPUT(u);
    CHECK_INPUT(filter);
    CHECK_INPUT(bias);

    const uint B = u.size(0);
    const uint D = u.size(1);
    const uint L = u.size(2);
    const uint K = filter.size(1);

    // Check inputted dimensions
    TORCH_CHECK(K <= MAX_K, "Received filter length of ", K, ", exceeded maximum acceptable length of ", MAX_K);
    TORCH_CHECK(filter.size(0) == D, "Expected dimension 0 of filter to be ", D, ", received ", filter.size(0));
    TORCH_CHECK(bias.size(0) == D, "Expected dimension 0 of bias to be ", D, ", received ", bias.size(0));

    torch::Tensor out = torch::empty({B, D, L}, u.options());

    launch_kernel(
        u.data_ptr<float>(), filter.data_ptr<float>(), bias.data_ptr<float>(), 
        out.data_ptr<float>(), B, L, D, K
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv1d,
      py::arg("u"), py::arg("filter"), py::arg("bias"),
      R"doc(
Applies 1D depthwise convolution with fixed dilation and stride of 1, for
short filter lengths (i.e. 1, 3, or 5).

Args:
    u (torch.Tensor): input signal of shape (batch, channels, width)
    filter (torch.Tensor): filter weights of shape (channels, kernel_size)
    bias (torch.Tensor): bias tensor of shape (channels)

Returns:
    torch.Tensor: Output tensor of shape (batch, channels, width), same as input size.

Notes:
    - Padding is automatically set to `kernel_size // 2` to preserve input size.
    - Dilation and stride are fixed to 1.
)doc");
}