#include "./CoordConv.h"


//==================================================================================================
/*
CoordConv
https ://arxiv.org/abs/1807.03247

code taken from and slightly modified :
https://github.com/walsvid/CoordConv

Some another code for conv2d but with only one dimension
(yy only, xx is omited)
https://github.com/jiupinjia/SkyAR/blob/main/networks.py
*/
//==================================================================================================

AddCoordsImpl::AddCoordsImpl(int dimension, bool with_r)
{
    this->dimension = dimension;
    this->with_r = with_r;
}

torch::Tensor AddCoordsImpl::forwardRank1(torch::Tensor input_tensor)
{
    auto sizes = input_tensor.sizes();
    int64_t batch_size = sizes[0];
    int64_t channels = sizes[1];
    int64_t dim_x = sizes[2];

    auto xx_range = torch::arange(dim_x, torch::dtype(torch::kInt32));
    auto xx_channel = xx_range.unsqueeze(0).unsqueeze(0).to(torch::kFloat32);
    xx_channel = xx_channel / (dim_x - 1);
    xx_channel = xx_channel * 2 - 1;
    xx_channel = xx_channel.repeat({ batch_size, 1, 1 }).to(input_tensor.device());

    auto out = torch::cat({ input_tensor, xx_channel }, 1);

    if (with_r)
    {
        auto rr = torch::sqrt(torch::pow(xx_channel - 0.5, 2));
        out = torch::cat({ out, rr }, 1);
    }

    return out;
}

torch::Tensor AddCoordsImpl::forwardRank2(torch::Tensor input_tensor)
{
    auto sizes = input_tensor.sizes();
    int64_t batch_size = sizes[0];
    int64_t channels = sizes[1];
    int64_t dim_y = sizes[2];
    int64_t dim_x = sizes[3];

    auto xx_ones = torch::ones({ 1, 1, 1, dim_x }, torch::dtype(torch::kInt32));
    auto yy_ones = torch::ones({ 1, 1, 1, dim_y }, torch::dtype(torch::kInt32));

    auto xx_range = torch::arange(dim_y, torch::dtype(torch::kInt32)).unsqueeze(0).unsqueeze(0).unsqueeze(-1);
    auto yy_range = torch::arange(dim_x, torch::dtype(torch::kInt32)).unsqueeze(0).unsqueeze(0).unsqueeze(-1);

    auto xx_channel = torch::matmul(xx_range, xx_ones);
    auto yy_channel = torch::matmul(yy_range, yy_ones).permute({ 0, 1, 3, 2 });

    xx_channel = xx_channel.to(torch::kFloat32) / (dim_y - 1);
    yy_channel = yy_channel.to(torch::kFloat32) / (dim_x - 1);

    xx_channel = xx_channel * 2 - 1;
    yy_channel = yy_channel * 2 - 1;

    xx_channel = xx_channel.repeat({ batch_size, 1, 1, 1 }).to(input_tensor.device());
    yy_channel = yy_channel.repeat({ batch_size, 1, 1, 1 }).to(input_tensor.device());

    auto out = torch::cat({ input_tensor, xx_channel, yy_channel }, 1);

    if (with_r)
    {
        auto rr = torch::sqrt(torch::pow(xx_channel - 0.5, 2) + torch::pow(yy_channel - 0.5, 2));
        out = torch::cat({ out, rr }, 1);
    }

    return out;
}

torch::Tensor AddCoordsImpl::forwardRank3(torch::Tensor input_tensor)
{
    // Stub: full 3D implementation can be added if needed
    // Placeholder implementation returning input
    // Comment: Implement 3D coordinate addition similar to Rank2 extension in future
    return input_tensor;
}

torch::Tensor AddCoordsImpl::forward(torch::Tensor input_tensor)
{
    if (dimension == 1)
    {
        return forwardRank1(input_tensor);
    }
    else if (dimension == 2)
    {
        return forwardRank2(input_tensor);
    }
    else if (dimension == 3)
    {
        return forwardRank3(input_tensor);
    }
    else
    {
        TORCH_CHECK(false, "Unsupported dimension in AddCoords");
    }
}


//==================================================================================================

CoordConv1dImpl::CoordConv1dImpl(int64_t in_channels, int64_t out_channels, int64_t kernel_size,
    int64_t stride, int64_t padding, int64_t dilation,
    int64_t groups, bool bias, bool with_r)
    : addcoords(AddCoords(1, with_r))
{
    conv = register_module("conv",
        torch::nn::Conv1d(torch::nn::Conv1dOptions(in_channels + 1 + (with_r ? 1 : 0),
            out_channels, kernel_size)
            .stride(stride)
            .padding(padding)
            .dilation(dilation)
            .groups(groups)
            .bias(bias)));
}

torch::Tensor CoordConv1dImpl::forward(torch::Tensor input_tensor)
{
    auto out = addcoords->forward(input_tensor);
    out = conv->forward(out);
    return out;
}

//==================================================================================================

CoordConv2dImpl::CoordConv2dImpl(int64_t in_channels, int64_t out_channels, int64_t kernel_size,
    int64_t stride, int64_t padding, int64_t dilation,
    int64_t groups, bool bias, bool with_r) : 
    addcoords(AddCoords(2, with_r))
{
    conv = register_module("conv",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels + 2 + (with_r ? 1 : 0),
            out_channels, kernel_size)
            .stride(stride)
            .padding(padding)
            .dilation(dilation)
            .groups(groups)
            .bias(bias)));
}

CoordConv2dImpl::CoordConv2dImpl(torch::nn::Conv2dOptions& head, bool with_r) :
    addcoords(AddCoords(2, with_r))
{
    head.in_channels(head.in_channels() + 2 + (with_r ? 1 : 0));

    conv = register_module("conv", torch::nn::Conv2d(head));
}

torch::Tensor CoordConv2dImpl::forward(torch::Tensor input_tensor)
{
    auto out = addcoords->forward(input_tensor);
    out = conv->forward(out);
    return out;
}

//==================================================================================================

CoordConv3dImpl::CoordConv3dImpl(int64_t in_channels, int64_t out_channels, int64_t kernel_size,
    int64_t stride, int64_t padding, int64_t dilation,
    int64_t groups, bool bias, bool with_r)
    : addcoords(AddCoords(3, with_r))
{
    conv = register_module("conv",
        torch::nn::Conv3d(torch::nn::Conv3dOptions(in_channels + 3 + (with_r ? 1 : 0),
            out_channels, kernel_size)
            .stride(stride)
            .padding(padding)
            .dilation(dilation)
            .groups(groups)
            .bias(bias)));
}

torch::Tensor CoordConv3dImpl::forward(torch::Tensor input_tensor)
{
    auto out = addcoords->forward(input_tensor);
    out = conv->forward(out);
    return out;
}