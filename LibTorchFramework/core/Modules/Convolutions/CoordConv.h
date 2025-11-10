#ifndef COORD_CONV_H
#define COORD_CONV_H

#include <torch/torch.h>


//==================================================================================================
// AddCoords class
//==================================================================================================
class AddCoordsImpl : public torch::nn::Module
{
public:
    AddCoordsImpl(int dimension, bool with_r = false);

    torch::Tensor forward(torch::Tensor input_tensor);

private:
    int dimension;
    bool with_r;

    torch::Tensor forwardRank1(torch::Tensor input_tensor);
    torch::Tensor forwardRank2(torch::Tensor input_tensor);
    torch::Tensor forwardRank3(torch::Tensor input_tensor);
};

TORCH_MODULE(AddCoords);


//==================================================================================================
// CoordConv1d
//==================================================================================================
class CoordConv1dImpl : public torch::nn::Module
{
public:
    CoordConv1dImpl(int64_t in_channels, int64_t out_channels, int64_t kernel_size,
        int64_t stride = 1, int64_t padding = 0, int64_t dilation = 1,
        int64_t groups = 1, bool bias = true, bool with_r = false);

    torch::Tensor forward(torch::Tensor input_tensor);

private:
    AddCoords addcoords;
    torch::nn::Conv1d conv{ nullptr };
};
TORCH_MODULE(CoordConv1d);

//==================================================================================================
// CoordConv2d
//==================================================================================================
class CoordConv2dImpl : public torch::nn::Module
{
public:
    CoordConv2dImpl(int64_t in_channels, int64_t out_channels, int64_t kernel_size,
        int64_t stride = 1, int64_t padding = 0, int64_t dilation = 1,
        int64_t groups = 1, bool bias = true, bool with_r = false);
    CoordConv2dImpl(torch::nn::Conv2dOptions& head, bool with_r = false);

    torch::Tensor forward(torch::Tensor input_tensor);

private:
    AddCoords addcoords;
    torch::nn::Conv2d conv{ nullptr };
};
TORCH_MODULE(CoordConv2d);

//==================================================================================================
// CoordConv3d
//==================================================================================================
class CoordConv3dImpl : public torch::nn::Module
{
public:
    CoordConv3dImpl(int64_t in_channels, int64_t out_channels, int64_t kernel_size,
        int64_t stride = 1, int64_t padding = 0, int64_t dilation = 1,
        int64_t groups = 1, bool bias = true, bool with_r = false);

    torch::Tensor forward(torch::Tensor input_tensor);

private:
    AddCoords addcoords;
    torch::nn::Conv3d conv{ nullptr };
};
TORCH_MODULE(CoordConv3d);

#endif