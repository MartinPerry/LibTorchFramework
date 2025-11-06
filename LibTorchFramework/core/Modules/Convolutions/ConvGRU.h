#ifndef CONV_GRU_H
#define CONV_GRU_H

#include <optional>

#include <torch/torch.h>

//======================================================================================
// ConvGRU
// https://github.com/openclimatefix/skillful_nowcasting/blob/main/dgmr/layers/ConvGRU.py
//======================================================================================

class ConvGRUCellImpl : public torch::nn::Module
{
public:
    ConvGRUCellImpl(int input_channels,
        int output_channels,
        int kernel_size = 3
    );

    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x, torch::Tensor prev_state);

private:
    torch::nn::Conv2d read_gate_conv{ nullptr };
    torch::nn::Conv2d update_gate_conv{ nullptr };
    torch::nn::Conv2d output_conv{ nullptr };

};

TORCH_MODULE(ConvGRUCell);

//======================================================================================

class ConvGRUImpl : public torch::nn::Module
{
public:
    ConvGRUImpl(int input_channels,
        int output_channels,
        int kernel_size = 3
    );

    torch::Tensor forward(torch::Tensor x, torch::Tensor hidden_state = torch::Tensor());

private:
    ConvGRUCell cell{ nullptr };
};

TORCH_MODULE(ConvGRU);

#endif
