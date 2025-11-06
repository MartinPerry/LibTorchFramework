#include "./ConvGRU.h"


//======================================================================================
//======================================================================================
//======================================================================================

ConvGRUCellImpl::ConvGRUCellImpl(int input_channels,
    int output_channels,
    int kernel_size)
{
    read_gate_conv = register_module(
        "read_gate_conv",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(input_channels, output_channels, kernel_size).padding(1)));

    update_gate_conv = register_module(
        "update_gate_conv",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(input_channels, output_channels, kernel_size).padding(1)));

    output_conv = register_module(
        "output_conv",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(input_channels, output_channels, kernel_size).padding(1)));

}


//======================================================================================

std::pair<torch::Tensor, torch::Tensor> ConvGRUCellImpl::forward(torch::Tensor x, torch::Tensor prev_state)
{
    if (!prev_state.defined())
    {
        // Initialize prev_state with zeros if not provided
        auto size = x.sizes();
        std::vector<int64_t> state_shape = { size[0], read_gate_conv->options.out_channels(), size[2], size[3] };
        prev_state = torch::zeros(state_shape, x.options());
    }

    torch::Tensor xh = torch::cat({ x, prev_state }, 1);

    torch::Tensor read_gate = torch::sigmoid(read_gate_conv->forward(xh));
    torch::Tensor update_gate = torch::sigmoid(update_gate_conv->forward(xh));

    torch::Tensor gated_input = torch::cat({ x, read_gate * prev_state }, 1);
    torch::Tensor c = torch::relu(output_conv->forward(gated_input));

    torch::Tensor out = update_gate * prev_state + (1.0 - update_gate) * c;
    torch::Tensor new_state = out.clone();

    return { out, new_state };
}

//======================================================================================
//======================================================================================
//======================================================================================

ConvGRUImpl::ConvGRUImpl(int input_channels,
    int output_channels,
    int kernel_size)
{
    cell = register_module("cell", ConvGRUCell(input_channels, output_channels, kernel_size));
}

//======================================================================================

torch::Tensor ConvGRUImpl::forward(torch::Tensor x, torch::Tensor hidden_state)
{
    std::vector<torch::Tensor> outputs;

    int time_steps = x.size(0);
    for (int step = 0; step < time_steps; ++step)
    {
        auto result = cell->forward(x[step], hidden_state);
        torch::Tensor output = result.first;
        hidden_state = result.second;
        outputs.push_back(output);
    }

    torch::Tensor stacked = torch::stack(outputs, 0);
    return stacked;
}
