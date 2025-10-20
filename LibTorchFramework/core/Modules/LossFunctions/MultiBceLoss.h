#ifndef MULTI_BCE_LOSS_H
#define MULTI_BCE_LOSS_H

#include <vector>

#include <torch/torch.h>

// ======================================================================================
// MultiBce
// ======================================================================================
struct MultiBceLossImpl : public torch::nn::Module
{
    torch::nn::BCEWithLogitsLoss bceLoss;

    explicit MultiBceLossImpl()        
    {
        bceLoss = torch::nn::BCEWithLogitsLoss();
        register_module("bceLoss", bceLoss);
    }

    torch::Tensor forward(const std::vector<torch::Tensor>& pred, const torch::Tensor& target)
    {
        auto loss = bceLoss(pred[0], target);
        for (int i = 1; i < pred.size(); i++)
        {
            loss += bceLoss(pred[i], target);
        }
        
        return loss.mean();
    }
};
TORCH_MODULE(MultiBceLoss);

#endif