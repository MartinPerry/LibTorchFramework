#ifndef DICE_LOSS_H
#define DICE_LOSS_H

#include <torch/torch.h>

// ======================================================================================
// DiceLoss
// ======================================================================================
struct DiceLossImpl : public torch::nn::Module 
{
    double smooth;

    explicit DiceLossImpl(double smooth = 1.0) : 
        smooth(smooth)
    {}

    torch::Tensor forward(const torch::Tensor& pred, const torch::Tensor& target) 
    {
        // pred, target: [N, C, H, W] (same as in Python)
        auto intersection = (pred * target).sum({ 2, 3 }); // sum over H, W
        auto denom = pred.sum({ 2, 3 }) + target.sum({ 2, 3 }) + smooth;

        auto loss = 1.0 - ((2.0 * intersection + smooth) / denom);
        return loss.mean();
    }
};
TORCH_MODULE(DiceLoss);

// ======================================================================================
// BceDiceLoss
// ======================================================================================
struct BceDiceLossImpl : public torch::nn::Module 
{
    double bceWeight;
    DiceLoss diceLoss;

    explicit BceDiceLossImpl(double smooth = 1.0, double bceWeight = 0.5) :
        bceWeight(bceWeight),
        diceLoss(DiceLoss(smooth)) 
    {
        register_module("diceLoss", diceLoss);
    }

    torch::Tensor forward(const torch::Tensor& pred, const torch::Tensor& target) 
    {
        // BCE with logits (built-in functional)
        auto bce = torch::nn::functional::binary_cross_entropy_with_logits(
            pred, target,
            torch::nn::functional::BinaryCrossEntropyWithLogitsFuncOptions()
        );

        auto prob = torch::sigmoid(pred);
        auto dice = diceLoss->forward(prob, target);

        auto loss = bce * bceWeight + dice * (1.0 - bceWeight);
        return loss;
    }
};
TORCH_MODULE(BceDiceLoss);

#endif