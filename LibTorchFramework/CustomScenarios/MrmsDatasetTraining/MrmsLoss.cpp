#include "./MrmsLoss.h"


using namespace CustomScenarios::MrmsTraining;

CustomSimVpLossImpl::CustomSimVpLossImpl(int chanCount)
{    
    mseLoss = torch::nn::MSELoss();

    ssimLoss = SSIMLoss(255.0f, 11, 1.5f, chanCount);
    
    ffLoss = FocalFrequencyLoss(100.0f);
}

torch::Tensor CustomSimVpLossImpl::forward(const torch::Tensor& pred, const torch::Tensor& target)
{
    auto predFlat = pred.flatten(0, 1);
    auto targetFlat = target.flatten(0, 1);

    auto featureLoss = ssimLoss->forward(predFlat, targetFlat);
    auto focalLoss = ffLoss->forward(predFlat, targetFlat);

    return featureLoss + focalLoss;
}