#include "./FACL.h"

#include <cmath>

FACLImpl::FACLImpl(
    int64_t totalStep,
    double constRatio,
    double probInit,
    double probEnd,
    bool includeSigmoid)
{
    int64_t constStep = static_cast<int64_t>(totalStep * constRatio);

    mProbInit = probInit;
    mProbEnd = probEnd;
    mIncludeSigmoid = includeSigmoid;
    mStep = 0;

    int64_t numSteps = std::max<int64_t>(1, totalStep - constStep);

    mProbThresholds = torch::linspace(
        probInit,
        probEnd,
        numSteps,
        torch::kFloat);
}

double FACLImpl::GetThreshold()
{
    int64_t index;

    if (mStep < mProbThresholds.size(0))
    {
        index = mStep;
    }
    else
    {
        index = mProbThresholds.size(0) - 1;
    }

    double prob = mProbThresholds[index].item<double>();

    ++mStep;

    return 1.0 - prob;
}

torch::Tensor FACLImpl::FAL(
    const torch::Tensor& fftPred,
    const torch::Tensor& fftGt)
{
    return torch::mse_loss(fftPred.abs(), fftGt.abs());
}

torch::Tensor FACLImpl::FCL(const torch::Tensor& fftPred, const torch::Tensor& fftGt)
{
    torch::Tensor conjPred = torch::conj(fftPred);

    torch::Tensor numerator = (conjPred * fftGt).sum(); // .real();

    torch::Tensor denominator = torch::sqrt((fftGt.abs().pow(2)).sum() * (fftPred.abs().pow(2)).sum());

    return 1.0 - numerator / denominator;
}

torch::Tensor FACLImpl::forward(const torch::Tensor& predIn, const torch::Tensor& gtIn)
{
    torch::Tensor pred = predIn;
    torch::Tensor gt = gtIn;

    if (mIncludeSigmoid)
    {
        pred = torch::sigmoid(pred);
        gt = torch::sigmoid(gt);
    }

    torch::Tensor fftPred = torch::fft::fftn(pred, {}, { -2, -1 }, "ortho");
    torch::Tensor fftGt = torch::fft::fftn(gt, {}, { -2, -1 }, "ortho");

    double prob = GetThreshold();

    int64_t h = pred.size(-2);
    int64_t w = pred.size(-1);

    double weight = std::sqrt(static_cast<double>(h * w));

    torch::Tensor loss =
        prob * FAL(fftPred, fftGt) +
        (1.0 - prob) * FCL(fftPred, fftGt);

    return loss * weight;
}