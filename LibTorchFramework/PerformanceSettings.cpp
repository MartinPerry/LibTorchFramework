#include "./PerformanceSettings.h"

#include <torch/torch.h>

PerformanceSettings::PerformanceSettings()
{
    this->EnableCudnn(true);
}

PerformanceSettings::MatMulPrecision PerformanceSettings::GetMatMulPrec() const
{
    auto v = torch::globalContext().float32MatmulPrecision();
    switch (v)
    {
    case at::Float32MatmulPrecision::HIGHEST:
        return MatMulPrecision::HIGHEST;
    case at::Float32MatmulPrecision::HIGH:
        return MatMulPrecision::HIGH;
    case at::Float32MatmulPrecision::MEDIUM:
        return MatMulPrecision::MEDIUM;
    default:
        return MatMulPrecision::HIGHEST;
    }
}

void PerformanceSettings::EnableCudnn(bool val)
{
    this->useCudnn = val;

    if (torch::cuda::cudnn_is_available())
    {
        torch::globalContext().setBenchmarkCuDNN(useCudnn);
        torch::globalContext().setDeterministicCuDNN(!useCudnn);
    }

}

void PerformanceSettings::EnableCudnnFloat32(bool val)
{
    if (torch::cuda::cudnn_is_available())
    {
        torch::globalContext().setAllowTF32CuDNN(val);
    }
}

void PerformanceSettings::SetMatMulPrec(MatMulPrecision m)
{
    switch (m)
    {
    case MatMulPrecision::HIGHEST:
        torch::globalContext().setFloat32MatmulPrecision("highest");
    case MatMulPrecision::HIGH:
        torch::globalContext().setFloat32MatmulPrecision("high");
    case MatMulPrecision::MEDIUM:
        torch::globalContext().setFloat32MatmulPrecision("medium");
    default:
        torch::globalContext().setFloat32MatmulPrecision("highest");
    }
}