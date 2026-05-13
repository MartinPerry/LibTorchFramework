#include "./PerformanceSettings.h"

#include <torch/torch.h>

//https://docs.pytorch.org/docs/2.10/notes/cuda.html

PerformanceSettings::PerformanceSettings()
{
    this->EnableCuDNN(true);

    this->EnableCuDNNFloat32(true);
    this->EnableCuBLASFloat32(true);
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

void PerformanceSettings::EnableCuDNN(bool val)
{
    this->useCudnn = val;

    if (torch::cuda::cudnn_is_available())
    {
        torch::globalContext().setBenchmarkCuDNN(useCudnn);
        torch::globalContext().setDeterministicCuDNN(!useCudnn);
    }

}

void PerformanceSettings::EnableCuDNNFloat32(bool val)
{
    if (torch::cuda::cudnn_is_available())
    {
        torch::globalContext().setAllowTF32CuDNN(val);
    }
}

void PerformanceSettings::EnableCuBLASFloat32(bool val)
{
    torch::globalContext().setAllowTF32CuBLAS(val);    
}

void PerformanceSettings::SetMatMulPrec(MatMulPrecision m)
{
    switch (m)
    {
    case MatMulPrecision::HIGHEST:
        torch::globalContext().setFloat32MatmulPrecision("highest");
        break;
    case MatMulPrecision::HIGH:
        torch::globalContext().setFloat32MatmulPrecision("high");
        break;
    case MatMulPrecision::MEDIUM:
        torch::globalContext().setFloat32MatmulPrecision("medium");
        break;
    default:
        torch::globalContext().setFloat32MatmulPrecision("highest");
        break;
    }
}