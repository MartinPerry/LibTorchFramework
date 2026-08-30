#include "./NcclTrainerContext.h"

#ifdef LIBTORCH_FRAMEWORK_HAS_NCCL

#include <c10/cuda/CUDAStream.h>


#include <nccl.h>
#ifdef _MSC_VER
#   pragma comment(lib, "nccl.lib")
#endif

#include <limits>
#include <numeric>

#include <Utils/Logger.h>

#ifndef ENABLE_ERROR_CHECKS
#   define ENABLE_ERROR_CHECKS
#endif

//===============================================================================================
// Error checking
//===============================================================================================

static void printErr(int c, const char* err, const char* stmt, const char* fname, int line)
{
    MY_LOG_ERROR("error: %d = %s", c, err);
    MY_LOG_ERROR("at %s:%i - for %s", fname, line, stmt);
}

#if defined(_DEBUG) || defined(DEBUG) || defined(ENABLE_ERROR_CHECKS)
#   define CUDA_CHECK(stmt) do { \
            auto c = stmt; \
            if ((cudaError_t)c != cudaSuccess) { \
                const char* err = cudaGetErrorString(c); \
                printErr(c, err, #stmt, __FILE__, __LINE__); \
            } \
        } while (0);
#else
#	define CUDA_CHECK(stmt) stmt
#endif

#if defined(_DEBUG) || defined(DEBUG) || defined(ENABLE_ERROR_CHECKS)
#   define NCCL_CHECK(stmt) do { \
            auto c = stmt; \
            if ((ncclResult_t)c != ncclSuccess) { \
                const char* err = ncclGetErrorString(c); \
                printErr(c, err, #stmt, __FILE__, __LINE__); \
            } \
        } while (0);
#else
#	define NCCL_CHECK(stmt) stmt
#endif

void NcclTrainerContext::ValidateTensorGroups(const DeviceTensorList& tensorsByDevice, const char* kind) const
{
#ifndef ENABLE_ERROR_CHECKS
    return;
#endif

    TORCH_CHECK(tensorsByDevice.size() == deviceIndices.size(),
        "Expected one ", kind, " tensor group per NCCL device");

    TORCH_CHECK(!tensorsByDevice.empty(), "NCCL tensor groups cannot be empty");

    const size_t tensorCount = tensorsByDevice[0].size();
    for (size_t device = 0; device < tensorsByDevice.size(); device++)
    {
        TORCH_CHECK(tensorsByDevice[device].size() == tensorCount,
            "All NCCL model replicas must have the same number of ", kind, " tensors");

        for (size_t tensorIndex = 0; tensorIndex < tensorCount; tensorIndex++)
        {
            const auto& tensor = tensorsByDevice[device][tensorIndex];
            ValidateCollectiveTensor(tensor, deviceIndices[device], kind);

            TORCH_CHECK(tensor.numel() == tensorsByDevice[0][tensorIndex].numel() &&
                tensor.scalar_type() == tensorsByDevice[0][tensorIndex].scalar_type(),
                "NCCL model replicas have incompatible ", kind, " tensor at index ", tensorIndex);

            TORCH_CHECK(tensor.requires_grad() == tensorsByDevice[0][tensorIndex].requires_grad(),
                "NCCL model replicas disagree on requires_grad for ", kind, " tensor at index ", tensorIndex);
        }
    }
}

void NcclTrainerContext::ValidateCollectiveTensor(const torch::Tensor& tensor, int expectedDevice,
    const char* kind)
{
#ifndef ENABLE_ERROR_CHECKS
    return;
#endif

    TORCH_CHECK(tensor.defined(), "NCCL ", kind, " tensor is undefined");
    TORCH_CHECK(tensor.is_cuda(), "NCCL ", kind, " tensor must be on CUDA");
    TORCH_CHECK(tensor.get_device() == expectedDevice, "NCCL ", kind, " tensor is on CUDA device ", tensor.get_device(), ", expected ", expectedDevice);
    TORCH_CHECK(tensor.layout() == torch::kStrided, "NCCL ", kind, " tensor must have a dense strided layout");
}

//===============================================================================================
//===============================================================================================
//===============================================================================================

ncclDataType_t GetNcclDataType(c10::ScalarType scalarType)
{
    switch (scalarType)
    {
    case torch::kUInt8:
    case torch::kBool:
        return ncclUint8;
    case torch::kInt8:
        return ncclInt8;
    case torch::kInt32:
        return ncclInt32;
    case torch::kInt64:
        return ncclInt64;
    case torch::kFloat16:
        return ncclFloat16;
    case torch::kFloat32:
        return ncclFloat32;
    case torch::kFloat64:
        return ncclFloat64;
#if (NCCL_MAJOR > 2) || ((NCCL_MAJOR == 2) && (NCCL_MINOR >= 10))
    case torch::kBFloat16:
        return ncclBfloat16;
#endif
    default:
        TORCH_CHECK(false, "NCCL does not support tensor dtype ", scalarType);
    }

    return ncclFloat32;
}


NcclTrainerContext::NcclTrainerContext(size_t deviceCount) :
    deviceIndices(deviceCount),
    communicators(deviceCount, nullptr)
{
    TORCH_CHECK(deviceCount > 0, "NCCL needs at least one model replica");

    int visibleDeviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&visibleDeviceCount));

    TORCH_CHECK(deviceCount <= static_cast<size_t>(visibleDeviceCount),
        "Requested ", deviceCount, " NCCL replicas, but only ",
        visibleDeviceCount, " CUDA devices are visible");

    std::iota(deviceIndices.begin(), deviceIndices.end(), 0);
    NCCL_CHECK(ncclCommInitAll(communicators.data(), static_cast<int>(deviceCount), deviceIndices.data()));
}

NcclTrainerContext::~NcclTrainerContext()
{
    for (size_t i = 0; i < communicators.size(); i++)
    {
        if (communicators[i] != nullptr)
        {
            cudaSetDevice(deviceIndices[i]);
            ncclCommDestroy(communicators[i]);
        }
    }
}

void NcclTrainerContext::Broadcast(const DeviceTensorList& tensorsByDevice)
{
    ValidateTensorGroups(tensorsByDevice, "model state");
    const auto streams = CurrentStreams();

    NCCL_CHECK(ncclGroupStart());
    for (size_t tensorIndex = 0; tensorIndex < tensorsByDevice[0].size(); tensorIndex++)
    {
        for (size_t device = 0; device < deviceIndices.size(); device++)
        {
            const auto& tensor = tensorsByDevice[device][tensorIndex];
            if (tensor.numel() == 0)
            {
                continue;
            }

            TORCH_CHECK(tensor.is_contiguous(), "NCCL model-state broadcast requires contiguous tensors");
            NCCL_CHECK(
                ncclBroadcast(
                    tensor.data_ptr(),
                    tensor.data_ptr(),
                    static_cast<size_t>(tensor.numel()),
                    GetNcclDataType(tensor.scalar_type()),
                    0,
                    communicators[device],
                    streams[device]));
        }
    }
    NCCL_CHECK(ncclGroupEnd());

    for (size_t device = 0; device < deviceIndices.size(); device++)
    {
        CUDA_CHECK(cudaSetDevice(deviceIndices[device]));
        CUDA_CHECK(cudaStreamSynchronize(streams[device]));
    }
}

void NcclTrainerContext::AllReduceGradients(const DeviceTensorList& parametersByDevice)
{
    EnsureGradients(parametersByDevice);
    ValidateTensorGroups(parametersByDevice, "parameter");
    const auto streams = CurrentStreams();

    struct PendingGradient
    {
        torch::Tensor original;
        torch::Tensor communication;
    };
    std::vector<std::vector<PendingGradient>> pending(deviceIndices.size());

    NCCL_CHECK(ncclGroupStart());
    for (size_t parameterIndex = 0;
        parameterIndex < parametersByDevice[0].size();
        parameterIndex++)
    {
        if (!parametersByDevice[0][parameterIndex].requires_grad())
        {
            continue;
        }

        for (size_t device = 0; device < deviceIndices.size(); device++)
        {
            auto gradient = parametersByDevice[device][parameterIndex].grad();
            ValidateCollectiveTensor(gradient, deviceIndices[device], "gradient");
            if (gradient.numel() == 0)
            {
                continue;
            }

            auto communication = gradient.is_contiguous() ? gradient : gradient.contiguous();

            pending[device].push_back({ gradient, communication });
            NCCL_CHECK(
                ncclAllReduce(
                    communication.data_ptr(),
                    communication.data_ptr(),
                    static_cast<size_t>(communication.numel()),
                    GetNcclDataType(communication.scalar_type()),
                    ncclSum,
                    communicators[device],
                    streams[device]));
        }
    }
    NCCL_CHECK(ncclGroupEnd());

    for (size_t device = 0; device < deviceIndices.size(); device++)
    {
        CUDA_CHECK(cudaSetDevice(deviceIndices[device]));
        CUDA_CHECK(cudaStreamSynchronize(streams[device]));
        for (auto& gradient : pending[device])
        {
            gradient.communication.div_(static_cast<double>(deviceIndices.size()));
            if (!gradient.original.is_contiguous())
            {
                gradient.original.copy_(gradient.communication);
            }
        }
    }
}

bool NcclTrainerContext::HasGlobalNonFiniteGradients(const DeviceTensorList& parametersByDevice)
{
    EnsureGradients(parametersByDevice);
    ValidateTensorGroups(parametersByDevice, "parameter");
    const auto streams = CurrentStreams();
    std::vector<torch::Tensor> flags;
    flags.reserve(deviceIndices.size());

    for (size_t device = 0; device < deviceIndices.size(); device++)
    {
        bool localNonFinite = false;
        for (auto& parameter : parametersByDevice[device])
        {
            if (parameter.requires_grad() && !torch::isfinite(parameter.grad()).all().item<bool>())
            {
                localNonFinite = true;
                break;
            }
        }

        flags.push_back(torch::full({}, localNonFinite ? 1 : 0,
            torch::TensorOptions()
                .dtype(torch::kInt32)
                .device(torch::kCUDA, deviceIndices[device]))
        );
    }

    NCCL_CHECK(ncclGroupStart());
    for (size_t device = 0; device < deviceIndices.size(); device++)
    {
        NCCL_CHECK(ncclAllReduce(
                flags[device].data_ptr(),
                flags[device].data_ptr(),
                1,
                ncclInt32,
                ncclMax,
                communicators[device],
                streams[device]));
    }
    NCCL_CHECK(ncclGroupEnd());

    for (size_t device = 0; device < deviceIndices.size(); device++)
    {
        CUDA_CHECK(cudaSetDevice(deviceIndices[device]));
        CUDA_CHECK(cudaStreamSynchronize(streams[device]));
    }
    return flags[0].item<int>() != 0;
}

void NcclTrainerContext::MarkGradientsNonFinite(const DeviceTensorList& parametersByDevice)
{
    EnsureGradients(parametersByDevice);
    for (auto& parameters : parametersByDevice)
    {
        bool marked = false;
        for (auto& parameter : parameters)
        {
            if (parameter.requires_grad() && (parameter.grad().numel() > 0))
            {
                parameter.mutable_grad()
                    .reshape({ -1 })
                    .select(0, 0)
                    .fill_(std::numeric_limits<float>::infinity());
                marked = true;
                break;
            }
        }
        TORCH_CHECK(marked, "Cannot mark overflow: model has no trainable parameters");
    }
}

std::vector<cudaStream_t> NcclTrainerContext::CurrentStreams() const
{
    std::vector<cudaStream_t> streams;
    streams.reserve(deviceIndices.size());
    for (const int device : deviceIndices)
    {
        CUDA_CHECK(cudaSetDevice(device));
        streams.push_back(c10::cuda::getCurrentCUDAStream(device).stream());
    }
    return streams;
}

void NcclTrainerContext::EnsureGradients(const DeviceTensorList& parametersByDevice)
{
    for (auto& parameters : parametersByDevice)
    {
        for (auto& parameter : parameters)
        {
            if (parameter.requires_grad() && !parameter.grad().defined())
            {
                parameter.mutable_grad() = torch::zeros_like(parameter);
            }
        }
    }
}




#endif
