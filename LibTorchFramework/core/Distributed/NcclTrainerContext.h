#ifndef NCCL_TRAINER_CONTEXT_H
#define NCCL_TRAINER_CONTEXT_H

//#define LIBTORCH_FRAMEWORK_HAS_NCCL

#ifdef LIBTORCH_FRAMEWORK_HAS_NCCL

struct ncclComm;
typedef struct ncclComm* ncclComm_t;

#include <torch/torch.h>

#include <cuda_runtime_api.h>
//#include <nccl.h>

#include <cstddef>
#include <vector>

class NcclTrainerContext
{
public:
    //[deviceId] - list of tensors
    using DeviceTensorList = std::vector<std::vector<torch::Tensor>>;

    explicit NcclTrainerContext(size_t deviceCount);
    ~NcclTrainerContext();

    void Broadcast(const DeviceTensorList& tensorsByDevice);
    void AllReduceGradients(const DeviceTensorList& parametersByDevice);
    bool HasGlobalNonFiniteGradients(const DeviceTensorList& parametersByDevice);
    void MarkGradientsNonFinite(const DeviceTensorList& parametersByDevice);

private:
    
    std::vector<int> deviceIndices;
    std::vector<ncclComm_t> communicators;

    std::vector<cudaStream_t> CurrentStreams() const;

    static void EnsureGradients(const DeviceTensorList& parametersByDevice);

    void ValidateTensorGroups(const DeviceTensorList& tensorsByDevice, const char* kind) const;
    static void ValidateCollectiveTensor(const torch::Tensor& tensor, int expectedDevice, const char* kind);    
};

#endif

#endif
