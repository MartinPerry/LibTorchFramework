

#include <iostream>

//#pragma comment(lib, "asmjit.lib")
//#pragma comment(lib, "fbgemm.lib")
//#pragma comment(lib, "fbjni.lib")
//#pragma comment(lib, "pytorch_jni.lib")

#ifdef _WIN32
#pragma comment(lib, "c10.lib")
#pragma comment(lib, "c10_cuda.lib")
#pragma comment(lib, "caffe2_nvrtc.lib")
#pragma comment(lib, "cpuinfo.lib")
#pragma comment(lib, "dnnl.lib")
#pragma comment(lib, "kineto.lib")
#pragma comment(lib, "pthreadpool.lib")
#pragma comment(lib, "torch.lib")
#pragma comment(lib, "torch_cpu.lib")
#pragma comment(lib, "torch_cuda.lib")
#pragma comment(lib, "XNNPACK.lib")

#ifdef _DEBUG
#   pragma comment(lib, "libprotobufd.lib")
#   pragma comment(lib, "libprotocd.lib")
#   pragma comment(lib, "Playgroundd.lib")
#else
#   pragma comment(lib, "libprotobuf.lib")
#   pragma comment(lib, "libprotoc.lib")
#   pragma comment(lib, "Playground.lib")
#endif
#endif

//=========================================================
// Core
//=========================================================

#include "./core/Structures.h"
#include "./core/Runner.h"
#include "./core/Trainer.h"
#include "./core/AbstractModel.h"

#include "./core/Metrics/PredictionEvaluators.h"
#include "./core/Metrics/MetricsDefault.h"
#include "./core/Metrics/MetricsImage.h"
#include "./core/Metrics/MetricsUploader.h"

#include "./core/Modules/LossFunctions/DiceLoss.h"
#include "./core/Modules/LossFunctions/MultiBceLoss.h"
#include "./core/Modules/LossFunctions/SSIMLoss.h"
#include "./core/Modules/LossFunctions/FocalFrequencyLoss.h"

#include "./core/Modules/Convolutions/ConvGRU.h"
#include "./core/Modules/Convolutions/CoordConv.h"

#include "./core/Modules/DownSample2d.h"
#include "./core/Modules/UpSample2d.h"
#include "./core/Modules/ResNetBlock.h"

#include "./core/Snapshot/PretrainedManager.h"
#include "./core/Snapshot/SnapshotSaver.h"
#include "./core/Snapshot/SnapshotLoader.h"

//=========================================================
// Inputs
//=========================================================

#include "./InputProcessing/DefaultDataset.h"
#include "./InputProcessing/InputLoadersWrapper.h"
#include "./InputProcessing/InputLoader.h"
#include "./InputProcessing/DataLoaderData.h"

#include "./InputProcessing/InputLoaders/SegmentationInputLoader.h"
#include "./InputProcessing/InputLoaders/VideoSequenceInputLoader.h"

//=========================================================
// ModelZoo
//=========================================================

#include "./ModelZoo/UNet/UNetModel.h"
#include "./ModelZoo/U2Net/U2NetModel.h"
#include "./ModelZoo/SimVPv2/SimVPv2Model.h"
#include "./ModelZoo/ResNet/ResNetModel.h"
#include "./ModelZoo/SDVAE/SDVAEModel.h"
#include "./ModelZoo/exPreCast/exPreCastModel.h"

//=========================================================
// Utils
//=========================================================

#include "./Utils/TorchUtils.h"
#include "./Utils/TorchImageUtils.h"
#include "./Utils/TrainingHelper.h"

//=========================================================

#include "./CustomScenarios/MrmsDatasetTraining/setup_mrms.h"
#include "./CustomScenarios/U2NetTraining/setup_u2net.h"
#include "./CustomScenarios/UNetTraining/setup_unet.h"
#include "./CustomScenarios/SDVAE/setup_sdvae.h"
#include "./CustomScenarios/LLMs/setup_llama.h"
#include "./CustomScenarios/exPreCastTraining/setup_exprecast.h"

#include "./Settings.h"

#include <Utils/Logger.h>

//https://perception-ml.com/getting-started-with-libtorch/

//https://pytorch.org/tutorials/advanced/cpp_frontend.html

//https://expoundai.wordpress.com/2020/10/13/setting-up-a-cpp-project-in-visual-studio-2019-with-libtorch-1-6/

//https://tebesu.github.io/posts/PyTorch-C++-Frontend


//https://medium.com/crim/from-pytorch-to-libtorch-tips-and-tricks-dc45b6c1b1ac

//===================================================================
//===================================================================
//===================================================================

#include "./core/Modules/Convolutions/DeformConv.h"
#include "./core/Modules/Convolutions/DeformConvImpl/torchvision/vision_deform_conv2d.h"

#include "./core/Modules/Convolutions/DeformConvImpl/tvdcn/ops/deform_conv2d.h"
#include "./core/Modules/Convolutions/DeformConvImpl/tvdcn/ops/deform_conv3d.h"

#include "./core/Modules/ChangableModule.h"


class Test : public ChangableModule<torch::nn::AnyModule>
{
public:

    Test()
    {                              
        AUTO_REGISTER_CHANGABLE_MODULE_WITH_ARGS(linearAny, torch::nn::Linear(torch::nn::LinearOptions(10, 20).bias(false)));

        
    }

private:
    
    torch::nn::AnyModule linearAny;
};

/*
#include <c10/core/CPUAllocator.h>
#ifdef USE_CUDA
#   include <c10/cuda/CUDACachingAllocator.h>
#endif


#ifdef _WIN32
#   include <Windows.h>
#   include <Psapi.h>
#endif
#include <memory>

#ifdef _WIN32
void PrintMemory(const char* label)
{
    PROCESS_MEMORY_COUNTERS_EX info{};
    GetProcessMemoryInfo(GetCurrentProcess(),
        reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&info), sizeof(info));

    //Private Bytes 
    // refer to the amount of memory that the process executable has asked for - 
    // not necessarily the amount it is actually using. 

    printf("[%s]\n", label);
    //printf("  WorkingSetSize     (current RSS) : %.3f GB\n",
    //    (float)info.WorkingSetSize / 1024 / 1024 / 1024);
    //printf("  PeakWorkingSetSize (high watermark, never drops): %.3f GB\n",
    //    (float)info.PeakWorkingSetSize / 1024 / 1024 / 1024);
    printf("  PrivateUsage       (committed pages): %.3f GB\n",
        (float)info.PrivateUsage / 1024 / 1024 / 1024);
}
#endif
*/

/*
#ifdef _WIN32
#define LIBTORCH_FRAMEWORK_HAS_NCCL
#endif

#ifdef LIBTORCH_FRAMEWORK_HAS_NCCL
#   include <nccl.h>
#   ifdef _MSC_VER
#       pragma comment(lib, "nccl.lib")
#   endif
#endif
*/

int main_x(int argc, char** argv)
{
    auto log = MyUtils::Logger::GetInstance();
    log->Enable(MyUtils::Logger::LogType::Error, MyUtils::Logger::LogOutput::StdOut);
    log->Enable(MyUtils::Logger::LogType::Warning, MyUtils::Logger::LogOutput::StdOut);
    log->Enable(MyUtils::Logger::LogType::Info, MyUtils::Logger::LogOutput::StdOut);

    /*
    {
        auto device = torch::kCPU;

        int N = 1, C_in = 3, C_out = 2;
        int H = 8, W = 8, D = 8;

        int kH = 3, kW = 3, kD = 3;
        int stride_h = 1, stride_w = 1, stride_d = 1;
        int pad_h = 1, pad_w = 1, pad_d = 1;
        int dilation_h = 1, dilation_w = 1, dilation_d = 1;
        int groups = 1, offset_groups = 1;
        bool use_mask = true;

        // Output size formula
        int out_h = (H + 2 * pad_h - dilation_h * (kH - 1) - 1) / stride_h + 1;
        int out_w = (W + 2 * pad_w - dilation_w * (kW - 1) - 1) / stride_w + 1;
        int out_d = (D + 2 * pad_d - dilation_d * (kD - 1) - 1) / stride_d + 1;

        torch::Tensor x = torch::rand({ N, C_in, H, W }, device).requires_grad_(true);
        torch::Tensor weight = torch::rand({ C_out, C_in, kH, kW }, device).requires_grad_(true);
        torch::Tensor offset = torch::rand({ N, 2 * kH * kW, out_h, out_w }, device).requires_grad_(true);
        torch::Tensor mask = torch::rand({ N, kH * kW, out_h, out_w }, device).requires_grad_(true);
        torch::Tensor bias = torch::rand({ C_out }, device).requires_grad_(true);

        offset = torch::Tensor(); // torch::zeros_like(x);
        mask = torch::Tensor();// torch::zeros_like(x);

                      
        auto out = tvdcn::ops::deform_conv2d(
            x,
            weight,
            offset,
            mask,
            //std::nullopt,
            bias,            
            //std::nullopt, 
            { 1, 1 },
            { 1, 1 },
            { 1, 1 },
            1
        );

        std::cout << "out = " << out.sizes() << "\n";
        std::cout << "requires_grad = " << out.requires_grad() << "\n";

        auto loss = out.sum();

        std::cout << "loss requires_grad = "
            << loss.requires_grad() << "\n";

        try {
            loss.backward();
        }
        catch (const c10::Error& e) {
            std::cerr << "LibTorch Exception: " << e.what() << "\n";
        }
        catch (const std::exception& e) {
            std::cerr << "Standard Exception: " << e.what() << "\n";
        }
        printf("x");
    }
    */

    /*
#ifdef LIBTORCH_FRAMEWORK_HAS_NCCL
    MY_LOG_INFO("NCCL supported");

    ncclComm_t comms[1];
    int devs[1] = { 0 };
    const ncclResult_t ncclResult = ncclCommInitAll(comms, 1, devs);
    if (ncclResult != ncclSuccess)
    {
        MY_LOG_ERROR("NCCL initialization failed: %s", ncclGetErrorString(ncclResult));
        return 1;
    }
    ncclCommDestroy(comms[0]);    
#endif
    */
    

    /*
    PrintMemory("Before");

    auto tensor = torch::zeros({ 16384, 16384, 4 });  // also  4GB float32

    PrintMemory("After Init");

    //auto* tmp0 = tensor.unsafeGetTensorImpl();

    tensor = tensor.to(torch::kCUDA);
    //tensor.reset();

    //auto* tmp = tensor.unsafeGetTensorImpl();

    PrintMemory("After Cuda");


    c10::cuda::CUDACachingAllocator::emptyCache();

    PrintMemory("After emptyCache");
    */

    /*
    std::cout << c10::Dispatcher::singleton()
        .findSchemaOrThrow("tvdcn::deform_conv3d", "")
        .hasKernelForDispatchKey(c10::DispatchKey::Autograd)
        << std::endl;
    */

    /*
    MetricsUploader::API_URL = "http://perry.cz/files/dashboard/upload.php";
    MetricsUploader::UPLOAD_TOKEN = "8796542131654564";

    MetricsUploader mup;
    mup.SetRunId("test8");
    mup.UploadMetrics({ {"test", 45}, {"loss", 78} });
    */

    CustomScenarios::exPreCastTraining::setup(argc, argv);
    return 0;
    /*
    Test rr;

    torch::nn::AnyModule conv2 = torch::nn::AnyModule(torch::nn::Conv2d(torch::nn::Conv2dOptions(5, 6, 3)));
    torch::nn::AnyModule linear2 = torch::nn::AnyModule(torch::nn::Linear(torch::nn::LinearOptions(30, 20).bias(false)));
    rr.ReplaceModule("linearAny", conv2);

    printf("x");
    */
    
    //std::cout << "Hello World!\n";
    //at::Tensor tensor = at::ones({ 3, 7, 2 }, at::kInt);
    //std::cout << tensor << std::endl;

    /*
    at::Tensor t0 = at::rand({ 3, 1, 64, 64 }, at::kFloat);
    at::Tensor t1 = at::rand({ 3, 1, 64, 64 }, at::kFloat);

    auto loss = torch::nn::functional::mse_loss(t0, t1);
    auto loss2 = loss.mean();
    auto manual_loss = (t0 - t1).pow(2).mean();

    std::cout << loss << std::endl;
    std::cout << loss2 << std::endl;
    std::cout << manual_loss << std::endl;
    */

    /*
    auto img0 = TorchImageUtils::TensorsToImage(t0);
    img0.Save("D://tt_b.png");

    at::Tensor t = at::ones({ 4, 3, 1, 64, 64 }, at::kFloat);
    auto img = TorchImageUtils::TensorsToImage(t);
    img.Save("D://tt_seq.png");
    */
    
    //torch::set_default_dtype(torch::scalarTypeToTypeMeta(torch::kFloat32));

    Settings::PrintCudaInfo();

    //auto device = torch::kCUDA;
    auto device = torch::kCPU;

    int N = 1, C_in = 3, C_out = 2;    
    int H = 8, W = 8, D = 8;
    
    int kH = 3, kW = 3, kD = 3;
    int stride_h = 1, stride_w = 1, stride_d = 1;
    int pad_h = 1, pad_w = 1, pad_d = 1;
    int dilation_h = 1, dilation_w = 1, dilation_d = 1;
    int groups = 1, offset_groups = 1;
    bool use_mask = true;

    // Output size formula
    int out_h = (H + 2 * pad_h - dilation_h * (kH - 1) - 1) / stride_h + 1;
    int out_w = (W + 2 * pad_w - dilation_w * (kW - 1) - 1) / stride_w + 1;
    int out_d = (D + 2 * pad_d - dilation_d * (kD - 1) - 1) / stride_d + 1;

    torch::Tensor input = torch::rand({ N, C_in, H, W }, device);
    torch::Tensor weight = torch::rand({ C_out, C_in, kH, kW }, device);
    torch::Tensor offset = torch::rand({ N, 2 * kH * kW, out_h, out_w }, device);
    torch::Tensor mask = torch::rand({ N, kH * kW, out_h, out_w }, device);
    torch::Tensor bias = torch::rand({ C_out }, device);

    auto out = vision::ops::deform_conv2d(
        input, weight, offset, mask, bias,
        stride_h, stride_w,
        pad_h, pad_w,
        dilation_h, dilation_w,
        groups, 
        offset_groups,
        use_mask
    );

    auto out2tv = tvdcn::ops::deform_conv2d(
        input, weight, offset, mask, bias,
        { stride_h, stride_w },
        { pad_h, pad_w },
        { dilation_h, dilation_w },
        groups
    );
        
    torch::Tensor input3 = torch::rand({ N, C_in, D, H, W }, device);
    torch::Tensor weight3 = torch::rand({ C_out, C_in, kD, kH, kW }, device);
    torch::Tensor offset3 = torch::rand({ N, 3 * kH * kW * kD, out_d, out_h, out_w }, device);
    torch::Tensor mask3 = torch::rand({ N, kH * kW * kD, out_d, out_h, out_w }, device);
    torch::Tensor bias3 = torch::rand({ C_out }, device);

    auto out3tv = tvdcn::ops::deform_conv3d(
        input3, weight3, offset3, mask3, bias3,
        { stride_h, stride_w, stride_d },
        { pad_h, pad_w, pad_d },
        { dilation_h, dilation_w, dilation_d },
        groups
    );

    
    auto df = DeformConv2d(C_in, C_out);
    auto out2 = df->forward(input);

    auto df3 = DeformConv3d(C_in, C_out);
    auto out3 = df3->forward(input3);
    

    CustomScenarios::LLMs::Llama::setup(argc, argv);

    //CustomScenarios::SDVAETraining::setup(argc, argv);
    //CustomScenarios::MrmsTraining::setup(argc, argv);
    //CustomScenarios::UNetTraining::setup(argc, argv);
    return 0;

    //torch::nn::MSELoss loss;
    //torch::nn::ModuleHolder mm = loss;
    //torch::nn::Module mod = torch::nn::MSELoss();
    
    static std::shared_ptr<PredictionEvaluator> predEval = std::make_shared<PredictionEvaluatorSigmoid>();

    BceDiceLoss bceLoss;
    MultiBceLoss multiLoss;

    
    Settings sets;
    //-----
    //model debug
    sets.numWorkers = 4;
    sets.device = torch::kCUDA;
    sets.perf.enableAutoCast = false;
    //-----

    //sets.numWorkers = 4;
    //sets.device = torch::kCUDA; //torch::kCUDA;    
    //sets.perf.enableAutoCast = true;
    sets.batchSize = 3;
    sets.metricsInitFn = [predEval = predEval]() -> auto {
        auto metr = std::make_shared<MetricsImage>(MetricsImage::MetricsType::SEGMENTATION);
        metr->SetPredictionEvaluator(predEval);
        return metr;
    };
    sets.lossFn = [&](const auto& output, const auto& targets) {
        //return torch::binary_cross_entropy(output[0], targets);
        //return torch::binary_cross_entropy_with_logits(output[0], targets);
        //return bceLoss(output[0], targets);
        return multiLoss(output, targets);
     };

    //if crashes with openMp - disable it
    // Assertion failed: nthr_ == nthr, file C:\actions-runner\_work\pytorch\pytorch\pytorch\third_party\ideep\mkl-dnn\src\common/dnnl_thread.hpp, line 293    
    //at::globalContext().setUserEnabledMkldnn(false);
   
    ImageSize imSize(3, 256, 256);
    ImageSize outSize(1, imSize.width, imSize.height);

    InputLoaderSettings loaderSets;
    //loaderSets.subsetSize = 200;

    auto ilw = std::make_shared<InputLoadersWrapper>(imSize);
    //ilw->InitLoaders<SegmentationInputLoader, std::string>({{ RunMode::TRAIN, loaderSets }}, "D:\\Datasets\\Skyfinder");
    ilw->InitLoaders<VideoSequenceInputLoader, std::string>({ { RunMode::TRAIN, loaderSets } }, "D:\\Datasets\\mrms_lz4", 4, 8);
    
    ilw->GetLoader(RunMode::TRAIN)->Load();

    //auto trainLoader = ilw->GetLoader(RunMode::TRAIN);
    
    //auto m = std::make_shared<ModelZoo::unet::UNetModel>(imSize, outSize);
    //auto m = std::make_shared<ModelZoo::u2net::U2NetModel>(imSize.channels, 1);
    auto m = std::make_shared<ModelZoo::SimVPv2::SimVPv2Model>(4, 8, imSize);
        
    m->CreateOptimizer<torch::optim::Adam>(torch::optim::AdamOptions(0.0001));

    sets.pretrainedManager = std::make_shared<PretrainedManager>("D://CppTorchModels");
    sets.pretrainedManager->EnableTrainingSnapshot(true);
    sets.pretrainedManager->EnableSaving(true);
    sets.pretrainedManager->EnableLoading(false);
    
    // 
    //SnapshotSaver saver(m.get());
    //saver.Save(sets.pretrainedManager);

    //SnapshotLoader loader(m.get());
    //loader.Load(sets.pretrainedManager);

    //auto trainer = Trainer(sets, m);
    //trainer.Run(trainLoader);

    TrainingHelper th(sets, m);
    th.Run(ilw);

    /*
    for (auto& batch : *trainDataloader)
    {                
        auto bData = batch.data()->input.to(device);
        auto bTarget = batch.data()->target.to(device);

        std::cout << bData << std::endl;
        std::cout << bTarget << std::endl;

        printf("xx");
    }
    */

	

	return 0;
}

/*
// repro_177520.cpp
//
// Reproducer for pytorch/pytorch#177520
//   "Memory management, memory not returned after CPU -> GPU"
//
// Reported on Windows with libtorch 2.10.0: after moving a 4 GiB CPU tensor to
// CUDA, the process' PrivateUsage stays at the post-allocation level, while
// simply resetting the same CPU tensor returns it immediately.
//
// This program runs four scenarios so the two possible explanations can be
// separated:
//   A  the exact sequence from the issue (to(CUDA) + emptyCache)
//   B  the control from the issue (allocate + reset, no CUDA involved)
//   C  keeps the CPU tensor alive, then drops it explicitly -- tells you
//      whether the CPU storage is still referenced or merely not decommitted
//   D  repeats the transfer N times -- tells you whether the retained host
//      memory plateaus (allocator/CRT heap retention) or grows without bound
//      (a real leak, which is what the reporter hits with large models)
//
// Usage:
//   repro_177520 [--gib N] [--iters N] [--scenario a|b|c|d|all] [--pause]
//
// Build: see CMakeLists.txt. Requires a CUDA-enabled libtorch and a GPU with
// enough free memory for the test tensor (4 GiB by default).

#include <torch/torch.h>
#include <torch/version.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
#endif

namespace {

    constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;

    double ToGiB(uint64_t bytes) {
        return static_cast<double>(bytes) / kGiB;
    }

#if !defined(_WIN32)
    // Linux fallback so the same source can be used to check whether the behaviour
    // is Windows-specific.
    uint64_t ReadProcStatusKb(const char* key) {
        std::FILE* f = std::fopen("/proc/self/status", "r");
        if (f == nullptr) {
            return 0;
        }
        char line[256];
        const size_t klen = std::strlen(key);
        unsigned long long kb = 0;
        while (std::fgets(line, sizeof(line), f) != nullptr) {
            if (std::strncmp(line, key, klen) == 0) {
                std::sscanf(line + klen, " %llu", &kb);
                break;
            }
        }
        std::fclose(f);
        return static_cast<uint64_t>(kb);
    }
#endif

    // Host memory as the issue measures it, plus the GPU side for context.
    void PrintMemory(const char* label) {
#if defined(_WIN32)
        PROCESS_MEMORY_COUNTERS_EX info{};
        info.cb = sizeof(info);
        GetProcessMemoryInfo(GetCurrentProcess(),
            reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&info),
            sizeof(info));
        std::printf("  [%-32s] private: %7.3f GB | workingset: %7.3f GB", label,
            ToGiB(info.PrivateUsage),
            ToGiB(static_cast<uint64_t>(info.WorkingSetSize)));
#else
        const uint64_t vm = ReadProcStatusKb("VmSize:") * 1024ull;
        const uint64_t rss = ReadProcStatusKb("VmRSS:") * 1024ull;
        std::printf("  [%-32s] vmsize:  %7.3f GB | rss:        %7.3f GB", label,
            ToGiB(vm), ToGiB(rss));
#endif

        size_t gpu_free = 0;
        size_t gpu_total = 0;
        if (cudaMemGetInfo(&gpu_free, &gpu_total) == cudaSuccess) {
            std::printf(" | gpu used: %6.3f GB", ToGiB(gpu_total - gpu_free));
        }
        std::printf("\n");
        std::fflush(stdout);
    }

    void Banner(const char* text) {
        std::printf("\n=== %s ===\n", text);
        std::fflush(stdout);
    }

    void PauseIfAsked(bool pause, const char* what) {
        if (!pause) {
            return;
        }
        std::printf("  (paused %s -- check Task Manager, then press Enter)\n", what);
        std::fflush(stdout);
        std::getchar();
    }

    // 4 GiB of float32, matching the issue's {16384, 16384, 4} exactly when the
    // default size is used; otherwise an equally sized N-GiB tensor.
    std::vector<int64_t> MakeShape(int gib) {
        if (gib == 4) {
            return { 16384, 16384, 4 };
        }
        return { gib, 1024, 1024, 256 };  // 1024*1024*256 floats == 1 GiB per slice
    }

    torch::Tensor MakeHostTensor(const std::vector<int64_t>& shape) {
        return torch::zeros(
            shape, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    }

    // ---------------------------------------------------------------------------
    // Scenario A: the sequence straight out of the issue.
    // Expected per the report: host memory never comes back down.
    // ---------------------------------------------------------------------------
    void ScenarioA(const std::vector<int64_t>& shape, bool pause) {
        Banner("A: tensor = tensor.to(kCUDA), then emptyCache  [issue repro]");

        PrintMemory("Before");

        auto tensor = MakeHostTensor(shape);
        PrintMemory("After init (CPU)");
        PauseIfAsked(pause, "after CPU allocation");

        // Rebinding drops the last reference to the CPU storage.
        tensor = tensor.to(torch::kCUDA);
        PrintMemory("After to(kCUDA)");

        torch::cuda::synchronize();
        PrintMemory("After cuda synchronize");

        c10::cuda::CUDACachingAllocator::emptyCache();
        PrintMemory("After emptyCache");
        PauseIfAsked(pause, "after emptyCache");

        // Keep the GPU tensor observably alive up to this point.
        std::printf("  (gpu tensor: device=%s numel=%lld)\n",
            tensor.device().str().c_str(),
            static_cast<long long>(tensor.numel()));

        tensor.reset();
        c10::cuda::CUDACachingAllocator::emptyCache();
        PrintMemory("After dropping GPU tensor");
    }

    // ---------------------------------------------------------------------------
    // Scenario B: the control from the issue. No CUDA anywhere.
    // Expected per the report: host memory returns to baseline.
    // ---------------------------------------------------------------------------
    void ScenarioB(const std::vector<int64_t>& shape, bool pause) {
        Banner("B: allocate + reset, no CUDA  [issue control]");

        PrintMemory("Before");

        auto tensor = MakeHostTensor(shape);
        PrintMemory("After init (CPU)");
        PauseIfAsked(pause, "after CPU allocation");

        tensor.reset();
        PrintMemory("After reset");
    }

    // ---------------------------------------------------------------------------
    // Scenario C: keep the CPU tensor alive across the copy, then drop it
    // explicitly. If host memory falls here but not in A, something about the
    // rebinding path is holding the storage; if it does not fall here either, the
    // storage is freed to the CRT heap and simply not decommitted to the OS.
    // ---------------------------------------------------------------------------
    void ScenarioC(const std::vector<int64_t>& shape, bool pause) {
        Banner("C: separate CPU and GPU handles, drop CPU explicitly");

        PrintMemory("Before");

        auto cpu_tensor = MakeHostTensor(shape);
        PrintMemory("After init (CPU)");

        auto gpu_tensor = cpu_tensor.to(torch::kCUDA);
        torch::cuda::synchronize();
        PrintMemory("After to(kCUDA), CPU alive");
        PauseIfAsked(pause, "with both tensors alive");

        std::printf("  (cpu storage use_count: %ld, data_ptr: %p)\n",
            static_cast<long>(cpu_tensor.storage().use_count()),
            cpu_tensor.data_ptr());

        cpu_tensor.reset();
        PrintMemory("After cpu_tensor.reset()");

        gpu_tensor.reset();
        c10::cuda::CUDACachingAllocator::emptyCache();
        PrintMemory("After gpu reset + emptyCache");
    }

    // ---------------------------------------------------------------------------
    // Scenario D: repeat the transfer. This is the case the reporter actually
    // cares about ("I cannot allocate more"). Watch whether host memory plateaus
    // at roughly one tensor's worth or climbs with every iteration.
    // ---------------------------------------------------------------------------
    void ScenarioD(const std::vector<int64_t>& shape, int iters, bool pause) {
        Banner("D: repeated CPU -> GPU transfers");

        PrintMemory("Before loop");

        for (int i = 0; i < iters; ++i) {
            {
                auto tensor = MakeHostTensor(shape);
                tensor = tensor.to(torch::kCUDA);
                torch::cuda::synchronize();
            }  // both the host staging tensor and the GPU tensor die here
            c10::cuda::CUDACachingAllocator::emptyCache();

            char label[64];
            std::snprintf(label, sizeof(label), "After iteration %d", i + 1);
            PrintMemory(label);
        }

        PauseIfAsked(pause, "at end of loop");
    }

    struct Options {
        int gib = 4;
        int iters = 5;
        std::string scenario = "all";
        bool pause = false;
    };

    Options ParseArgs(int argc, char** argv) {
        Options opt;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--gib" && i + 1 < argc) {
                opt.gib = std::atoi(argv[++i]);
            }
            else if (arg == "--iters" && i + 1 < argc) {
                opt.iters = std::atoi(argv[++i]);
            }
            else if (arg == "--scenario" && i + 1 < argc) {
                opt.scenario = argv[++i];
            }
            else if (arg == "--pause") {
                opt.pause = true;
            }
            else {
                std::printf(
                    "usage: %s [--gib N] [--iters N] [--scenario a|b|c|d|all] [--pause]\n",
                    argv[0]);
                //std::exit(arg == "--help" || arg == "-h" ? 0 : 2);
            }
        }
        if (opt.gib < 1) {
            opt.gib = 1;
        }
        if (opt.iters < 1) {
            opt.iters = 1;
        }
        return opt;
    }

    bool Wants(const Options& opt, const char* name) {
        return opt.scenario == "all" || opt.scenario == name;
    }

}  // namespace

int main_memorytest(int argc, char** argv) {
    const Options opt = ParseArgs(argc, argv);

    std::printf("libtorch %d.%d.%d\n", TORCH_VERSION_MAJOR, TORCH_VERSION_MINOR,
        TORCH_VERSION_PATCH);

    if (!torch::cuda::is_available()) {
        std::printf("CUDA is not available -- scenarios A, C and D need a GPU.\n");
        if (opt.scenario != "b") {
            return 1;
        }
    }

    const std::vector<int64_t> shape = MakeShape(opt.gib);
    int64_t numel = 1;
    for (const int64_t d : shape) {
        numel *= d;
    }
    std::printf("tensor: float32, shape {");
    for (size_t i = 0; i < shape.size(); ++i) {
        std::printf("%s%lld", i ? ", " : "", static_cast<long long>(shape[i]));
    }
    std::printf("} = %.3f GB\n", ToGiB(static_cast<uint64_t>(numel) * 4ull));

    // Pay the one-off CUDA context cost before measuring, so it does not show up
    // inside a scenario as if it were the tensor's memory.
    if (torch::cuda::is_available()) {
        auto warmup = torch::zeros({ 1 }, torch::TensorOptions().device(torch::kCUDA));
        torch::cuda::synchronize();
        warmup.reset();
        c10::cuda::CUDACachingAllocator::emptyCache();
    }
    PrintMemory("Baseline (ctx initialized)");

    if (Wants(opt, "a")) {
        ScenarioA(shape, opt.pause);
    }
    if (Wants(opt, "b")) {
        ScenarioB(shape, opt.pause);
    }
    if (Wants(opt, "c")) {
        ScenarioC(shape, opt.pause);
    }
    if (Wants(opt, "d")) {
        ScenarioD(shape, opt.iters, opt.pause);
    }

    Banner("Done");
    PrintMemory("Final");
    return 0;
}
*/

#include <torch/torch.h>
#include <torch/version.h>
#include <c10/core/Allocator.h>
#include <c10/core/CPUAllocator.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if !defined(_WIN32)
#error "This program targets the Windows behaviour described in the issue."
#endif

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")

namespace {

    constexpr double kMiB = 1024.0 * 1024.0;
    constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;

    uint64_t g_threshold_bytes = 256ull * 1024 * 1024;  // only log big chunks
    std::chrono::steady_clock::time_point g_t0;
    std::mutex g_log_mutex;

    double NowMs() {
        return std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - g_t0)
            .count();
    }

    void Log(const char* tag, const char* fmt, ...) {
        std::lock_guard<std::mutex> lock(g_log_mutex);
        std::printf("%9.1f ms  %-5s ", NowMs(), tag);
        va_list args;
        va_start(args, fmt);
        std::vprintf(fmt, args);
        va_end(args);
        std::printf("\n");
        std::fflush(stdout);
    }

    uint64_t CommitBytes() {
        PROCESS_MEMORY_COUNTERS_EX info{};
        info.cb = sizeof(info);
        GetProcessMemoryInfo(GetCurrentProcess(),
            reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&info),
            sizeof(info));
        return static_cast<uint64_t>(info.PrivateUsage);
    }

    uint64_t WorkingSetBytes() {
        PROCESS_MEMORY_COUNTERS_EX info{};
        info.cb = sizeof(info);
        GetProcessMemoryInfo(GetCurrentProcess(),
            reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&info),
            sizeof(info));
        return static_cast<uint64_t>(info.WorkingSetSize);
    }

    // ---------------------------------------------------------------------------
    // [cpu] PyTorch CPU allocator wrapper.
    //
    // Wraps every allocation, logs the large ones, and forwards everything to the
    // allocator that was installed before us. The deleter runs our logging first,
    // then releases the inner DataPtr, which calls the real free.
    // ---------------------------------------------------------------------------
    class TracingCPUAllocator final : public c10::Allocator {
    public:
        explicit TracingCPUAllocator(c10::Allocator* base) : base_(base) {}

        // NOTE: if your tree declares this as `allocate(size_t) const`, add const.
        c10::DataPtr allocate(size_t n) override {
            c10::DataPtr inner = base_->allocate(n);
            void* const ptr = inner.get();

            if (n >= g_threshold_bytes) {
                Log("cpu", "ALLOC  %p  %8.3f GB   (PyTorch CPU allocator)", ptr,
                    static_cast<double>(n) / kGiB);
            }

            auto* ctx = new Context{ std::move(inner), n };
            return c10::DataPtr(ptr, ctx, &TracingCPUAllocator::ReleaseContext,
                c10::Device(c10::DeviceType::CPU));
        }

        void copy_data(void* dest, const void* src, std::size_t count)
            const override {
            base_->copy_data(dest, src, count);
        }

        c10::DeleterFnPtr raw_deleter() const override {
            return &TracingCPUAllocator::ReleaseContext;
        }

    private:
        struct Context {
            c10::DataPtr inner;
            size_t size = 0;
        };

        static void ReleaseContext(void* raw_ctx) {
            auto* ctx = static_cast<Context*>(raw_ctx);
            if (ctx->size >= g_threshold_bytes) {
                Log("cpu", "FREE   %p  %8.3f GB   (PyTorch CPU allocator)",
                    ctx->inner.get(), static_cast<double>(ctx->size) / kGiB);
            }
            delete ctx;  // destroying inner DataPtr performs the real free
        }

        c10::Allocator* base_;
    };

    // ---------------------------------------------------------------------------
    // [mem] Large private committed regions, one entry per VirtualAlloc base.
    // ---------------------------------------------------------------------------
    struct Region {
        uintptr_t base = 0;
        uint64_t committed = 0;
        DWORD protect = 0;
    };

    std::string ProtectStr(DWORD protect) {
        const DWORD base = protect & 0xFF;
        switch (base) {
        case PAGE_NOACCESS:          return "NOACCESS";
        case PAGE_READONLY:          return "READONLY";
        case PAGE_READWRITE:         return "READWRITE";
        case PAGE_WRITECOPY:         return "WRITECOPY";
        case PAGE_EXECUTE:           return "EXECUTE";
        case PAGE_EXECUTE_READ:      return "EXECUTE_READ";
        case PAGE_EXECUTE_READWRITE: return "EXECUTE_RW";
        default:                     return "0x" + std::to_string(base);
        }
    }

    std::vector<Region> SnapshotRegions(uint64_t min_bytes) {
        SYSTEM_INFO si{};
        GetSystemInfo(&si);

        std::vector<Region> grouped;
        uintptr_t addr = reinterpret_cast<uintptr_t>(si.lpMinimumApplicationAddress);
        const uintptr_t end =
            reinterpret_cast<uintptr_t>(si.lpMaximumApplicationAddress);

        MEMORY_BASIC_INFORMATION mbi{};
        while (addr < end && VirtualQuery(reinterpret_cast<void*>(addr), &mbi,
            sizeof(mbi)) == sizeof(mbi)) {
            if (mbi.State == MEM_COMMIT && mbi.Type == MEM_PRIVATE) {
                const uintptr_t base = reinterpret_cast<uintptr_t>(mbi.AllocationBase);
                if (!grouped.empty() && grouped.back().base == base) {
                    grouped.back().committed += mbi.RegionSize;
                }
                else {
                    grouped.push_back({ base, static_cast<uint64_t>(mbi.RegionSize),
                                       mbi.Protect });
                }
            }
            const uintptr_t next = addr + mbi.RegionSize;
            if (next <= addr) {
                break;
            }
            addr = next;
        }

        std::vector<Region> big;
        for (const Region& r : grouped) {
            if (r.committed >= min_bytes) {
                big.push_back(r);
            }
        }
        return big;
    }

    // ---------------------------------------------------------------------------
    // Sampler thread: commit charge, large regions, device memory.
    // ---------------------------------------------------------------------------
    std::atomic<bool> g_sampler_run{ true };

    void SamplerThread(int interval_ms) {
        cudaSetDevice(0);

        uint64_t last_commit = CommitBytes();
        std::vector<Region> last_regions = SnapshotRegions(g_threshold_bytes);

        size_t last_gpu_used = 0;
        {
            size_t f = 0;
            size_t t = 0;
            if (cudaMemGetInfo(&f, &t) == cudaSuccess) {
                last_gpu_used = t - f;
            }
        }

        const uint64_t report_delta = 64ull * 1024 * 1024;

        while (g_sampler_run.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));

            const uint64_t commit = CommitBytes();
            const int64_t delta = static_cast<int64_t>(commit) -
                static_cast<int64_t>(last_commit);
            if (std::llabs(delta) >= static_cast<int64_t>(report_delta)) {
                Log("mem", "commit %+8.3f GB  -> %7.3f GB   (ws %7.3f GB)",
                    static_cast<double>(delta) / kGiB, static_cast<double>(commit) / kGiB,
                    static_cast<double>(WorkingSetBytes()) / kGiB);

                std::vector<Region> regions = SnapshotRegions(g_threshold_bytes);
                auto contains = [](const std::vector<Region>& v, uintptr_t base) {
                    for (const Region& r : v) {
                        if (r.base == base) {
                            return true;
                        }
                    }
                    return false;
                    };
                for (const Region& r : last_regions) {
                    if (!contains(regions, r.base)) {
                        Log("mem", "  region GONE  base %016llx  %7.3f GB  %s",
                            static_cast<unsigned long long>(r.base),
                            static_cast<double>(r.committed) / kGiB,
                            ProtectStr(r.protect).c_str());
                    }
                }
                for (const Region& r : regions) {
                    if (!contains(last_regions, r.base)) {
                        Log("mem", "  region NEW   base %016llx  %7.3f GB  %s",
                            static_cast<unsigned long long>(r.base),
                            static_cast<double>(r.committed) / kGiB,
                            ProtectStr(r.protect).c_str());
                    }
                }
                last_regions.swap(regions);
                last_commit = commit;
            }

            size_t gpu_free = 0;
            size_t gpu_total = 0;
            if (cudaMemGetInfo(&gpu_free, &gpu_total) == cudaSuccess) {
                const size_t used = gpu_total - gpu_free;
                const int64_t gpu_delta = static_cast<int64_t>(used) -
                    static_cast<int64_t>(last_gpu_used);
                if (std::llabs(gpu_delta) >= static_cast<int64_t>(report_delta)) {
                    Log("gpu", "device %+8.3f GB  -> %7.3f GB",
                        static_cast<double>(gpu_delta) / kGiB,
                        static_cast<double>(used) / kGiB);
                    last_gpu_used = used;
                }
            }
        }
    }

    std::vector<int64_t> MakeShape(int gib) {
        if (gib == 4) {
            return { 16384, 16384, 4 };  // the issue's exact shape
        }
        return { gib, 1024, 1024, 256 };
    }

    void Mark(const char* what) {
        Log("step", "---- %s  (commit %7.3f GB, ws %7.3f GB)", what,
            static_cast<double>(CommitBytes()) / kGiB,
            static_cast<double>(WorkingSetBytes()) / kGiB);
    }

}  // namespace

int main(int argc, char** argv) {
    int gib = 4;
    int interval_ms = 5;
    int threshold_mb = 256;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--gib" && i + 1 < argc) {
            gib = std::atoi(argv[++i]);
        }
        else if (arg == "--interval-ms" && i + 1 < argc) {
            interval_ms = std::atoi(argv[++i]);
        }
        else if (arg == "--threshold-mb" && i + 1 < argc) {
            threshold_mb = std::atoi(argv[++i]);
        }
        else {
            std::printf(
                "usage: %s [--gib N] [--interval-ms N] [--threshold-mb N]\n", argv[0]);
            return arg == "--help" || arg == "-h" ? 0 : 2;
        }
    }
    g_threshold_bytes = static_cast<uint64_t>(1 > threshold_mb ? 1 : threshold_mb) * 1024ull * 1024ull;

    std::printf("libtorch %d.%d.%d\n", TORCH_VERSION_MAJOR, TORCH_VERSION_MINOR,
        TORCH_VERSION_PATCH);
    if (!torch::cuda::is_available()) {
        std::printf("CUDA is not available.\n");
        return 1;
    }

    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
        std::printf("device: %s | driver model: %s\n", prop.name,
            prop.tccDriver ? "TCC" : "WDDM");
    }

    // Install the tracing CPU allocator before anything allocates a CPU tensor.
    static TracingCPUAllocator* tracer =
        new TracingCPUAllocator(c10::GetCPUAllocator());
    c10::SetCPUAllocator(tracer, /*priority=*/100);

    const std::vector<int64_t> shape = MakeShape(gib);
    uint64_t bytes = 4;
    for (const int64_t d : shape) {
        bytes *= static_cast<uint64_t>(d);
    }
    std::printf("tensor: float32, %.3f GB | log threshold %d MB | sampler %d ms\n",
        static_cast<double>(bytes) / kGiB, threshold_mb, interval_ms);

    // CUDA context first, so its cost is not inside the traced window.
    {
        auto warmup = torch::zeros({ 1 }, torch::TensorOptions().device(torch::kCUDA));
        torch::cuda::synchronize();
        warmup.reset();
        c10::cuda::CUDACachingAllocator::emptyCache();
    }

    std::printf("\n%9s  %-5s %s\n", "time", "tag", "event");
    std::printf("------------------------------------------------------------\n");

    g_t0 = std::chrono::steady_clock::now();
    std::thread sampler(SamplerThread, (1 > interval_ms ? 1 : interval_ms));

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    Mark("baseline");

    Mark("torch::zeros begin");
    auto tensor = torch::zeros(
        shape, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    Mark("torch::zeros end");

    void* const cpu_ptr = tensor.data_ptr();
    Log("info", "cpu tensor data_ptr = %p", cpu_ptr);

    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    Mark("tensor = tensor.to(kCUDA) begin");
    tensor = tensor.to(torch::kCUDA);
    Mark("tensor = tensor.to(kCUDA) end");

    torch::cuda::synchronize();
    Mark("synchronize done");

    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    Mark("emptyCache begin");
    c10::cuda::CUDACachingAllocator::emptyCache();
    Mark("emptyCache end");

    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    Mark("drop gpu tensor begin");
    tensor.reset();
    Mark("drop gpu tensor end");

    Mark("final emptyCache begin");
    c10::cuda::CUDACachingAllocator::emptyCache();
    Mark("final emptyCache end");

    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    Mark("done");

    g_sampler_run.store(false, std::memory_order_relaxed);
    sampler.join();
    return 0;
}