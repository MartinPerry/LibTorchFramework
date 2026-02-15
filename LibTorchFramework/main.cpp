

#include <iostream>

//#pragma comment(lib, "asmjit.lib")
//#pragma comment(lib, "fbgemm.lib")
//#pragma comment(lib, "fbjni.lib")
//#pragma comment(lib, "pytorch_jni.lib")

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
#include "./core/Modules/Convolutions/DeformConvImpl/deform_conv2d.h"

int main()
{
    auto log = MyUtils::Logger::GetInstance();
    log->Enable(MyUtils::Logger::LogType::Error, MyUtils::Logger::LogOutput::StdOut);
    log->Enable(MyUtils::Logger::LogType::Warning, MyUtils::Logger::LogOutput::StdOut);
    log->Enable(MyUtils::Logger::LogType::Info, MyUtils::Logger::LogOutput::StdOut);

    
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
    int H = 8, W = 8;
    
    int kH = 3, kW = 3;
    int stride_h = 1, stride_w = 1;
    int pad_h = 1, pad_w = 1;
    int dilation_h = 1, dilation_w = 1;
    int groups = 1, offset_groups = 1;
    bool use_mask = true;

    // Output size formula
    int out_h = (H + 2 * pad_h - dilation_h * (kH - 1) - 1) / stride_h + 1;
    int out_w = (W + 2 * pad_w - dilation_w * (kW - 1) - 1) / stride_w + 1;

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
        groups, offset_groups,
        use_mask
    );


    auto df = DeformConv2d(C_in, C_out);
    auto out2 = df->forward(input);
    

    CustomScenarios::LLMs::Llama::setup();

    //CustomScenarios::SDVAETraining::setup();
    //CustomScenarios::MrmsTraining::setup();
    //CustomScenarios::UNetTraining::setup();
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