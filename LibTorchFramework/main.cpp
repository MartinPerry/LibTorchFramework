

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

//=========================================================
// ModelZoo
//=========================================================

#include "./ModelZoo/UNet/UNetModel.h"
#include "./ModelZoo/U2Net/U2NetModel.h"

//=========================================================
// Utils
//=========================================================

#include "./Utils/TorchUtils.h"
#include "./Utils/TorchImageUtils.h"
#include "./Utils/TrainingHelper.h"

//=========================================================

#include "./Settings.h"

#include <Utils/Logger.h>

//https://perception-ml.com/getting-started-with-libtorch/

//https://pytorch.org/tutorials/advanced/cpp_frontend.html

//https://expoundai.wordpress.com/2020/10/13/setting-up-a-cpp-project-in-visual-studio-2019-with-libtorch-1-6/

//https://tebesu.github.io/posts/PyTorch-C++-Frontend

//===================================================================
//===================================================================
//===================================================================


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
    at::Tensor t0 = at::ones({ 3, 1, 64, 64 }, at::kFloat);
    auto img0 = TorchImageUtils::TensorsToImage(t0);
    img0.Save("D://tt_b.png");

    at::Tensor t = at::ones({ 4, 3, 1, 64, 64 }, at::kFloat);
    auto img = TorchImageUtils::TensorsToImage(t);
    img.Save("D://tt_seq.png");
    */

    Settings::PrintCudaInfo();

    //torch::nn::MSELoss loss;
    //torch::nn::ModuleHolder mm = loss;
    //torch::nn::Module mod = torch::nn::MSELoss();
    
    static std::shared_ptr<PredictionEvaluator> predEval = std::make_shared<PredictionEvaluatorSigmoid>();

    BceDiceLoss bceLoss;
    MultiBceLoss multiLoss;

    Settings sets;
    //-----
    //model debug
    sets.numWorkers = 0;
    sets.device = torch::kCPU;
    sets.perf.enableAutoCast = false;
    //-----

    //sets.numWorkers = 4;
    //sets.device = torch::kCUDA; //torch::kCUDA;    
    //sets.perf.enableAutoCast = true;
    sets.batchSize = 3;
    sets.metricsInitFn = [predEval= predEval]() -> auto {
        auto metr = std::make_shared<MetricsImage>(MetricsImage::MetricsType::SEGMENTATION);
        metr->SetPredictionEvaluator(predEval);
        return metr;
    };
    sets.lossFn = [&](const auto& output, const auto& targets) {
        //return torch::binary_cross_entropy(output[0], targets);
        //return torch::binary_cross_entropy_with_logits(output[0], targets);
        return bceLoss(output[0], targets);
        //return multiLoss(output, targets);
     };

   
   
    ImageSize imSize(3, 256, 256);

    InputLoaderSettings loaderSets;
    //loaderSets.subsetSize = 200;

    auto ilw = std::make_shared<InputLoadersWrapper>(imSize);
    ilw->InitLoaders<SegmentationInputLoader, std::string>({{ RunMode::TRAIN, loaderSets }}, "E:\\Datasets\\Skyfinder");

    //auto trainLoader = ilw->GetLoader(RunMode::TRAIN);
    
    //auto m = std::make_shared<ModelZoo::unet::UNetModel>(imSize.channels, 1, imSize.width, imSize.height);
    auto m = std::make_shared<ModelZoo::u2net::U2NetModel>(imSize.channels, 1);
        
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