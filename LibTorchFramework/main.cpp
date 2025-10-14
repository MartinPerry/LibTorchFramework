

#include <iostream>

#pragma comment(lib, "asmjit.lib")
#pragma comment(lib, "c10.lib")
#pragma comment(lib, "c10_cuda.lib")
#pragma comment(lib, "caffe2_nvrtc.lib")
#pragma comment(lib, "cpuinfo.lib")
#pragma comment(lib, "dnnl.lib")
#pragma comment(lib, "fbgemm.lib")
#pragma comment(lib, "fbjni.lib")
#pragma comment(lib, "kineto.lib")
#pragma comment(lib, "pthreadpool.lib")
#pragma comment(lib, "pytorch_jni.lib")
#pragma comment(lib, "torch.lib")
#pragma comment(lib, "torch_cpu.lib")
#pragma comment(lib, "torch_cuda.lib")
#pragma comment(lib, "XNNPACK.lib")

//debug versions
#pragma comment(lib, "libprotobufd.lib")
#pragma comment(lib, "libprotocd.lib")


#pragma comment(lib, "Playgroundd.lib")

#include "./core/Structures.h"
#include "./core/Runner.h"
#include "./core/Trainer.h"
#include "./core/AbstractModel.h"
#include "./core/Metrics/PredictionEvaluators.h"
#include "./core/Metrics/MetricsDefault.h"
#include "./core/Metrics/MetricsImage.h"
#include "./core/Modules/LossFunctions/DiceLoss.h"
#include "./core/Snapshot/PretrainedManager.h"
#include "./core/Snapshot/SnapshotSaver.h"
#include "./core/Snapshot/SnapshotLoader.h"

#include "./InputProcessing/DefaultDataset.h"
#include "./InputProcessing/InputLoadersWrapper.h"
#include "./InputProcessing/InputLoader.h"
#include "./InputProcessing/DataLoaderData.h"
#include "./InputProcessing/InputLoaders/SegmentationInputLoader.h"

#include "./ModelZoo/UNet/UNetModel.h"

#include "./Utils/TorchUtils.h"
#include "./Utils/TorchImageUtils.h"
#include "./Utils/TrainingHelper.h"

#include "./Settings.h"

#include <Utils/Logger.h>

//https://perception-ml.com/getting-started-with-libtorch/

//https://pytorch.org/tutorials/advanced/cpp_frontend.html

//https://expoundai.wordpress.com/2020/10/13/setting-up-a-cpp-project-in-visual-studio-2019-with-libtorch-1-6/

//https://tebesu.github.io/posts/PyTorch-C++-Frontend

//===================================================================
// Test input loader
//===================================================================

#include <RasterData/Image2d.h>
#include <RasterData/ImageResize.h>

class TestInputLoader : public InputLoader
{
public:
    TestInputLoader(RunMode type, std::weak_ptr<InputLoadersWrapper> parent);
    ~TestInputLoader() = default;

    size_t GetSize() const override;
    void Load()  override;
    void FillData(size_t index, DataLoaderData& ld)  override;


protected:
    std::vector<std::string> data;
};

TestInputLoader::TestInputLoader(RunMode type, std::weak_ptr<InputLoadersWrapper> parent) :
    InputLoader(type, parent)
{
}

size_t TestInputLoader::GetSize() const
{
    return 7;
}

void TestInputLoader::Load()
{
    if (this->data.size() != 0)
    {
        //already loaded
        return;
    }

    std::vector<std::string> tmp;
    for (size_t i = 0; i < 100; i++)
    {
        tmp.emplace_back(std::to_string(i));
    }

    this->data = this->BuildSplits(tmp);

    MY_LOG_INFO("Loaded %d, dataset size: %d", static_cast<int>(this->type), this->data.size());
}

void TestInputLoader::FillData(size_t index, DataLoaderData& ld)
{
    Image2d<uint8_t> img = Image2d<uint8_t>(20, 20, ColorSpace::PixelFormat::RGB);

    img = ImageResize<uint8_t>::ResizeBilinear(img, ImageDimension(30, 30));
    img = *ColorSpace::ConvertToGray(img);
    auto imgf = img.CreateAsMapped<float>(0, 255);
    
    auto t = TorchUtils::make_tensor(imgf.MoveData());
    
    ld.input = at::ones({ 2, 2 }, at::kFloat);
    ld.target = torch::tensor({ (long long)index }, torch::kLong);
    //at::ones({ (long long)index }, at::kLong);
}

//https://discuss.pytorch.org/t/how-to-convert-vector-int-into-c-torch-tensor/66539/4
//https://stackoverflow.com/questions/63466847/how-is-it-possible-to-convert-a-stdvectorstdvectordouble-to-a-torchten

using Example = torch::data::Example<>;

//===================================================================
// Test model
//===================================================================

class TestModel : public AbstractModel
{
public:

    TestModel();
    ~TestModel() = default;

    std::vector<at::Tensor> RunForward(DataLoaderData& batch) override;

protected:

};

TestModel::TestModel()
{    
    this->optimizer = std::make_shared<torch::optim::Adam>(std::vector<at::Tensor>{}, torch::optim::AdamOptions());
}

std::vector<at::Tensor> TestModel::RunForward(DataLoaderData& batch)
{    
    printf("RunForward\n");

    auto xx = torch::tensor({ (long long)1 }, torch::kLong);
    
    return {xx};
}

//===================================================================
//===================================================================
//===================================================================


int main()
{

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

    Settings sets;
    sets.device = torch::kCUDA;
    sets.batchSize = 150;
    sets.metricsInitFn = [predEval= predEval]() -> auto {
        auto metr = std::make_shared<MetricsImage>(MetricsImage::MetricsType::SEGMENTATION);
        metr->SetPredictionEvaluator(predEval);
        return metr;
    };
    sets.lossFn = [&](const auto& output, const auto& targets) {
        //return torch::binary_cross_entropy(output[0], targets);
        //return torch::binary_cross_entropy_with_logits(output[0], targets);
        return bceLoss(output[0], targets);
     };

    //auto ex = Example();
    //ex.data

    //auto m = std::make_shared<TestModel>();

    ImageSize imSize(3, 64, 64);

    InputLoaderSettings loaderSets;
    loaderSets.subsetSize = 200;

    auto ilw = std::make_shared<InputLoadersWrapper>(imSize);
    ilw->InitLoaders<SegmentationInputLoader, std::string>({{ RunMode::TRAIN, loaderSets }}, "D:\\Datasets\\Skyfinder");

    //auto trainLoader = ilw->GetLoader(RunMode::TRAIN);
    
    auto m = std::make_shared<UNetModel>(imSize.channels, 1, imSize.width, imSize.height);
        
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