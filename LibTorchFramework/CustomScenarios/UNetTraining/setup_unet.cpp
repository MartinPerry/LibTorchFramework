#include "./setup_unet.h"

//=========================================================
// Core
//=========================================================

#include "../../core/Structures.h"
#include "../../core/Runner.h"
#include "../../core/Trainer.h"
#include "../../core/AbstractModel.h"

#include "../../core/Metrics/PredictionEvaluators.h"
#include "../../core/Metrics/MetricsDefault.h"
#include "../../core/Metrics/MetricsImage.h"

#include "../../core/Modules/LossFunctions/DiceLoss.h"
#include "../../core/Modules/LossFunctions/MultiBceLoss.h"

#include "../../core/Snapshot/PretrainedManager.h"
#include "../../core/Snapshot/SnapshotSaver.h"
#include "../../core/Snapshot/SnapshotLoader.h"

//=========================================================
// Inputs
//=========================================================

#include "../../InputProcessing/DefaultDataset.h"
#include "../../InputProcessing/InputLoadersWrapper.h"
#include "../../InputProcessing/InputLoader.h"
#include "../../InputProcessing/DataLoaderData.h"

#include "../../InputProcessing/InputLoaders/SegmentationInputLoader.h"

//=========================================================
// ModelZoo
//=========================================================

#include "../../ModelZoo/UNet/UNetModel.h"

//=========================================================
// Utils
//=========================================================

#include "../../Utils/TorchUtils.h"
#include "../../Utils/TorchImageUtils.h"
#include "../../Utils/TrainingHelper.h"

//=========================================================


namespace CustomScenarios::UNetTraining
{
	void setup()
	{
		static std::shared_ptr<PredictionEvaluator> predEval = std::make_shared<PredictionEvaluatorSigmoid>();

		BceDiceLoss bceLoss;

		Settings sets;
		//-----
		//model debug
		sets.numWorkers = 0;
		sets.device = torch::kCUDA;
		sets.perf.enableAutoCast = true;
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
			return bceLoss(output[0], targets);
		};

		//if crashes with openMp - disable it
		// Assertion failed: nthr_ == nthr, file C:\actions-runner\_work\pytorch\pytorch\pytorch\third_party\ideep\mkl-dnn\src\common/dnnl_thread.hpp, line 293    
		//at::globalContext().setUserEnabledMkldnn(false);

		ImageSize imSize(3, 256, 256);
		ImageSize outSize(1, imSize.width, imSize.height);

		InputLoaderSettings loaderSets;
		loaderSets.subsetSize = 200;

		auto ilw = std::make_shared<InputLoadersWrapper>(imSize);
		ilw->InitLoaders<SegmentationInputLoader, std::string>({ { RunMode::TRAIN, loaderSets } }, "D:\\Datasets\\Skyfinder");

		auto m = std::make_shared<ModelZoo::unet::UNetModel>(imSize, outSize);
		
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

		TrainingHelper th(sets, m);
		th.Run(ilw);
	}
}