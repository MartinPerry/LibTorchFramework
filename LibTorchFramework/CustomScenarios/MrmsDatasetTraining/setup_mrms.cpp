#include "./setup_mrms.h"

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

#include "../../InputProcessing/InputLoaders/VideoSequenceInputLoader.h"

//=========================================================
// ModelZoo
//=========================================================

#include "../../ModelZoo/SimVPv2/SimVPv2Model.h"

//=========================================================
// Utils
//=========================================================

#include "../../Utils/TorchUtils.h"
#include "../../Utils/TorchImageUtils.h"
#include "../../Utils/TrainingHelper.h"

//=========================================================

#include "./MrmsInputLoader.h"


namespace CustomScenarios::MrmsTraining
{
	void setup()
	{
		//static std::shared_ptr<PredictionEvaluator> predEval = std::make_shared<PredictionEvaluatorSigmoid>();
		
		Settings sets;
		//-----
		//model debug
		sets.numWorkers = 4;
		sets.device = torch::kCUDA;
		sets.perf.enableAutoCast = true;		
		//-----

		//sets.numWorkers = 4;
		//sets.device = torch::kCUDA; //torch::kCUDA;    
		//sets.perf.enableAutoCast = true;
		sets.batchSize = 2;
		sets.metricsInitFn = []() -> auto {
			auto metr = std::make_shared<MetricsImage>(MetricsImage::MetricsType::UNKNOWN);
			metr->SetColorMappingFileName("D://turbo.png");
			//metr->SetPredictionEvaluator(predEval);
			return metr;
			};
		sets.lossFn = [&](const auto& output, const auto& targets) {
			auto loss = torch::nn::functional::mse_loss(output[0], targets);						
			return loss;
			};

		//if crashes with openMp - disable it
		// Assertion failed: nthr_ == nthr, file C:\actions-runner\_work\pytorch\pytorch\pytorch\third_party\ideep\mkl-dnn\src\common/dnnl_thread.hpp, line 293    
		//at::globalContext().setUserEnabledMkldnn(false);

		ImageSize imSize(1, 256, 256);
		ImageSize outSize(1, imSize.width, imSize.height);

		int prevCount = 5;
		int futureCount = 12;

		InputLoaderSettings loaderSets;
		loaderSets.subsetSize = 200;

		auto ilw = std::make_shared<InputLoadersWrapper>(imSize);
		ilw->InitLoaders<MrmsInputLoader, std::string>({ { RunMode::TRAIN, loaderSets } }, "D:\\Datasets\\mrms_lz4", prevCount, futureCount);

		auto loader = ilw->GetLoader<MrmsInputLoader>(RunMode::TRAIN);
		loader->Load();
		loader->SaveSequence(0, "D://seq.png", "D://turbo.png");


		auto m = std::make_shared<ModelZoo::SimVPv2::SimVPv2Model>(prevCount, futureCount, imSize);

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