#include "./setup_sdvae.h"

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

#include "../../InputProcessing/InputLoaders/EncoderDecoderInputLoader.h"

//=========================================================
// ModelZoo
//=========================================================

#include "../../ModelZoo/SDVAE/SDVAEModel.h"

//=========================================================
// Utils
//=========================================================

#include "../../Utils/TorchUtils.h"
#include "../../Utils/TorchImageUtils.h"
#include "../../Utils/TrainingHelper.h"

//=========================================================


namespace CustomScenarios::SDVAETraining
{
	void runEncodeDecode(std::shared_ptr<ModelZoo::sdvae::SDVAEModel> m, std::shared_ptr<InputLoader> loader)
	{
		loader->Load();

		DataLoaderData ld(0);
		loader->FillData(0, ld);
		
		//add batch
		ld.Unsqueeze(0);

		auto encoded = m->encoder->forward(ld.input);

		auto decoded = m->decoder->forward(std::get<0>(encoded));
		
		auto img = TorchImageUtils::TensorsToImage(decoded);
		img.Save("D://decoded.png");
	}

	void setup()
	{
				
		MultiBceLoss multiLoss;

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
		sets.batchSize = 3;
		sets.metricsInitFn = []() -> auto {
			auto metr = std::make_shared<MetricsImage>(MetricsImage::MetricsType::UNKNOWN);			
			return metr;
		};
		sets.lossFn = [&](const auto& output, const auto& targets) {
			return multiLoss(output, targets);
		};

		//if crashes with openMp - disable it
		// Assertion failed: nthr_ == nthr, file C:\actions-runner\_work\pytorch\pytorch\pytorch\third_party\ideep\mkl-dnn\src\common/dnnl_thread.hpp, line 293    
		//at::globalContext().setUserEnabledMkldnn(false);

		ImageSize imSize(3, 256, 256);
		
		InputLoaderSettings loaderSets;
		loaderSets.subsetSize = 200;

		auto ilw = std::make_shared<InputLoadersWrapper>(imSize);
		ilw->InitLoaders<EncoderDecoderInputLoader, std::string>({ { RunMode::TRAIN, loaderSets } }, "D:\\Datasets\\Skyfinder");
				
		auto m = std::make_shared<ModelZoo::sdvae::SDVAEModel>();

		
		runEncodeDecode(m, ilw->GetLoader(RunMode::TRAIN));


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