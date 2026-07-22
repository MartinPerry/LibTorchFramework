#include "./setup_exprecast.h"

#include <memory>
#include <string>
#include <tuple>

//=========================================================
// Core
//=========================================================

#include "../../Settings.h"

#include "../../core/Structures.h"
#include "../../core/Runner.h"
#include "../../core/Trainer.h"
#include "../../core/AbstractModel.h"

#include "../../core/Metrics/PredictionEvaluators.h"
#include "../../core/Metrics/MetricsDefault.h"
#include "../../core/Metrics/MetricsImage.h"

#include "../../core/Modules/LossFunctions/DiceLoss.h"
#include "../../core/Modules/LossFunctions/MultiBceLoss.h"
#include "../../core/Modules/LossFunctions/FACL.h"

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

#include "../../ModelZoo/exPreCast/exPreCastModel.h"

//=========================================================
// Utils
//=========================================================

#include "../../Utils/TorchUtils.h"
#include "../../Utils/TorchImageUtils.h"
#include "../../Utils/TrainingHelper.h"

//=========================================================

#include "./MeteonetInputLoader.h"

namespace CustomScenarios::exPreCastTraining
{
	
	void setup()
	{

		int epochCount = 100;

		FACL facl(epochCount);
		
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
		sets.epochCount = epochCount;
		sets.batchSize = 2;
		sets.metricsInitFn = []() -> auto {
			auto metr = std::make_shared<MetricsImage>(MetricsImage::MetricsType::UNKNOWN);
			//metr->SetColorMappingFileName("D://turbo.png");
			return metr;
			};
		sets.lossFn = [&](const auto& output, const auto& targets) {
			auto loss = facl(output[0], targets);
			return loss;
		};

		//if crashes with openMp - disable it
		// Assertion failed: nthr_ == nthr, file C:\actions-runner\_work\pytorch\pytorch\pytorch\third_party\ideep\mkl-dnn\src\common/dnnl_thread.hpp, line 293    
		//at::globalContext().setUserEnabledMkldnn(false);

		ImageSize imSize(1, 256, 256);


		int prevCount = 12;
		int futureCount = 12;

		InputLoaderSettings loaderSets;
		loaderSets.subsetSize = 200;

		auto ilw = std::make_shared<InputLoadersWrapper>(imSize);
		ilw->InitLoaders<MeteonetInputLoader, std::string>({ { RunMode::TRAIN, loaderSets } }, 
			"d:/python/Processing-Radar-Datasets-main/Processing-Radar-Datasets-main/meteonet_256/SE", 
			prevCount, futureCount);

		//-------
		// test
		auto loader = ilw->GetLoader<MeteonetInputLoader>(RunMode::TRAIN);
		loader->Load();
		loader->SaveSequence(0, "D://seq.png", "D://turbo.png");
		//-------

		auto m = std::make_shared<ModelZoo::exPreCast::exPreCastModel>();


		//expected input shape: [4, 1, 12, 256, 256]
		//expected output/gt shape: [4, 12, 256, 256]
		
		m->CreateOptimizer<torch::optim::AdamW>(torch::optim::AdamWOptions(0.0));

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