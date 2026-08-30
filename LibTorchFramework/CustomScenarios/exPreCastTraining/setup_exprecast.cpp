#include "./setup_exprecast.h"

#include <memory>
#include <string>
#include <tuple>

//=========================================================
// Core
//=========================================================

#include "../../SettingsLoader.h"
#include "../../Settings.h"

#include "../../core/Structures.h"
#include "../../core/Runner.h"
#include "../../core/Trainer.h"
#include "../../core/AbstractModel.h"

#include "../../core/Metrics/PredictionEvaluators.h"
#include "../../core/Metrics/MetricsDefault.h"
#include "../../core/Metrics/MetricsImage.h"
#include "../../core/Metrics/MetricsVideo.h"
#include "../../core/Metrics/MetricsUploader.h"

#include "../../core/Modules/LossFunctions/DiceLoss.h"
#include "../../core/Modules/LossFunctions/MultiBceLoss.h"
#include "../../core/Modules/LossFunctions/FACL.h"

#include "../../core/Snapshot/PretrainedManager.h"
#include "../../core/Snapshot/SnapshotSaver.h"
#include "../../core/Snapshot/SnapshotLoader.h"
#include "../../core/Snapshot/SafeTensorLoader.h"

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

#include <Utils/Strings/StringUtils.h>
#include <Utils/CmdParser.h>

//=========================================================

#include "./MeteonetInputLoader.h"

namespace CustomScenarios::exPreCastTraining
{
	
	void setup(int argc, char** argv)
	{

		CmdParser cmd(argc, argv);

		//./app --config config.json		
		ModelSettings settings = SettingsLoader::LoadFromFile(cmd, "config.json");


		std::shared_ptr<MetricsUploader> dashboard = nullptr;		
		if (settings.dashboard.token != "")
		{
			MetricsUploader::API_URL = settings.dashboard.url;
			MetricsUploader::UPLOAD_TOKEN = settings.dashboard.token;
			
			dashboard = std::make_shared<MetricsUploader>();
			dashboard->SetRunId("exprecast_" + std::to_string(time(0)));
		}
		

		FACL facl(settings.training.epochCount);
		
		Settings sets;
		sets.device = (settings.device == "cpu") ? torch::kCPU : torch::kCUDA;
		sets.numWorkers = settings.training.numWorkers;		
		sets.perf.enableAutoCast = settings.training.autocast;				
		sets.epochCount = settings.training.epochCount;
		sets.batchSize = settings.training.batchSize;
		sets.metricsInitFn = [dashboard]() -> auto {
			auto metr = std::make_shared<MetricsVideo>();

			metr->SetDashborad(dashboard);
			metr->SetCsiThresholds({ 19 / 255.0f, 28 / 255.0f, 35 / 255.0f, 40 / 255.0f, 47 / 255.0f });

			TorchImageUtils::IntervalMapping intervalMapping;
			intervalMapping.enabled = false;
			//sets.intervalMapping.mapRange = TorchImageUtils::MappingRange<float>();
			metr->SetDataMapping(intervalMapping);
#ifdef _WIN32
			metr->SetColorMappingFileName("D://turbo.png");
#else
			metr->SetColorMappingFileName("turbo.png");
#endif

			return metr;
		};
		sets.lossFn = [&](const auto& output, const auto& targets) {
			auto loss = facl(output[0], targets);
			return loss;
		};

		//if crashes with openMp - disable it
		// Assertion failed: nthr_ == nthr, file C:\actions-runner\_work\pytorch\pytorch\pytorch\third_party\ideep\mkl-dnn\src\common/dnnl_thread.hpp, line 293    
		//at::globalContext().setUserEnabledMkldnn(false);

		ImageSize imSize(settings.dataset.channelsCount, settings.dataset.width, settings.dataset.height);


		int prevCount = settings.dataset.prevCount;
		int futureCount = settings.dataset.futureCount;

		InputLoaderSettings loaderSets;
		loaderSets.subsetSize = settings.dataset.subsetSize;
				
		auto ilw = std::make_shared<InputLoadersWrapper>(imSize);	
		ilw->SetShuffleSeed(settings.dataset.seed);
		ilw->SetTrainValTestSplit(0.8, 0.0);
		ilw->InitLoaders<MeteonetInputLoader, std::string>({ { RunMode::TRAIN, loaderSets } },
			settings.dataset.path, prevCount, futureCount);
		ilw->InitLoaders<MeteonetInputLoader, std::string>({ { RunMode::TEST, loaderSets } },
			settings.dataset.path, 		
			prevCount, futureCount);

		//-------
		
		// test
		auto loader = ilw->GetLoader<MeteonetInputLoader>(RunMode::TRAIN);
		if (loader)
		{
			loader->Load();
#ifdef _WIN32
			loader->SaveSequence(0, "D://seq.png", "D://turbo.png");
#else
			loader->SaveSequence(0, "seq.png", "turbo.png");
#endif
		}
		
		//-------
		
		auto modelIniter = [&](size_t device) -> std::shared_ptr<AbstractModel> {

			auto m = std::make_shared<ModelZoo::exPreCast::exPreCastModel>();

			if (settings.snapshot.weights != "")
			{
				SafeTensorLoader tl;
				auto loadRes = tl.LoadModel(settings.snapshot.weights,
					*m.get(), false, [](const std::string& name) -> std::string {
						std::string newName = name;
						StringUtils::ReplaceSubStr(newName, "module.", "");

						return newName;
					});
			}

			//expected input shape: [4, 1, 12, 256, 256]
			//expected output/gt shape: [4, 12, 256, 256]

			m->CreateOptimizer<torch::optim::AdamW>(torch::optim::AdamWOptions(1e-3).weight_decay(0.0));

			sets.pretrainedManager = std::make_shared<PretrainedManager>(settings.snapshot.path);
			sets.pretrainedManager->EnableTrainingSnapshot(true);
			sets.pretrainedManager->EnableSaving(settings.snapshot.enableSave);
			sets.pretrainedManager->EnableLoading(settings.snapshot.enableLoad);

			// 
			//SnapshotSaver saver(m.get());
			//saver.Save(sets.pretrainedManager);

			//SnapshotLoader loader(m.get());
			//loader.Load(sets.pretrainedManager);

			//todo - specify device
			m->to(sets.device);

			return m;
		};

		TrainingHelper th(sets, modelIniter, 1);
		th.Run(ilw);
	}
}

/*

template <typename ModelType>
std::shared_ptr<ModelType> TrainingHelper::CreateModelDeepCopy(std::shared_ptr<ModelType> model)
{
	if (model == nullptr)
	{
		return nullptr;
	}


	auto params = model->named_parameters(true);
	auto clonedParams = clonedModel->named_parameters(true);
	for (auto& item : params)
	{
		auto it = clonedParams.find(item);
		if (it == clonedParams.end())
		{
			continue;
		}

		it->second.copy(item, true);
	}

	auto buffers = model->named_buffers(true);
	auto clonedBuffers = clonedModel->named_buffers(true);
	for (auto& item : buffers)
	{
		auto it = clonedParams.find(item);
		if (it == clonedParams.end())
		{
			continue;
		}

		it->second.copy(item, true);
	}

	return nullptr;
}
*/