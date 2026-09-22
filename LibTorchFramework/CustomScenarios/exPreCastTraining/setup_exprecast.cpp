#include "./setup_exprecast.h"

#include <memory>
#include <string>
#include <tuple>
#include <format>

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

#include "../../core/Schedulers/WarmupCosineScheduler.h"

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

#include <RasterData/Colors/ColorUtils.h>
#include <RasterData/Colors/ColorSpace.h>
#include <RasterData/OpticalFlow/Trec.h>
#include <RasterData/OpticalFlow/LucasKanade.h>
#include <RasterData/OpticalFlow/OpticalFlowBase.h>

//=========================================================

#include "./MeteonetInputLoader.h"

namespace CustomScenarios::exPreCastTraining
{


	void AppendFromRaw(float* raw, int imgId, std::vector<Image2d<float>>& result)
	{
		if (raw == nullptr)
		{
			return;
		}

		auto pf = ColorSpace::GetFormatFromChannelsCount<float>(1);

		int imgOffset = imgId * 256 * 256;

		float* tmp = raw + imgOffset;
		result.emplace_back(256, 256, tmp, pf);

	}

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
			dashboard->SetImageUploadEnabled(settings.dashboard.enableImageUpload);
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


		int prevCount = settings.dataset.GetParamAs<int>("prev_count");
		int futureCount = settings.dataset.GetParamAs<int>("future_count");

		InputLoaderSettings loaderSets;
		loaderSets.subsetSize = settings.dataset.subsetSize;
				
		auto ilw = std::make_shared<InputLoadersWrapper>(imSize);	
		ilw->SetShuffleSeed(settings.dataset.seed);
		ilw->SetTrainValTestSplit(0.8, 0.0);
		ilw->InitLoaders<MeteonetInputLoader, std::string>(
			{ 
				{ RunMode::TRAIN, loaderSets }, 
				{ RunMode::TEST, loaderSets } 
			}, 
			settings.dataset.path, prevCount, futureCount, settings.dataset
		);
		
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

			loader->PrecalcVectorField();

#ifdef _WIN32
			

			std::shared_ptr<OpticalFlowBase> flow;

			flow = std::make_shared<LucasKanade>(20);
			//LucasKanadePyramid lkp = LucasKanadePyramid(10);
			//HornSchunck hs;

			Trec::TrecSettings ts;
			ts.kernelRadius = 5;
			ts.matchSearchAreaRadius = 100;
			//flow = std::make_shared<Trec>(ts);


			flow->SetWarpAlgorithm(OpticalFlowBase::WarpAlgorithm::Bicubic);
			
			auto seq0 = loader->GetData(0);

			//TorchImageUtils::TensorsToImageSettings sets;
			//sets.intervalMapping.enabled = false;
			//sets.intervalMapping.mapRange = TorchImageUtils::MappingRange<float>();

			auto imgs = TorchImageUtils::TensorsToImages<float>(seq0.input);
			auto imgsFuture = TorchImageUtils::TensorsToImages<float>(seq0.target);

			auto turboScale = Image2d<uint8_t>("D://turbo.png");

			for (int i = 0; i < imgs.size(); i++)
			{
				imgs[i].Save(std::format("{}/forig_{}.png", "D://W//_0//raw", i).c_str());

				auto r = ColorUtils::MapColorScale<float>(imgs[i], 0, 1, turboScale);				
				r.Save(std::format("{}/orig_{}.png", "D://W//_0", i).c_str());
			}
			for (int i = 0; i < imgsFuture.size(); i++)
			{
				imgsFuture[i].Save(std::format("{}/forig_{}_f.png", "D://W//_0//raw", imgs.size() + i).c_str());

				auto r = ColorUtils::MapColorScale<float>(imgsFuture[i], 0, 1, turboScale);
				r.Save(std::format("{}/orig_{}_f.png", "D://W//_0", imgs.size() + i).c_str());
			}

			auto tmpStart = imgs[imgs.size() - 2];
			auto tmpEnd = imgs[imgs.size() - 1];

			flow->Run(tmpEnd, tmpStart);
			
			float* resData = new float[futureCount * 256 * 256];

			for (int i = 0; i < futureCount; i++)
			{
				//flow->Run(tmpEnd, tmpStart);

				Image2d<float> trecRec = flow->Warp(tmpEnd, -1);

				int imgOffset = i * 256 * 256;
				std::copy(trecRec.GetData().begin(), trecRec.GetData().end(), resData + imgOffset);

				tmpStart = std::move(tmpEnd);
				tmpEnd = std::move(trecRec);
			}

			std::vector<Image2d<float>> resImages;
			for (int i = 0; i < futureCount; i++)
			{				
				AppendFromRaw(resData, i, resImages);
			}
			
			std::vector<Image2d<uint8_t>> resImagesTurbo;

			for (int i = 0; i < resImages.size(); i++)
			{				
				auto r = ColorUtils::MapColorScale<float>(resImages[i], 0, 1, turboScale);
				resImagesTurbo.push_back(r);

				r.Save(std::format("{}/trec_{}.png", "D://W//_0", i).c_str());
			}

			TorchImageUtils::SaveAsGif("D://W//_0//anim.gif", resImagesTurbo);

			printf("");
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
			
			m->CreateScheduler<WarmupCosineScheduler>(settings.training.epochCount, 1e-3);

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

		
		TrainingHelper th(sets, modelIniter, settings.training.gpuCount);
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