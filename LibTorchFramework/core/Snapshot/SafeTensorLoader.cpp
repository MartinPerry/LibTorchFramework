#include "./SafeTensorLoader.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_set>

#include <Utils/Logger.h>

#include "./3rdParty/safetensors.hpp"

#include "../AbstractModel.h"


TensorMap SafeTensorLoader::LoadFromFile(const std::filesystem::path& fileName)
{
	auto tensors = safetensors::load_safetensors(fileName.string());

	return tensors;
}

TensorMap SafeTensorLoader::LoadSafetensorsSharded(const std::filesystem::path& modelDir)
{
	if (!std::filesystem::exists(modelDir))
	{
		MY_LOG_ERROR("Model directory does not exist: %s", modelDir.string().c_str());
		throw std::runtime_error("Model directory does not exist: " + modelDir.string());
	}
	if (!std::filesystem::is_directory(modelDir))
	{
		MY_LOG_ERROR("Model path is not a directory: %s", modelDir.string().c_str());
		throw std::runtime_error("Model path is not a directory: " + modelDir.string());
	}

	const std::filesystem::path indexPath = modelDir / "model.safetensors.index.json";
	TensorMap stateDict;

	if (std::filesystem::exists(indexPath))
	{
		std::vector<std::filesystem::path> shards;
		for (const auto& entry : std::filesystem::directory_iterator(modelDir))
		{
			if (!entry.is_regular_file())
			{
				continue;
			}
			if (entry.path().extension() == ".safetensors")
			{
				shards.push_back(entry.path());
			}
		}
		std::sort(shards.begin(), shards.end());
		for (const auto& shard : shards)
		{
			auto shardData = LoadFromFile(shard);
			MergeTensorMap(stateDict, shardData);
		}
		return stateDict;
	}

	std::filesystem::path singlePath = modelDir / "model.safetensors";
	if (!std::filesystem::exists(singlePath))
	{
		std::vector<std::filesystem::path> candidates;
		for (const auto& entry : std::filesystem::directory_iterator(modelDir))
		{
			if (!entry.is_regular_file())
			{
				continue;
			}
			if (entry.path().extension() == ".safetensors")
			{
				candidates.push_back(entry.path());
			}
		}
		if (candidates.empty())
		{
			MY_LOG_ERROR("No .safetensors files found in: %s", modelDir.string().c_str());
			throw std::runtime_error("No .safetensors files found in: " + modelDir.string());
		}
		std::sort(candidates.begin(), candidates.end());
		singlePath = candidates.front();
	}

	return LoadFromFile(singlePath);
}



void SafeTensorLoader::MergeTensorMap(TensorMap& out, const TensorMap& add) const
{
	for (const auto& [key, value] : add)
	{
		out[key] = value;
	}
}


LoadStateDictReport SafeTensorLoader::LoadMappedStateDict(
	AbstractModel& model,
	const TensorMap& mappedStateDict,
	bool strict)
{
	std::unordered_map<std::string, torch::Tensor*> targetTensors;
	std::vector<std::string> allTargetKeys;

	auto params = model.named_parameters(true);
	for (auto& item : params)
	{
		targetTensors.try_emplace(item.key(), &item.value());
		allTargetKeys.push_back(item.key());
	}

	std::unordered_set<std::string> loadedKeys;
	std::vector<std::string> unexpected;

	{
		torch::NoGradGuard noGrad;
		for (const auto& [key, srcTensor] : mappedStateDict)
		{
			auto it = targetTensors.find(key);
			if (it == targetTensors.end())
			{
				unexpected.push_back(key);
				continue;
			}

			torch::Tensor& dstTensor = *it->second;
			if (dstTensor.sizes() != srcTensor.sizes())
			{				
				MY_LOG_ERROR("Shape mismatch for key '%s'", key.c_str());

				//: model=, checkpoint=
				//	dstTensor.sizes(), srcTensor.sizes());
				
				throw std::runtime_error("Shape mismatch");
			}

			torch::Tensor converted = srcTensor.to(
				dstTensor.device(),
				dstTensor.scalar_type(),
				false,
				false);

			dstTensor.copy_(converted);
			loadedKeys.insert(key);
		}
	}

	std::vector<std::string> missing;
	missing.reserve(allTargetKeys.size());
	for (const auto& key : allTargetKeys)
	{
		if (loadedKeys.find(key) == loadedKeys.end())
		{
			missing.push_back(key);
		}
	}

	if (strict && (!missing.empty() || !unexpected.empty()))
	{
		MY_LOG_ERROR("Strict state-dict load failed");

		//Missing=" << missing.size() ", Unexpected=" << unexpected.size();	
		// 
		throw std::runtime_error("Strict state-dict load failed");	
	}

	return { missing, unexpected };
}
