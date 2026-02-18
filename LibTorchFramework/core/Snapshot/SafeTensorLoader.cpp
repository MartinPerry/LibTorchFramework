#include "./SafeTensorLoader.h"

#include <algorithm>
#include <unordered_set>

#include <Utils/Logger.h>

#include "./safetensors.h"

#include "../AbstractModel.h"

/// <summary>
/// Load safetensor to map and return loaded data
/// </summary>
/// <param name="modelDir"></param>
/// <returns></returns>
TensorMap SafeTensorLoader::LoadSafetensors(const std::filesystem::path& modelDir)
{
	return this->LoadSafetensors(modelDir, nullptr);
}

/// <summary>
/// Load safetensor to map and return loaded data
/// name of tensors are remaped based on remapName callback
/// </summary>
/// <param name="modelDir"></param>
/// <param name="remapName"></param>
/// <returns></returns>
TensorMap SafeTensorLoader::LoadSafetensors(const std::filesystem::path& modelDir,
	std::function<std::string(const std::string&)> remapName)
{
	auto shards = this->LoadShardsFileNames(modelDir);
	if (shards.size() == 0)
	{
		return {};
	}

	TensorMap stateDict;
	for (const auto& shard : shards)
	{
		safetensors::SafeTensorManager sm;
		auto shardData = sm.Load(shard.string(), remapName);
		
		this->MergeTensorMap(stateDict, shardData);
	}

	return stateDict;
}

/// <summary>
/// Load safetensor and fioll data directly
/// </summary>
/// <param name="modelDir"></param>
/// <param name="fill"></param>
LoadStateDictReport SafeTensorLoader::LoadSafetensors(const std::filesystem::path& modelDir,
	AbstractModel& model,
	bool strict,
	std::function<std::string(const std::string&)> remapName)
{
	auto shards = this->LoadShardsFileNames(modelDir);
	if (shards.size() == 0)
	{
		return {};
	}

	std::unordered_map<std::string, torch::Tensor*> modelStateDict;

	auto params = model.named_parameters(true);
	for (auto& item : params)
	{
		modelStateDict.try_emplace(item.key(), &item.value());
	}

	std::unordered_set<std::string> loadedKeys;
	std::vector<std::string> unexpected;

	torch::NoGradGuard noGrad;

	TensorMap stateDict;
	for (const auto& shard : shards)
	{
		safetensors::SafeTensorManager sm;
		sm.Load(shard.string(), [&](const std::string& name, const torch::Tensor& t) {
			auto key = (remapName) ? remapName(name) : name;

			auto it = modelStateDict.find(key);
			if (it == modelStateDict.end())
			{
				unexpected.push_back(key);
				return;
			}

			torch::Tensor& dstTensor = *it->second;
			if (dstTensor.sizes() != t.sizes())
			{
				MY_LOG_ERROR("Shape mismatch for key '%s'", key.c_str());
				return;
			}

			torch::Tensor converted = t.to(
				dstTensor.device(),
				dstTensor.scalar_type(),
				false,
				false);
			//.clone() ?
			
			dstTensor.copy_(converted);
			loadedKeys.insert(key);
		});
	}	

	std::vector<std::string> missing;
	missing.reserve(modelStateDict.size());
	for (const auto& [key, _] : modelStateDict)
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
		return {};
	}

	return { missing, unexpected };
}

//======================

std::vector<std::filesystem::path> SafeTensorLoader::LoadShardsFileNames(const std::filesystem::path& modelDir)
{	
	if (!std::filesystem::exists(modelDir))
	{
		MY_LOG_ERROR("Model directory does not exist: %s", modelDir.string().c_str());
		return {};
	}
	if (!std::filesystem::is_directory(modelDir))
	{
		MY_LOG_ERROR("Model path is not a directory: %s", modelDir.string().c_str());
		return {};
	}

	std::vector<std::filesystem::path> shards;

	const std::filesystem::path indexPath = modelDir / "model.safetensors.index.json";
	TensorMap stateDict;

	if (std::filesystem::exists(indexPath))
	{		
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
		
		return shards;
	}

	std::filesystem::path singlePath = modelDir / "model.safetensors";
	if (std::filesystem::exists(singlePath) == false)
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
			return {};
		}
		std::sort(candidates.begin(), candidates.end());
		singlePath = candidates.front();

		shards.push_back(singlePath);
	}
	else 
	{
		shards.push_back(singlePath);
	}

	return shards;
}

void SafeTensorLoader::MergeTensorMap(TensorMap& out, const TensorMap& add) const
{
	for (const auto& [key, value] : add)
	{
		out[key] = value;
	}
}


LoadStateDictReport SafeTensorLoader::FillModelStateDict(
	AbstractModel& model,
	const TensorMap& mappedStateDict,
	bool strict)
{
	std::unordered_map<std::string, torch::Tensor*> modelStateDict;

	auto params = model.named_parameters(true);
	for (auto& item : params)
	{
		modelStateDict.try_emplace(item.key(), &item.value());
	}
	
	std::unordered_set<std::string> loadedKeys;
	std::vector<std::string> unexpected;

	{
		torch::NoGradGuard noGrad;
		for (const auto& [key, srcTensor] : mappedStateDict)
		{
			auto it = modelStateDict.find(key);
			if (it == modelStateDict.end())
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
				
				continue;
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
	missing.reserve(modelStateDict.size());
	for (const auto& [key, _] : modelStateDict)
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
		return {};
	}

	return { missing, unexpected };
}
