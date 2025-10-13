#ifndef INPUT_LOADERS_WRAPPER_H
#define INPUT_LOADERS_WRAPPER_H

class InputLoader;
struct InputLoaderSettings;
struct Settings;

#include <memory>
#include <optional>
#include <unordered_set>
#include <unordered_map>

#include "../core/Structures.h"

class InputLoadersWrapper : public std::enable_shared_from_this<InputLoadersWrapper>
{
public:
	InputLoadersWrapper(const std::vector<uint16_t>& shape);
	~InputLoadersWrapper();

	template <typename InitedLoaderType>
	std::shared_ptr<InitedLoaderType> GetLoader(RunMode type) const;
	std::shared_ptr<InputLoader> GetLoader(RunMode type) const;
	
	std::optional<int> GetShuffleSeed() const;

	float GetTrainRatio() const;
	float GetValRatio() const;
	std::optional<float> GetTestRatio() const;

	const std::vector<uint16_t>& GetShape() const;

	template <typename InitedLoaderType>
	void InitLoaders(std::unordered_set<RunMode> types);

	template <typename InitedLoaderType, typename... Params>
	void InitLoaders(std::unordered_set<RunMode> types, const Params&... p);

	template <typename InitedLoaderType>
	void InitLoaders(std::unordered_map<RunMode, InputLoaderSettings> types);

	template <typename InitedLoaderType, typename... Params>
	void InitLoaders(std::unordered_map<RunMode, InputLoaderSettings> types, const Params&... p);

protected:
	std::shared_ptr<InputLoader> train;
	std::shared_ptr<InputLoader> valid;
	std::shared_ptr<InputLoader> test;

	std::optional<int> shuffleSeed;

	float trainRatio;
	float valRatio;
	std::optional<float> testRatio;

	std::vector<uint16_t> shape;

	void SetLoadersSettings(const std::unordered_map<RunMode, InputLoaderSettings>& types);
};

//======================================================================================

template <typename InitedLoaderType>
std::shared_ptr<InitedLoaderType> InputLoadersWrapper::GetLoader(RunMode type) const
{
	auto i = this->GetLoader(type);

	return std::dynamic_pointer_cast<InitedLoaderType>(i);
}

template <typename InitedLoaderType>
void InputLoadersWrapper::InitLoaders(std::unordered_set<RunMode> types)
{
	for (auto t : types)
	{
		if ((t == RunMode::TRAIN) && (this->train == nullptr))
		{
			this->train = std::make_shared<InitedLoaderType>(t, weak_from_this());
		}
		else if ((t == RunMode::VALID) && (this->valid == nullptr))
		{
			this->valid = std::make_shared<InitedLoaderType>(t, weak_from_this());
		}
		else if ((t == RunMode::TEST) && (this->test == nullptr))
		{
			this->test = std::make_shared<InitedLoaderType>(t, weak_from_this());
		}
	}
}

template <typename InitedLoaderType, typename... Params>
void InputLoadersWrapper::InitLoaders(std::unordered_set<RunMode> types, const Params&... p)
{
	for (auto t : types)
	{
		if ((t == RunMode::TRAIN) && (this->train == nullptr))
		{
			this->train = std::make_shared<InitedLoaderType>(t, weak_from_this(), p...);
		}
		else if ((t == RunMode::VALID) && (this->valid == nullptr))
		{
			this->valid = std::make_shared<InitedLoaderType>(t, weak_from_this(), p...);
		}
		else if ((t == RunMode::TEST) && (this->test == nullptr))
		{
			this->test = std::make_shared<InitedLoaderType>(t, weak_from_this(), p...);
		}
	}
	
}

template <typename InitedLoaderType>
void InputLoadersWrapper::InitLoaders(std::unordered_map<RunMode, InputLoaderSettings> types)
{
	for (auto& [t, loaderSets] : types)
	{
		if ((t == RunMode::TRAIN) && (this->train == nullptr))
		{
			this->train = std::make_shared<InitedLoaderType>(t, weak_from_this());			
		}
		else if ((t == RunMode::VALID) && (this->valid == nullptr))
		{
			this->valid = std::make_shared<InitedLoaderType>(t, weak_from_this());			
		}
		else if ((t == RunMode::TEST) && (this->test == nullptr))
		{
			this->test = std::make_shared<InitedLoaderType>(t, weak_from_this());			
		}
	}

	this->SetLoadersSettings(types);
}

template <typename InitedLoaderType, typename... Params>
void InputLoadersWrapper::InitLoaders(std::unordered_map<RunMode, InputLoaderSettings> types, const Params&... p)
{
	for (auto& [t, loaderSets] : types)
	{
		if ((t == RunMode::TRAIN) && (this->train == nullptr))
		{
			this->train = std::make_shared<InitedLoaderType>(t, weak_from_this(), p...);			
		}
		else if ((t == RunMode::VALID) && (this->valid == nullptr))
		{
			this->valid = std::make_shared<InitedLoaderType>(t, weak_from_this(), p...);			
		}
		else if ((t == RunMode::TEST) && (this->test == nullptr))
		{
			this->test = std::make_shared<InitedLoaderType>(t, weak_from_this(), p...);
		}
	}

	this->SetLoadersSettings(types);
}


#endif
