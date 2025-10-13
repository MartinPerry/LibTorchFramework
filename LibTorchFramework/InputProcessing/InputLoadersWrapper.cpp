#include "./InputLoadersWrapper.h"

#include <ctime>

#include "./InputLoader.h"

InputLoadersWrapper::InputLoadersWrapper(const std::vector<uint16_t>& shape) :
	train(nullptr),
	valid(nullptr),
	test(nullptr),
	shape(shape)
{
	shuffleSeed = int(time(nullptr));

	//ratios to split input dataset
	// -> if each dataset is a separate file
	// set these values to 1.0
	trainRatio = 0.8;
	valRatio = 0.1;

	//if set, given test ratio is used to split dataset
	//if None => test ratio is calculated as: 1 - (valRatio + trainRatio)
	testRatio = std::nullopt;
}

InputLoadersWrapper::~InputLoadersWrapper()
{
}

std::optional<int> InputLoadersWrapper::GetShuffleSeed() const
{
	return this->shuffleSeed;
}

float InputLoadersWrapper::GetTrainRatio() const
{
	return this->trainRatio;
}

float InputLoadersWrapper::GetValRatio() const
{
	return this->valRatio;
}

std::optional<float> InputLoadersWrapper::GetTestRatio() const
{
	return this->testRatio;
}

const std::vector<uint16_t>& InputLoadersWrapper::GetShape() const
{
	return this->shape;
}

std::shared_ptr<InputLoader> InputLoadersWrapper::GetLoader(RunMode type) const
{
	if (type == RunMode::TRAIN)
	{
		return this->train;
	}
	else if (type == RunMode::VALID)
	{
		return this->valid;
	}
	else if (type == RunMode::TEST)
	{
		return this->test;
	}

	return nullptr;
}

void InputLoadersWrapper::SetLoadersSettings(const std::unordered_map<RunMode, InputLoaderSettings>& types)
{
	for (auto& [t, loaderSets] : types)
	{
		if ((t == RunMode::TRAIN) && (this->train != nullptr))
		{
			this->train->SetLoaderSettings(loaderSets);
		}
		else if ((t == RunMode::VALID) && (this->valid != nullptr))
		{
			this->valid->SetLoaderSettings(loaderSets);
		}
		else if ((t == RunMode::TEST) && (this->test != nullptr))
		{
			this->test->SetLoaderSettings(loaderSets);
		}
	}
}