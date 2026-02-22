#include "./TextFilesInputLoader.h"

#include <filesystem>

#include <Utils/Logger.h>


#include "../InputLoadersWrapper.h"

#include "../../Utils/TorchUtils.h"
#include "../../Utils/TorchImageUtils.h"



TextFilesInputLoader::TextFilesInputLoader(
	RunMode type,
	std::weak_ptr<InputLoadersWrapper> parent,
	std::shared_ptr<Tokenizer> tokenizer,
	int32_t seqLen,
	const std::string& datasetPath) :
	InputLoader(type, parent),
	tokenizer(tokenizer),
	seqLen(seqLen)
{
	auto ptr = parent.lock();


}

size_t TextFilesInputLoader::GetSize() const
{
	return 10;
}

void TextFilesInputLoader::Load()
{

}

void TextFilesInputLoader::FillData(size_t index, DataLoaderData& ld)
{
	if (ids.size() < static_cast<size_t>(seqLen + 1))
	{
		//load next line, strip it
		StringUtf8 prompt = (u8"Hello! Briefly explain what weather warnings are.");

		auto tmp = tokenizer->Encode(prompt);

		ids.insert(ids.end(), tmp.begin(), tmp.end());
	}



	std::vector<int64_t> chunk(ids.begin(), ids.begin() + (seqLen + 1));
	ids.erase(ids.begin(), ids.begin() + (seqLen + 1));

	std::vector<int64_t> x_vec(chunk.begin(), chunk.end() - 1);

	std::vector<int64_t> y_vec(chunk.begin() + 1, chunk.end());

	torch::Tensor x = torch::tensor(x_vec, torch::dtype(torch::kLong));
	torch::Tensor y = torch::tensor(y_vec, torch::dtype(torch::kLong));

	ld.input = x;
	ld.target = y;
}