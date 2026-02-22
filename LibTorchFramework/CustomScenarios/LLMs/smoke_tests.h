#pragma once

class TokenizerBPE;
namespace ModelZoo
{
	namespace llama
	{
		class LlamaForCausalLM;
	}
}

#include <cstdint>
#include <memory>

namespace CustomScenarios
{
	namespace LLMs
	{
		namespace Llama
		{
			void GreedySmokeTestInference(
				std::shared_ptr<ModelZoo::llama::LlamaForCausalLM> model,
				std::shared_ptr<TokenizerBPE> bpe,
				int64_t seqLen = 128,
				int64_t steps = 30
			);

			void SmokeTestInference(
				std::shared_ptr<ModelZoo::llama::LlamaForCausalLM> model,
				std::shared_ptr<TokenizerBPE> bpe,
				int64_t seq_len = 128,
				int64_t steps = 30,
				double temperature = 0.8,
				int64_t top_k = 40,
				double top_p = 0.9,
				double repetition_penalty = 1.15
			);
		}
	}
}
