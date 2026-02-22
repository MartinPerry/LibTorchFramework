#include "./smoke_tests.h"

#include <vector>

#include <torch/torch.h>

#include "../../core/Tokenizers/TokenizerBPE.h"

#include "../../ModelZoo/LLMs/llama.h"

using namespace ModelZoo::llama;

namespace CustomScenarios::LLMs::Llama
{

	std::vector<int64_t> TensorToInt64Vector(const torch::Tensor& t)
	{
		torch::Tensor cpu = t.to(torch::kCPU, torch::kLong).contiguous();
		const int64_t* p = cpu.data_ptr<int64_t>();
		return std::vector<int64_t>(p, p + cpu.numel());
	}

	void GreedySmokeTestInference(
		std::shared_ptr<ModelZoo::llama::LlamaForCausalLM> model,
		std::shared_ptr<TokenizerBPE> bpe,
		int64_t seqLen,
		int64_t steps)
	{
		auto device = torch::kCUDA;

		StringUtf8 prompt = LlamaConfig::InstructPrompt(u8"Hello! Briefly explain what weather warnings are.\n");


		torch::NoGradGuard noGrad;
		model->eval();
		
		std::vector<TokenId> ids;
		ids = bpe->Encode(prompt, false, false);

		if (ids.size() < 4)
		{
			throw std::runtime_error("Tokenizer returned too few tokens; special tokens may be wrong.");
		}

		if (static_cast<int64_t>(ids.size()) > seqLen)
		{
			ids.resize(static_cast<size_t>(seqLen));
		}

		torch::Tensor x = torch::tensor(ids, torch::TensorOptions().dtype(torch::kLong).device(device)).unsqueeze(0);
		torch::Tensor logits = model->forward(x);

		std::cout << "SMOKE logits: " << logits.sizes() << " dtype: " << logits.dtype() << std::endl;

		const int64_t vocabSize = logits.size(-1);


		if (x.size(1) >= 2)
		{
			torch::Tensor y = x.index({ torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None) }).contiguous();
			torch::Tensor shifted = logits.index({ torch::indexing::Slice(), torch::indexing::Slice(0, -1), torch::indexing::Slice() }).contiguous();
			torch::Tensor loss = torch::nn::functional::cross_entropy(
				shifted.view({ -1, vocabSize }),
				y.view({ -1 }));

			std::cout << "SMOKE loss: " << loss.item<double>() << std::endl;
			if (!torch::isfinite(loss).item<bool>())
			{
				throw std::runtime_error("Loss is not finite; weights or dtype issue.");
			}
		}

		//================================================
		//Greedy generation
		torch::Tensor generated = x.clone();
		for (int64_t i = 0; i < steps; ++i)
		{
			const int64_t start = std::max<int64_t>(0, generated.size(1) - seqLen);
			torch::Tensor context = generated.index(
				{ torch::indexing::Slice(), torch::indexing::Slice(start, torch::indexing::None) });

			logits = model->forward(context);
			torch::Tensor nextId = std::get<1>(logits.index({ 0, -1 }).max(-1, true)).view({ 1, 1 });
			generated = torch::cat({ generated, nextId }, 1);

			if ((bpe->GetEos().id != -1) && (nextId.item<int64_t>() == bpe->GetEos().id))
			{
				break;
			}
		}
		//================================================
		
		std::vector<TokenId> outIds(generated.size(1));
		torch::Tensor generatedCpu = generated.to(torch::kCPU).contiguous();
		auto* ptr = generatedCpu.data_ptr<int64_t>();
		for (int64_t i = 0; i < generatedCpu.size(1); ++i)
		{
			outIds[static_cast<size_t>(i)] = static_cast<TokenId>(ptr[i]);
		}
		
		StringUtf8 decoded = bpe->Decode(outIds);
		std::cout << "\n=== SMOKE GENERATED ===\n" << ((const char*)decoded.c_str()) << "\n======================\n" << std::endl;
		
	}



	void SmokeTestInference(
		std::shared_ptr<ModelZoo::llama::LlamaForCausalLM> model,
		std::shared_ptr<TokenizerBPE> bpe,
		int64_t seq_len,
		int64_t steps,
		double temperature,
		int64_t top_k,
		double top_p,
		double repetition_penalty
	)
	{
		StringUtf8 prompt = LlamaConfig::InstructPrompt(u8"Hello! Briefly explain what weather warnings are.\n");


		auto device = torch::kCUDA;

		torch::NoGradGuard noGrad;
		model->eval();

		std::vector<TokenId> ids;
		ids = bpe->Encode(prompt, false, false);
		
		if (ids.size() < 4)
		{
			throw std::runtime_error("Tokenizer returned too few tokens; special tokens may be wrong.");
		}

		if (static_cast<int64_t>(ids.size()) > seq_len)
		{
			ids.resize(static_cast<size_t>(seq_len));
		}

		torch::Tensor x = torch::tensor(ids, torch::TensorOptions().dtype(torch::kLong).device(device)).unsqueeze(0);
		{
			auto logits = model->forward(x);

			std::cout << "SMOKE logits: (";
			for (int64_t i = 0; i < logits.dim(); ++i)
			{
				std::cout << logits.size(i) << (i + 1 < logits.dim() ? ", " : "");
			}
			std::cout << ") dtype: " << logits.dtype() << "\n";

			const int64_t v = logits.size(-1);
			//if (v != model.vocab_size)
			//{
			//	throw std::runtime_error("Vocab mismatch: logits V=" + std::to_string(v) +
			//		" vs model.vocab_size=" + std::to_string(model.vocab_size));
			//}

			if (x.size(1) >= 2)
			{
				torch::Tensor y = x.index({ torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None) }).contiguous();
				torch::Tensor l = logits.index({ torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, -1), torch::indexing::Slice() }).contiguous();
				torch::Tensor loss = torch::nn::functional::cross_entropy(
					l.view({ -1, v }),
					y.view({ -1 }),
					torch::nn::functional::CrossEntropyFuncOptions()
				);
				std::cout << "SMOKE loss: " << loss.item<double>() << "\n";
				if (!torch::isfinite(loss).all().item<bool>())
				{
					throw std::runtime_error("Loss is not finite -> weights or dtype issues.");
				}
			}
		}

		torch::Tensor logits;
		std::vector<KVCache> kvCache;

		torch::Tensor gen = x.clone();
		std::tie(logits, kvCache) = model->forward_with_cache(x, {}, true);
		
		for (int64_t step = 0; step < steps; ++step)
		{
			torch::Tensor logits_step = logits.index({ 0, -1 }).to(torch::kFloat32).contiguous(); // (V)

			if (repetition_penalty > 1.0)
			{
				std::unordered_set<int64_t> seen;
				for (int64_t tid : TensorToInt64Vector(gen.index({ 0 })))
				{
					seen.insert(tid);
				}
				for (int64_t tid : seen)
				{
					float val = logits_step.index({ tid }).item<float>();
					float adjusted = (val >= 0.0f)
						? (val / static_cast<float>(repetition_penalty))
						: (val * static_cast<float>(repetition_penalty));
					logits_step.index_put_({ tid }, adjusted);
				}
			}

			torch::Tensor next_id;
			if (temperature <= 1e-6)
			{
				next_id = logits_step.argmax().view({ 1, 1 }).to(torch::kLong);
			}
			else
			{
				logits_step = logits_step / temperature;

				if (top_k > 0)
				{
					const int64_t k = std::min(top_k, logits_step.size(0));
					torch::Tensor topk_vals = std::get<0>(torch::topk(logits_step, k));
					torch::Tensor threshold = topk_vals.index({ k - 1 });
					logits_step = torch::where(
						logits_step < threshold,
						torch::full_like(logits_step, -std::numeric_limits<float>::infinity()),
						logits_step
					);
				}

				if (top_p > 0.0 && top_p < 1.0)
				{
					auto sorted = torch::sort(logits_step, -1, true);
					torch::Tensor sorted_logits = std::get<0>(sorted);
					torch::Tensor sorted_indices = std::get<1>(sorted);
					torch::Tensor sorted_probs = torch::softmax(sorted_logits, -1);
					torch::Tensor cumsum = torch::cumsum(sorted_probs, -1);
					torch::Tensor sorted_remove = cumsum > top_p;

					if (sorted_remove.size(0) > 1)
					{
						sorted_remove.index_put_(
							{ torch::indexing::Slice(1, torch::indexing::None) },
							sorted_remove.index({ torch::indexing::Slice(torch::indexing::None, -1) }).clone()
						);
					}
					sorted_remove.index_put_({ 0 }, false);

					torch::Tensor remove = torch::zeros_like(sorted_remove);
					remove.scatter_(0, sorted_indices, sorted_remove);
					logits_step = logits_step.masked_fill(remove, -std::numeric_limits<float>::infinity());
				}

				torch::Tensor probs = torch::softmax(logits_step, -1);
				next_id = torch::multinomial(probs, 1).view({ 1, 1 }).to(torch::kLong);
			}

			torch::Tensor next_id_dev = next_id.to(device);
			gen = torch::cat({ gen, next_id_dev }, 1);

			const int64_t next_token = next_id.item<int64_t>();			
			if ((bpe->GetEos().id != -1) && (next_token == bpe->GetEos().id))
			{
				break;
			}

			if (!kvCache.empty() && kvCache[0].k.size(2) >= seq_len)
			{
				break;
			}

			std::tie(logits, kvCache) = model->forward_with_cache(next_id_dev, kvCache, true);
			
		}
		//================================================
		
		std::vector<TokenId> outIds(gen.size(1));
		torch::Tensor generatedCpu = gen.to(torch::kCPU).contiguous();
		auto* ptr = generatedCpu.data_ptr<int64_t>();
		for (int64_t i = 0; i < generatedCpu.size(1); ++i)
		{
			outIds[static_cast<size_t>(i)] = static_cast<TokenId>(ptr[i]);
		}

		StringUtf8 decoded = bpe->Decode(outIds);
		std::cout << "\n=== SMOKE GENERATED ===\n" << ((const char*)decoded.c_str()) << "\n======================\n" << std::endl;

		//std::string out = bpe.decode(TensorToInt64Vector(gen.index({ 0 })));
		//std::cout << "\n=== SMOKE GENERATED ===\n" << out << "\n======================\n\n";
		
	}
}