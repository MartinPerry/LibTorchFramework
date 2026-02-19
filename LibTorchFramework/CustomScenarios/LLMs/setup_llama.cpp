#include "./setup_llama.h"

//=========================================================
// Core
//=========================================================

#include "../../core/Structures.h"
#include "../../core/Runner.h"
#include "../../core/Trainer.h"
#include "../../core/AbstractModel.h"

#include "../../core/Metrics/PredictionEvaluators.h"
#include "../../core/Metrics/MetricsDefault.h"
#include "../../core/Metrics/MetricsImage.h"

#include "../../core/Modules/LossFunctions/DiceLoss.h"
#include "../../core/Modules/LossFunctions/MultiBceLoss.h"

#include "../../core/Snapshot/PretrainedManager.h"
#include "../../core/Snapshot/SnapshotSaver.h"
#include "../../core/Snapshot/SnapshotLoader.h"

#include "../../core/Tokenizers/Tokenizers.h"
#include "../../core/Tokenizers/TokenizerBPE.h"

//=========================================================
// Inputs
//=========================================================

#include "../../InputProcessing/DefaultDataset.h"
#include "../../InputProcessing/InputLoadersWrapper.h"
#include "../../InputProcessing/InputLoader.h"
#include "../../InputProcessing/DataLoaderData.h"

#include "../../InputProcessing/InputLoaders/SegmentationInputLoader.h"

//=========================================================
// ModelZoo
//=========================================================

#include "../../ModelZoo/LLMs/llama.h"
#include "../../ModelZoo/LLMs/LLamaSafeTensorLoader.h"

//=========================================================
// Utils
//=========================================================

#include "../../Utils/TorchUtils.h"
#include "../../Utils/TorchImageUtils.h"
#include "../../Utils/TrainingHelper.h"

//=========================================================

using namespace ModelZoo::llama;


namespace CustomScenarios::LLMs::Llama
{
    
    void SmokeTestInference(
        LlamaForCausalLM& model,
        TokenizerBPE& bpe,
        const torch::Device& device,
        int64_t seqLen,
        int64_t steps)
    {
        torch::NoGradGuard noGrad;
        model.eval();

        StringUtf8 prompt = LlamaConfig::InstructPrompt(u8"Hello! Briefly explain what weather warnings are.\n");

        std::vector<TokenId> ids;
        ids = bpe.Encode(prompt, false, false);
        
        if (ids.size() < 4)
        {
            throw std::runtime_error("Tokenizer returned too few tokens; special tokens may be wrong.");
        }

        if (static_cast<int64_t>(ids.size()) > seqLen)
        {
            ids.resize(static_cast<size_t>(seqLen));
        }

        torch::Tensor x = torch::tensor(ids, torch::TensorOptions().dtype(torch::kLong).device(device)).unsqueeze(0);
        torch::Tensor logits = model.forward(x);

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

            logits = model.forward(context);
            torch::Tensor nextId = std::get<1>(logits.index({ 0, -1 }).max(-1, true)).view({ 1, 1 });
            generated = torch::cat({ generated, nextId }, 1);

            if ((bpe.GetEos().id != -1) && (nextId.item<int64_t>() == bpe.GetEos().id))
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

        StringUtf8 decoded = bpe.Decode(outIds);
        std::cout << "\n=== SMOKE GENERATED ===\n" << ((const char*)decoded.c_str()) << "\n======================\n" << std::endl;
        
        model.train();
    }


	void setup()
	{
        //https://huggingface.co/spaces/Xenova/the-tokenizer-playground

        auto bpeGemma = TokenizerBPE("d://tokenizer_gemma.json");
        bpeGemma.Load();
        auto idsGemma = bpeGemma.Encode(u8"Hello! Briefly explain what weather warnings are.", false, false);

		//std::string modelDir = "e:\\Programming\\Python\\test\\PythonApplication1\\py_cpt\\Llama-3.2-3B-Instruct\\";
        std::string modelDir = "e:\\_models_\\Llama-3.2-3B-Instruct\\";

        auto bpe = TokenizerBPE("d://tokenizer.json");
        bpe.Load();
        StringUtf8 prompt = LlamaConfig::InstructPrompt(u8"Hello! Briefly explain what weather warnings are.\n");

        std::vector<TokenId> ids;
        ids = bpe.Encode(prompt, false, false);

		LlamaConfig cfg = LlamaConfig::FromJsonFile(modelDir + "config.json");

		LlamaForCausalLM llama(cfg);

		LLamaSafeTensorLoader tl;
		tl.LoadFromHfSafetensors(llama, modelDir);

		
        auto device = torch::kCUDA;
        
        llama.to(device);

        SmokeTestInference(llama, bpe, device, 256, 40);

        printf("=====");
	}
}