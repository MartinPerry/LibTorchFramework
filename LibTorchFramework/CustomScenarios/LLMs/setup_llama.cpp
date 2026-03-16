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
#include "../../core/Modules/LoRALinear.h"

#include "../../core/Optimizers/AdamW8bit.h"
#include "../../core/Optimizers/FusedAdamW8bit.h"
#include "../../core/Optimizers/LAMB.h"

#include "../../core/Snapshot/PretrainedManager.h"
#include "../../core/Snapshot/SnapshotSaver.h"
#include "../../core/Snapshot/SnapshotLoader.h"
#include "../../core/Snapshot/FreezeInfo.h"

#include "../../core/Tokenizers/Tokenizers.h"
#include "../../core/Tokenizers/TokenizerBPE.h"

//=========================================================
// Inputs
//=========================================================

#include "../../InputProcessing/DefaultDataset.h"
#include "../../InputProcessing/InputLoadersWrapper.h"
#include "../../InputProcessing/InputLoader.h"
#include "../../InputProcessing/DataLoaderData.h"

#include "../../InputProcessing/InputLoaders/TextFilesInputLoader.h"

//=========================================================
// ModelZoo
//=========================================================

#include "../../ModelZoo/LLMs/llama.h"
#include "../../ModelZoo/LLMs/LLamaSafeTensorLoader.h"

//=========================================================
// Utils
//=========================================================

#include "../../Utils/ModelInfo.h"
#include "../../Utils/TorchUtils.h"
#include "../../Utils/TorchImageUtils.h"
#include "../../Utils/TrainingHelper.h"

#include <FileUtils/Reading/TextFileReader.h>
#include <Utils/cJSON.h>

//=========================================================

#include "../_tests_/llm_smoke_tests.h"
#include "../_tests_/tokenizer_tests.h"
#include "../_tests_/optimizers_tests.h"

using namespace ModelZoo::llama;

#include <Windows.h>
#include <Psapi.h>
#include <memory>

class LinearTestModule : public torch::nn::Module
{
public:
    LinearTestModule(int64_t in_features, int64_t out_features);

    torch::Tensor forward(const torch::Tensor& x);

private:
    torch::nn::Linear linear_{ nullptr };
};

LinearTestModule::LinearTestModule(int64_t in_features, int64_t out_features)
{
    linear_ = register_module(
        "linear",
        torch::nn::Linear(
            torch::nn::LinearOptions(in_features, out_features)
        )
    );
}

torch::Tensor LinearTestModule::forward(const torch::Tensor& x)
{
    torch::Tensor output = linear_->forward(x);
    return output;
}

void PrintMemory(const char* label)
{
    PROCESS_MEMORY_COUNTERS_EX info{};
    GetProcessMemoryInfo(GetCurrentProcess(),
        reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&info), sizeof(info));

    //Private Bytes 
    // refer to the amount of memory that the process executable has asked for - 
    // not necessarily the amount it is actually using. 

    printf("[%s]\n", label);
    printf("  WorkingSetSize     (current RSS) : %.3f GB\n",
        (float)info.WorkingSetSize / 1024 / 1024 / 1024);
    printf("  PeakWorkingSetSize (high watermark, never drops): %.3f GB\n",
        (float)info.PeakWorkingSetSize / 1024 / 1024 / 1024);
    printf("  PrivateUsage       (committed pages): %.3f GB\n",
        (float)info.PrivateUsage / 1024 / 1024 / 1024);
}

#include <c10/core/CPUAllocator.h>

void ReleaseCPUCache()
{
    // Release LibTorch's internal CPU memory cache
    c10::GetCPUAllocator()->raw_deallocate(nullptr); // no-op but flushes

    // The real one:
    at::DataPtr empty;
    c10::GetDefaultCPUAllocator()->raw_deallocate(nullptr);
}

namespace CustomScenarios::LLMs::Llama
{
    
    void TestLoRA() {
        int64_t in_features = 2048;
        int64_t out_features = 2048;
        int64_t r = 8;
        double alpha = 16.0;
        double dropout = 0.05;

        auto base = torch::nn::Linear(
            torch::nn::LinearOptions(in_features, out_features).bias(false));
        auto lora = std::make_shared<LoRALinearImpl<torch::nn::Linear>>(base, r, alpha, dropout);

        // Dummy input
        auto x = torch::randn({ 2, in_features }, torch::kFloat32);
        x.set_requires_grad(true);

        auto y = lora->forward(x);
        std::cout << "y.requires_grad = " << y.requires_grad() << "\n";
        std::cout << "y.grad_fn       = " << (y.grad_fn() ? "NON-NULL" : "NULL") << "\n";

        auto loss = y.pow(2).mean();
        std::cout << "loss.requires_grad = " << loss.requires_grad() << "\n";
        std::cout << "loss.grad_fn       = " << (loss.grad_fn() ? "NON-NULL" : "NULL") << "\n";

        loss.backward(); // should succeed and give grads on A/B
    }

	void setup()
	{
        //https://huggingface.co/spaces/Xenova/the-tokenizer-playground

        auto t = std::make_shared<LinearTestModule>(10, 30);
        

        //LAMB lamb(t->parameters(), LambOptions(0.01).betas(std::make_tuple(0.9, 0.95)));
        //AdamW8bit lamb(t->parameters(), AdamW8bitOptions());
        FusedAdamW8bit lamb(t->parameters(), FusedAdamW8bitOptions());
        //torch::optim::AdamW lamb(t->parameters(), torch::optim::AdamWOptions(0.01).betas(std::make_tuple(0.9, 0.95)));
        auto ooo = lamb.options();

        //CustomScenarios::_tests_::test_matches_adamw_when_quant_off();
        CustomScenarios::_tests_::test_loss_decreases_toy_regression_adamw8();
        CustomScenarios::_tests_::test_loss_decreases_toy_regression_fused_adamw8();
        
        
        //auto bpeGemma = TokenizerBPE("d://tokenizer_gemma.json");
        //bpeGemma.Load();
        //RunBpeJsonTests("D://res_tokenizer_gemma.json", bpeGemma);
        //auto idsGemma = bpeGemma.Encode(u8"Hello! Briefly explain what weather warnings are.\n", false, false);

        		
        std::string modelDir = "e:\\_models_\\Llama-3.2-3B-Instruct\\";
        //std::string modelDir = "e:\\_models_\\Llama-3.2-1B\\";

        std::shared_ptr<TokenizerBPE> bpe = std::make_shared<TokenizerBPE>("d://tokenizer.json");
        bpe->Load();
        //RunBpeJsonTests("D://res_tokenizer_llama3.2.json", *bpe.get());

        StringUtf8 prompt = LlamaConfig::InstructPrompt(u8"Hello! Briefly explain what weather warnings are.\n");

        std::vector<TokenId> ids;
        ids = bpe->Encode(prompt, false, false);

        //--------------------------------------------------------------------

        Settings sets;
        //-----
        //model debug
        sets.numWorkers = 0;
        sets.epochCount = 1;
        sets.device = torch::kCUDA;
        sets.perf.enableAutoCast = true;
        //-----
        
		LlamaConfig cfg = LlamaConfig::FromJsonFile(modelDir + "config.json");
        cfg.randomInitWeights = false;

        PrintMemory("Fresh start");

        //torch::NoGradGuard();

        /*
        auto emb = CustomEmbedding(
            CustomEmbeddingOptions(cfg.vocab_size, cfg.hidden_size).init_params(cfg.randomInitWeights)
        );
        PrintMemory("After Model Init");
        emb->to(sets.device);

        ReleaseCPUCache();
        SetProcessWorkingSetSize(GetCurrentProcess(), (SIZE_T)-1, (SIZE_T)-1);
        PrintMemory("After to CUDA");
        */

        //SetProcessWorkingSetSize(GetCurrentProcess(), (SIZE_T)-1, (SIZE_T)-1);
        //PrintMemory("2");

        auto llama = std::make_shared<ModelZoo::llama::LlamaForCausalLM>(cfg);        

        ModelInfo mi(*llama.get());

        PrintMemory("After Model Init");
       
        auto memInfo = mi.GetMemorySize();

        MY_LOG_INFO("Model size: CPU> %f GB, GPU> %f GB",
            memInfo.cpuBytes / float(1024 * 1024 * 1024),
            memInfo.gpuBytes / float(1024 * 1024 * 1024));


        ReleaseCPUCache();
        SetProcessWorkingSetSize(GetCurrentProcess(), (SIZE_T)-1, (SIZE_T)-1);
        PrintMemory("After to CUDA");

        llama->to(sets.device);
        
        llama->SetFrozen(std::make_shared<FreezeInfo>(true));

        

        memInfo = mi.GetMemorySize();
       
        MY_LOG_INFO("Model size: CPU> %f GB, GPU> %f GB", 
            memInfo.cpuBytes / float(1024 * 1024 * 1024),
            memInfo.gpuBytes / float(1024 * 1024 * 1024));

        
        sets.batchSize = 1;
        //sets.metricsInitFn = [predEval = predEval]() -> auto {
        //    auto metr = std::make_shared<MetricsImage>(MetricsImage::MetricsType::SEGMENTATION);
        //    metr->SetPredictionEvaluator(predEval);
        //    return metr;
        //    };
        
        sets.lossFn = [&](const auto& output, const auto& targets) {
            //F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

            //[1, 4096, 128256]
            //[1, 4096]
            auto vocab_size = output[0].size(-1);
            auto x = output[0].view({ -1, vocab_size });
            auto gt = targets.view({ -1 });
            auto loss = torch::nn::functional::cross_entropy(x, gt);
            auto lossVal = loss.cpu();

            return loss;
        };


        llama->to(sets.device);
        
        memInfo = mi.GetMemorySize();
        MY_LOG_INFO("Model size: CPU> %f GB, GPU> %f GB",
            memInfo.cpuBytes / float(1024 * 1024 * 1024),
            memInfo.gpuBytes / float(1024 * 1024 * 1024));

        {
            LLamaSafeTensorLoader tl;
            tl.LoadFromHfSafetensors(*llama.get(), modelDir);
        }
        
        //CustomScenarios::_tests_::Llama::GreedySmokeTestInference(llama, bpe, 256, 40);
        //CustomScenarios::_tests_::Llama::SmokeTestInference(llama, bpe, 256, 40);
        //return;

        int lora_r = 8;
        float lora_alpha = 16.0f;
        float lora_dropout = 0.05f;
        std::unordered_set<std::string> targets = { "q_proj", "k_proj", "v_proj", "o_proj" };
        LoRAWrap(llama, "", lora_r, lora_alpha, lora_dropout, targets);

        llama->to(sets.device);

        memInfo = mi.GetMemorySize();
        MY_LOG_INFO("Model size: CPU> %f GB, GPU> %f GB",
            memInfo.cpuBytes / float(1024 * 1024 * 1024),
            memInfo.gpuBytes / float(1024 * 1024 * 1024));
                
        auto params = mi.CountParams();
        MY_LOG_INFO("Params: trainable> %f M, total> %f M", params.trainable / 1e6, params.total / 1e6);


        uint16_t ctxLen = 128;
        auto ilw = std::make_shared<InputLoadersWrapper>(std::vector<uint16_t>{ ctxLen });
        ilw->InitLoaders<TextFilesInputLoader, std::shared_ptr<Tokenizer>, int32_t, std::string>({ RunMode::TRAIN }, bpe, ctxLen,  "");

                
        //llama->CreateOptimizer<torch::optim::AdamW>(torch::optim::AdamWOptions(5e-5).weight_decay(0.01).betas(std::make_tuple(0.9, 0.95)));
        //llama->CreateOptimizer<AdamW8bit>(AdamW8bitOptions());
        llama->CreateOptimizer<FusedAdamW8bit>(FusedAdamW8bitOptions());


        //sets.pretrainedManager = std::make_shared<PretrainedManager>("D://CppTorchModels");
        //sets.pretrainedManager->EnableTrainingSnapshot(false);
        //sets.pretrainedManager->EnableSaving(false);
        //sets.pretrainedManager->EnableLoading(false);

        TrainingHelper th(sets, llama);
        th.Run(ilw);
        
        printf("=====");
	}
}