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

#include "./smoke_tests.h"

using namespace ModelZoo::llama;


namespace CustomScenarios::LLMs::Llama
{
    static std::vector<int32_t> JsonGetIntArray(cJSON* obj, const char* key)
    {
        std::vector<int32_t> res;

        cJSON* arr = cJSON_GetObjectItemCaseSensitive(obj, key);
        if (!cJSON_IsArray(arr)) return res;

        const int n = cJSON_GetArraySize(arr);
        res.reserve((size_t)std::max(0, n));

        for (int i = 0; i < n; ++i)
        {
            cJSON* v = cJSON_GetArrayItem(arr, i);                        
            const double d = v->valuedouble;
            const int64_t x = (int64_t)d;            
            res.push_back((int32_t)x);
        }

        return res;
    }

    static void PrintIds(const char* label, const std::vector<int32_t>& ids, size_t maxPrint = 64)
    {
        std::printf("%s[%zu]: ", label, ids.size());
        const size_t n = std::min(ids.size(), maxPrint);
        for (size_t i = 0; i < n; ++i)
        {
            if (i) std::printf(" ");
            std::printf("%d", ids[i]);
        }
        if (ids.size() > maxPrint) std::printf(" ...");
        std::printf("\n");
    }

    void RunBpeJsonTests(const char* jsonPath, TokenizerBPE& tok)
    {
        
        TextFileReader tf(jsonPath);
        std::string json = tf.GetText();
        tf.Close();

        
        cJSON* root = cJSON_ParseWithLength(json.c_str(), json.size());
        
        const int count = cJSON_GetArraySize(root);
        
        for (int i = 0; i < count; ++i)
        {
            cJSON* item = cJSON_GetArrayItem(root, i);

            StringUtf8 prompt = AsStringUtf8(cJSON_GetObjectItemCaseSensitive(item, "prompt")->valuestring);
            std::vector<int32_t> expected = JsonGetIntArray(item, "ids");

            prompt = AsStringUtf8(cJSON_GetObjectItemCaseSensitive(item, "prompt")->valuestring);



            auto got = tok.Encode(prompt, false, false);

            if (got == expected)
            {
                continue;
            }

            std::printf("---- FAIL #%d ----\n", i);
            
            // Print prompt safely: it may contain NUL; write as hex + best-effort text
            {
                // best-effort text (will truncate at NUL)
                std::printf("Text (best-effort): %s\n", (const char*)(prompt.c_str()));
                // hex dump
                std::printf("Hex: ");
                for (size_t k = 0; k < prompt.size(); ++k)
                {
                    std::printf("%02X", (unsigned char)prompt[k]);
                }
                std::printf("\n");
            }


            PrintIds("Expected ", expected);
            PrintIds("Got      ", got);
            break;
        }

        cJSON_Delete(root);        
    }



	void setup()
	{
        //https://huggingface.co/spaces/Xenova/the-tokenizer-playground

        //auto bpeGemma = TokenizerBPE("d://tokenizer_gemma.json");
        //bpeGemma.Load();
        //RunBpeJsonTests("D://res_tokenizer_gemma.json", bpeGemma);
        //auto idsGemma = bpeGemma.Encode(u8"Hello! Briefly explain what weather warnings are.\n", false, false);

        		
        //std::string modelDir = "e:\\_models_\\Llama-3.2-3B-Instruct\\";
        std::string modelDir = "e:\\_models_\\Llama-3.2-1B\\";

        std::shared_ptr<TokenizerBPE> bpe = std::make_shared<TokenizerBPE>("d://tokenizer.json");
        bpe->Load();
        //RunBpeJsonTests("D://res_tokenizer_llama3.2.json", *bpe.get());

        StringUtf8 prompt = LlamaConfig::InstructPrompt(u8"Hello! Briefly explain what weather warnings are.\n");

        std::vector<TokenId> ids;
        ids = bpe->Encode(prompt, false, false);

        //--------------------------------------------------------------------

		LlamaConfig cfg = LlamaConfig::FromJsonFile(modelDir + "config.json");

        auto llama = std::make_shared<ModelZoo::llama::LlamaForCausalLM>(cfg);
        llama->SetFrozen(std::make_shared<FreezeInfo>(true));

        Settings sets;
        //-----
        //model debug
        sets.numWorkers = 4;
        sets.device = torch::kCUDA;
        sets.perf.enableAutoCast = true;
        //-----


        sets.batchSize = 3;
        //sets.metricsInitFn = [predEval = predEval]() -> auto {
        //    auto metr = std::make_shared<MetricsImage>(MetricsImage::MetricsType::SEGMENTATION);
        //    metr->SetPredictionEvaluator(predEval);
        //    return metr;
        //    };
        
        sets.lossFn = [&](const auto& output, const auto& targets) {
            //F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            return torch::cross_entropy_loss(output[0], targets);
        };


        llama->to(sets.device);

        LLamaSafeTensorLoader tl;
        //tl.LoadFromHfSafetensors(*llama.get(), modelDir);
        
        
        //GreedySmokeTestInference(llama, bpe, 256, 40);
        //SmokeTestInference(llama, bpe, 256, 40);
        //return;

        int lora_r = 8;
        float lora_alpha = 16.0f;
        float lora_dropout = 0.05f;
        std::unordered_set<std::string> targets = { "q_proj", "k_proj", "v_proj", "o_proj" };
        LoRAWrap(llama, "", lora_r, lora_alpha, lora_dropout, targets);

        ModelInfo mi(*llama.get());
        auto params = mi.CountParams();
        MY_LOG_INFO("Params trainable: %f M / total %f M", params.trainable / 1e6, params.total / 1e6);



        auto ilw = std::make_shared<InputLoadersWrapper>(std::vector<uint16_t>{ 4096 });
        ilw->InitLoaders<TextFilesInputLoader, std::shared_ptr<Tokenizer>, int32_t, std::string>({ RunMode::TRAIN }, bpe, 4096,  "");

                
        llama->CreateOptimizer<torch::optim::AdamW>(torch::optim::AdamWOptions(5e-5).weight_decay(0.01).betas(std::make_tuple(0.9, 0.95)));
        

        sets.pretrainedManager = std::make_shared<PretrainedManager>("D://CppTorchModels");
        sets.pretrainedManager->EnableTrainingSnapshot(false);
        sets.pretrainedManager->EnableSaving(false);
        sets.pretrainedManager->EnableLoading(false);

        TrainingHelper th(sets, llama);
        th.Run(ilw);
        
        printf("=====");
	}
}