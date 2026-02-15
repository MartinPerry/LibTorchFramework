#ifndef LLAMA_SAFE_TENSOR_LOADER_H
#define LLAMA_SAFE_TENSOR_LOADER_H

namespace ModelZoo
{
    namespace llama
    {
        struct LlamaConfig;
        struct LlamaForCausalLM;
    }
}

#include "../../core/Snapshot/SafeTensorLoader.h"

namespace ModelZoo
{
    namespace llama
    {

        class LLamaSafeTensorLoader : public SafeTensorLoader
        {
        public:

            TensorMap MapHfKeysToOurs(
                const TensorMap& rawStateDict,
                const LlamaConfig& cfg);

            LoadStateDictReport LoadFromHfSafetensors(
                LlamaForCausalLM& model,                
                const std::filesystem::path& modelDir,                
                bool strict = false);


        };

    }
}


#endif