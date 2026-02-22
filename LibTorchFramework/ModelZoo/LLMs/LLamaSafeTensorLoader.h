#ifndef LLAMA_SAFE_TENSOR_LOADER_H
#define LLAMA_SAFE_TENSOR_LOADER_H

namespace ModelZoo
{
    namespace llama
    {
        struct LlamaConfig;
        class LlamaForCausalLM;
    }
}

#include <unordered_map>
#include <string>
#include <filesystem>

#include "../../core/Snapshot/SafeTensorLoader.h"

namespace ModelZoo
{
    namespace llama
    {

        class LLamaSafeTensorLoader : public SafeTensorLoader
        {
        public:

            LoadStateDictReport LoadFromHfSafetensors(
                LlamaForCausalLM& model,                
                const std::filesystem::path& modelDir,                
                bool strict = false);
            

        protected:
            std::unordered_map<std::string, std::string> mapping;

            void CreateMapping(const LlamaConfig& cfg);

            std::string MappingHfKeysToOurs(const std::string& hfName);
          

        };

    }
}


#endif