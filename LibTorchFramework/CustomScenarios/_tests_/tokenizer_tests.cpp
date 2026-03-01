#include "./tokenizer_tests.h"

#include <vector>

#include <Utils/cJSON.h>
#include <FileUtils/Reading/TextFileReader.h>

#include "../../core/Tokenizers/TokenizerBPE.h"

namespace CustomScenarios::_tests_
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

}