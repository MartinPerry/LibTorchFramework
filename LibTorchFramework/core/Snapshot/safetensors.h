// Copyright (c) 2024 - Present, Carson Poole
#ifndef SAFETENSORS_HPP
#define SAFETENSORS_HPP

#include <torch/torch.h>
#include <string>
#include <unordered_map>
#include <vector>
#include <array>
#include <functional>
#include <stdexcept>

#define SAFETENSORS_MAX_DIM 8
#define SAFETENSORS_MAX_TENSORS 2048
#define SAFETENSORS_MAX_FILE_SIZE (2ULL << 40)
#define SAFETENSORS_MAX_STRING_SIZE 2048
#define SAFETENSORS_MAX_METADATA_SIZE 8192

namespace safetensors
{

    class SafetensorsException : public std::runtime_error
    {
    public:
        explicit SafetensorsException(const std::string& message) : 
            std::runtime_error(message) 
        {}

    };

    //==========================================================

    struct TensorInfo
    {
        std::string dtype;
        std::vector<int64_t> shape;
        std::array<size_t, 2> data_offsets;
    };
    
    //==========================================================

    class SimpleJSONParser
    {
    public:
        explicit SimpleJSONParser(const char* json_str);
        std::unordered_map<std::string, TensorInfo> parse();

    private:
        const char* json;
        size_t pos;

        void skipWhitespace();
        std::string parseString();
        std::vector<int64_t> parseArray();
        std::array<size_t, 2> parseDataOffsets();
        void skipValue();
        TensorInfo parseTensorInfo();
    };

    //==========================================================

    class SafeTensorManager
    {
    public:
        using TensorMap = std::unordered_map<std::string, torch::Tensor>;

        SafeTensorManager() = default;
        ~SafeTensorManager() = default;

        TensorMap Load(const std::string& filename);

        TensorMap Load(const std::string& filename,
            std::function<std::string(const std::string&)> remapName);

        void Load(const std::string& filename,
            std::function<void(const std::string&, const torch::Tensor&)> fill);

        void Save(const TensorMap& tensors,
            const std::string& filename,
            const std::unordered_map<std::string, std::string>& metadata = {});

    protected:
        static torch::ScalarType get_torch_dtype(const std::string& dtype_str);
        static std::string_view get_safetensors_dtype(torch::ScalarType dtype);

        static void validate_string_length(const std::string& str, const std::string& context);
        static bool is_big_endian();

        static std::unordered_map<std::string, TensorInfo> ParseHeaderInfo(const char* data, size_t size);

        template <typename T>
        static inline T swap_endian(T u)
        {
            static_assert(CHAR_BIT == 8, "CHAR_BIT != 8");

            union
            {
                T u;
                unsigned char u8[sizeof(T)];
            } source, dest;

            source.u = u;

            for (size_t k = 0; k < sizeof(T); k++)
                dest.u8[k] = source.u8[sizeof(T) - k - 1];

            return dest.u;
        }


        TensorMap Load(const std::string& filename,
            std::function<std::string(const std::string&)> remapName,
            std::function<void(const std::string&, const torch::Tensor&)> fill);

    };
} // namespace safetensors

#endif
