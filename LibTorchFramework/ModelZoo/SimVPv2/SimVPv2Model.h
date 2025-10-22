#ifndef SIMVP_V2_MODEL_H
#define SIMVP_V2_MODEL_H

struct ImageSize;

#include <vector>
#include <optional>

#include <torch/torch.h>

#include "./modules.h"

#include "../../core/AbstractModel.h"

namespace ModelZoo {
    namespace SimVPv2 {
                
        //=================== Modules ===================
        struct BasicConv2dImpl : torch::nn::Module
        {
            bool act_norm;
            torch::nn::AnyModule conv;
            torch::nn::GroupNorm norm{ nullptr };
            torch::nn::SiLU act{ nullptr };

            BasicConv2dImpl(int64_t in_channels, int64_t out_channels, int64_t kernel_size,
                int64_t stride, int64_t padding, int64_t dilation = 1, bool upsampling = false, bool act_norm = false);

            torch::Tensor forward(const torch::Tensor& x);
        };
        TORCH_MODULE(BasicConv2d);

        struct ConvSCImpl : torch::nn::Module
        {
            BasicConv2d conv{ nullptr };

            ConvSCImpl(int64_t C_in, int64_t C_out, int64_t kernel_size = 3, bool downsampling = false, bool upsampling = false);
            torch::Tensor forward(const torch::Tensor& x);
        };
        TORCH_MODULE(ConvSC);

        struct GroupConv2dImpl : torch::nn::Module
        {
            bool act_norm;
            torch::nn::Conv2d conv{ nullptr };
            torch::nn::GroupNorm norm{ nullptr };
            torch::nn::LeakyReLU activate{ nullptr };

            GroupConv2dImpl(int64_t in_channels, int64_t out_channels, int64_t kernel_size,
                int64_t stride, int64_t padding, int64_t groups, bool act_norm = false);
            torch::Tensor forward(const torch::Tensor& x);
        };
        TORCH_MODULE(GroupConv2d);

        struct gInception_STImpl : torch::nn::Module
        {
            torch::nn::Conv2d conv1{ nullptr };
            torch::nn::Sequential layers;

            gInception_STImpl(int64_t C_in, int64_t C_hid, int64_t C_out,
                std::vector<int> incep_ker = { 3,5,7,11 }, int groups = 8);
            torch::Tensor forward(const torch::Tensor& x);
        };
        TORCH_MODULE(gInception_ST);

        struct EncoderImpl : torch::nn::Module
        {
            torch::nn::Sequential enc;

            EncoderImpl(int64_t C_in, int64_t C_hid, int N_S, int spatio_kernel);
            std::pair<torch::Tensor, torch::Tensor> forward(const torch::Tensor& x);
        };
        TORCH_MODULE(Encoder);

        struct DecoderImpl : torch::nn::Module
        {
            torch::nn::Sequential dec;
            torch::nn::Conv2d readout{ nullptr };

            DecoderImpl(int64_t C_hid, int64_t C_out, int N_S, int spatio_kernel);
            torch::Tensor forward(const torch::Tensor& hid, const std::optional<torch::Tensor>& enc1 = std::nullopt);
        };
        TORCH_MODULE(Decoder);

        struct GABlockImpl : torch::nn::Module
        {
            int64_t in_channels, out_channels;
            GASubBlock block{ nullptr };
            torch::nn::Conv2d reduction{ nullptr };

            GABlockImpl(int64_t in_channels, int64_t out_channels, float mlp_ratio = 8., float drop = 0.0, float drop_path = 0.0);
            torch::Tensor forward(const torch::Tensor& x);
        };
        TORCH_MODULE(GABlock);

        struct Mid_GANetImpl : torch::nn::Module
        {
            int64_t N_T;
            torch::nn::Sequential enc;

            Mid_GANetImpl(int pastCount, int futureCount, int s, int channel_hid, int N_T,
                float mlp_ratio = 4., float drop = 0.0, float drop_path = 0.1);
            torch::Tensor forward(const torch::Tensor& x);
        
        };
        TORCH_MODULE(Mid_GANet);

        struct Mid_IncepNetImpl : torch::nn::Module
        {
            int64_t N_T;
            torch::nn::Sequential enc;
            torch::nn::Sequential dec;

            Mid_IncepNetImpl(int pastCount, int futureCount, int s, int channel_hid, int N_T,
                std::vector<int> incep_ker = { 3,5,7,11 }, int groups = 8);
            torch::Tensor forward(const torch::Tensor& x);
        };
        TORCH_MODULE(Mid_IncepNet);

        struct SimVPv2Model : public AbstractModel
        {
            int pastCount, futureCount;
            Encoder enc{ nullptr };
            Decoder dec{ nullptr };
            torch::nn::AnyModule hid;

            SimVPv2Model(int past_count, int future_count, const ImageSize& imSize,
                int hid_S = 64, int hid_T = 512, int N_S = 4, int N_T = 8,
                float mlp_ratio = 8.0, float drop = 0.0, float drop_path = 0.0,
                int spatio_kernel_enc = 3, int spatio_kernel_dec = 3,
                const std::string& model_type = "");

            torch::Tensor forward(const torch::Tensor& x_raw);

            const char* GetName() const override;

            std::vector<torch::Tensor> RunForward(DataLoaderData& batch) override;

        };
        
    }
}
#endif