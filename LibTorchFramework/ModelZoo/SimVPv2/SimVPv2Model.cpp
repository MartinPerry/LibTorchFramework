#include "./SimVPv2Model.h"

#include <cmath>
#include <algorithm>

#include "../../core/Modules/WeightsInit/TruncatedNormalInit.h"

#include "../../core/Modules/UpSample2d.h"
#include "../../core/Modules/DownSample2d.h"

#include "../../InputProcessing/DataLoaderData.h"

#include "../../Utils/TorchUtils.h"
#include "../../Utils/TorchImageUtils.h"

using namespace ModelZoo::SimVPv2;


//=================== Helper ===================
static std::vector<bool> sampling_generator(int N, bool reverse)
{
    std::vector<bool> samplings;
    for (int i = 0; i < N / 2; i++)
    {
        samplings.push_back(false);
        samplings.push_back(true);
    }

    if (reverse)
    {
        std::reverse(samplings.begin(), samplings.begin() + N);
    }

    samplings.resize(N);
    return samplings;
}

//=================== BasicConv2d ===================
BasicConv2dImpl::BasicConv2dImpl(int64_t in_channels, int64_t out_channels,
    int64_t kernel_size, int64_t stride,
    int64_t padding, int64_t dilation,
    bool upsampling, bool act_norm_)
{
    act_norm = act_norm_;

    if (upsampling)
    {
        int64_t upscaleFactor = 2;
        AUTO_REGISTER_NEW_MODULE(conv, UpSample2d(in_channels, out_channels, upscaleFactor, kernel_size, padding, dilation));
    }
    else if (stride > 1)
    {
        AUTO_REGISTER_NEW_MODULE(conv, DownSample2d(in_channels, out_channels, stride, kernel_size, padding, dilation));
    }
    else
    {
        AUTO_REGISTER_NEW_MODULE(conv, torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
            .stride(stride)
            .padding(padding)
            .dilation(dilation)));        
    }
    
    
    AUTO_REGISTER_NEW_MODULE(norm, torch::nn::GroupNorm(torch::nn::GroupNormOptions(2, out_channels)));
    AUTO_REGISTER_NEW_MODULE(act, torch::nn::SiLU());

    auto oldVal = TruncatedNormalInit::stdErr;
    TruncatedNormalInit::stdErr = 0.02;
    TruncatedNormalInit(*this);
    TruncatedNormalInit::stdErr = oldVal;
}

torch::Tensor BasicConv2dImpl::forward(const torch::Tensor& x)
{
    torch::Tensor y = conv.forward(x);
    if (act_norm)
    {
        y = act(norm(y));
    }
    return y;
}

//=================== ConvSC ===================
ConvSCImpl::ConvSCImpl(int64_t C_in, int64_t C_out, int64_t kernel_size, bool downsampling, bool upsampling)
{
    int64_t stride = downsampling ? 2 : 1;
    int64_t padding = (kernel_size - stride + 1) / 2;
        
    AUTO_REGISTER_NEW_MODULE(conv, BasicConv2d(C_in, C_out, kernel_size, stride, padding, 1, upsampling, true));
}

torch::Tensor ConvSCImpl::forward(const torch::Tensor& x)
{
    return conv->forward(x);
}

//=================== GroupConv2d ===================
GroupConv2dImpl::GroupConv2dImpl(int64_t in_channels, int64_t out_channels, int64_t kernel_size,
    int64_t stride, int64_t padding, int64_t groups_, bool act_norm_)
{
    act_norm = act_norm_;
    if (in_channels % groups_ != 0) { groups_ = 1; }

    conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
        .stride(stride)
        .padding(padding)
        .groups(groups_));
    norm = torch::nn::GroupNorm(torch::nn::GroupNormOptions(groups_, out_channels));
    activate = torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2).inplace(true));

    AUTO_REGISTER_EXISTING_MODULE(conv);
    AUTO_REGISTER_EXISTING_MODULE(norm);
    AUTO_REGISTER_EXISTING_MODULE(activate);
}

torch::Tensor GroupConv2dImpl::forward(const torch::Tensor& x)
{
    torch::Tensor y = conv->forward(x);
    if (act_norm)
    {
        y = activate(norm(y));
    }
    return y;
}

//=================== gInception_ST ===================
gInception_STImpl::gInception_STImpl(int64_t C_in, int64_t C_hid, int64_t C_out,
    std::vector<int> incep_ker, int groups)
{    
    AUTO_REGISTER_NEW_MODULE(conv1, torch::nn::Conv2d(torch::nn::Conv2dOptions(C_in, C_hid, 1).stride(1).padding(0)));

    for (size_t i = 0; i < incep_ker.size(); i++)
    {
        layers->push_back(GroupConv2d(C_hid, C_out, incep_ker[i], 1, incep_ker[i] / 2, groups, true));
    }
    
    AUTO_REGISTER_EXISTING_MODULE(layers);
}

torch::Tensor gInception_STImpl::forward(const torch::Tensor& x)
{
    torch::Tensor x1 = conv1->forward(x);
    torch::Tensor y = torch::zeros_like(x1);
    for (auto& layer : *layers)
    {                
        y += layer.forward(x1);
    }
    return y;
}

//=================== Encoder ===================
EncoderImpl::EncoderImpl(int64_t C_in, int64_t C_hid, int N_S, int spatio_kernel)
{
    auto samplings = sampling_generator(N_S, false);

    enc->push_back(ConvSC(C_in, C_hid, spatio_kernel, samplings[0]));
    for (size_t i = 1; i < samplings.size(); i++)
    {
        enc->push_back(ConvSC(C_hid, C_hid, spatio_kernel, samplings[i]));
    }

    AUTO_REGISTER_EXISTING_MODULE(enc);
}

std::pair<torch::Tensor, torch::Tensor> EncoderImpl::forward(const torch::Tensor& x)
{
    torch::Tensor enc1 = enc[0]->as<ConvSC>()->forward(x);
    torch::Tensor latent = enc1;
    for (size_t i = 1; i < enc->size(); i++)
    {
        latent = enc[i]->as<ConvSC>()->forward(latent);
    }
    return { latent, enc1 };
}

//=================== Decoder ===================
DecoderImpl::DecoderImpl(int64_t C_hid, int64_t C_out, int N_S, int spatio_kernel)
{
    auto samplings = sampling_generator(N_S, true);
    for (size_t i = 0; i < samplings.size() - 1; i++)
    {
        dec->push_back(ConvSC(C_hid, C_hid, spatio_kernel, false, samplings[i]));
    }
    dec->push_back(ConvSC(C_hid, C_hid, spatio_kernel, false, samplings.back()));
    readout = torch::nn::Conv2d(torch::nn::Conv2dOptions(C_hid, C_out, 1));
    
    AUTO_REGISTER_EXISTING_MODULE(dec);
    AUTO_REGISTER_EXISTING_MODULE(readout);
}

torch::Tensor DecoderImpl::forward(const torch::Tensor& hid, const std::optional<torch::Tensor>& enc1)
{
    torch::Tensor z = hid;
    for (size_t i = 0; i < dec->size() - 1; i++)
    {        
        z = dec[i]->as<ConvSC>()->forward(z);
    }

    torch::Tensor Y;
    if (enc1.has_value() && z.size(0) == enc1.value().size(0))
    {                
        Y = dec[dec->size() - 1]->as<ConvSC>()->forward(z + enc1.value());
    }
    else
    {
        Y = z;
    }

    Y = readout->forward(Y);
    return Y;
}

//=================== GABlock ===================
GABlockImpl::GABlockImpl(int64_t in_ch, int64_t out_ch, float mlp_ratio, float drop, float drop_path)
{
    in_channels = in_ch;
    out_channels = out_ch;    
    AUTO_REGISTER_NEW_MODULE(block, GASubBlock(in_ch, 21, mlp_ratio, drop, drop_path));

    if (in_channels != out_channels)
    {
        AUTO_REGISTER_NEW_MODULE(reduction, torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 1).stride(1).padding(0)));
    }
}

torch::Tensor GABlockImpl::forward(const torch::Tensor& x)
{
    torch::Tensor z = block->forward(x);
    if (in_channels == out_channels)
    {
        return z;
    }
    return reduction->forward(z);
}

//=================== Mid_GANet ===================
Mid_GANetImpl::Mid_GANetImpl(int pastCount, int futureCount, int s, int channel_hid, int N_T_,
    float mlp_ratio, float drop, float drop_path)
{
    N_T = N_T_;
    int64_t channel_in = pastCount * s;
    int64_t channel_out = futureCount * s;

    enc->push_back(GABlock(channel_in, channel_hid, mlp_ratio, drop, drop_path));
    for (int i = 1; i < N_T - 1; i++)
    {
        enc->push_back(GABlock(channel_hid, channel_hid, mlp_ratio, drop, drop_path));
    }
    enc->push_back(GABlock(channel_hid, channel_out, mlp_ratio, drop, drop_path));

    AUTO_REGISTER_EXISTING_MODULE(enc);
}

torch::Tensor Mid_GANetImpl::forward(const torch::Tensor& x)
{
    auto B = x.size(0), T = x.size(1), C = x.size(2), H = x.size(3), W = x.size(4);
    torch::Tensor z = x.view({ B, -1, H, W });

    for (size_t i = 0; i < enc->size(); i++)
    {
        z = enc[i]->as<GABlock>()->forward(z);
    }
    
    return z.view({ B, -1, C, H, W });
}

//=================== Mid_IncepNet ===================
Mid_IncepNetImpl::Mid_IncepNetImpl(int pastCount, int futureCount, int s, int channel_hid, int N_T_,
    std::vector<int> incep_ker, int groups)
{
    N_T = N_T_;
    int64_t channel_in = pastCount * s;
    int64_t channel_out = futureCount * s;

    // Encoder
    enc->push_back(gInception_ST(channel_in, channel_hid / 2, channel_hid, incep_ker, groups));
    for (int i = 1; i < N_T - 1; i++)
    {
        enc->push_back(gInception_ST(channel_hid, channel_hid / 2, channel_hid, incep_ker, groups));
    }
    enc->push_back(gInception_ST(channel_hid, channel_hid / 2, channel_hid, incep_ker, groups));

    // Decoder
    dec->push_back(gInception_ST(channel_hid, channel_hid / 2, channel_hid, incep_ker, groups));
    for (int i = 1; i < N_T - 1; i++)
    {
        dec->push_back(gInception_ST(2 * channel_hid, channel_hid / 2, channel_hid, incep_ker, groups));
    }
    dec->push_back(gInception_ST(2 * channel_hid, channel_hid / 2, channel_out, incep_ker, groups));

    AUTO_REGISTER_EXISTING_MODULE(enc);
    AUTO_REGISTER_EXISTING_MODULE(dec);
}

torch::Tensor Mid_IncepNetImpl::forward(const torch::Tensor& x)
{
    auto B = x.size(0), T = x.size(1), C = x.size(2), H = x.size(3), W = x.size(4);
    torch::Tensor z = x.view({ B, -1, H, W });

    std::vector<torch::Tensor> skips;
    for (size_t i = 0; i < N_T; i++)
    {
        z = enc[i]->as<gInception_ST>()->forward(z);
        if (i < N_T - 1) { skips.push_back(z); }
    }

    z = dec[0]->as<gInception_ST>()->forward(z);
    for (size_t i = 1; i < N_T; i++)
    {
        z = dec[i]->as<gInception_ST>()->forward(torch::cat({ z, skips[N_T - 1 - i] }, 1));
    }

    return z.view({ B, -1, C, H, W });
}

//=================== SimVPv2Model ===================
SimVPv2Model::SimVPv2Model(int past_count, int future_count, const ImageSize& imSize,
    int hid_S, int hid_T, int N_S, int N_T,
    float mlp_ratio, float drop, float drop_path,
    int spatio_kernel_enc, int spatio_kernel_dec,
    const std::string& model_type)
{
    pastCount = past_count;
    futureCount = future_count;
    
    AUTO_REGISTER_NEW_MODULE(enc, Encoder(imSize.channels, hid_S, N_S, spatio_kernel_enc));
    AUTO_REGISTER_NEW_MODULE(dec, Decoder(hid_S, imSize.channels, N_S, spatio_kernel_dec));

    if (model_type == "IncepU")
    {
        AUTO_REGISTER_NEW_MODULE(hid, Mid_IncepNet(past_count, future_count, hid_S, hid_T, N_T));
    }
    else
    {
        AUTO_REGISTER_NEW_MODULE(hid, Mid_GANet(past_count, future_count, hid_S, hid_T, N_T, mlp_ratio, drop, drop_path));
    }    
}

const char* SimVPv2Model::GetName() const
{
    return "SimVPv2Model";
}

torch::Tensor SimVPv2Model::forward(const torch::Tensor& x_raw)
{
    auto B = x_raw.size(0), T = x_raw.size(1), C = x_raw.size(2), H = x_raw.size(3), W = x_raw.size(4);
    torch::Tensor x = x_raw.view({ B * T, C, H, W });

    auto [embed, skip] = enc->forward(x);
    auto C_ = embed.size(1), H_ = embed.size(2), W_ = embed.size(3);
    
    torch::Tensor z = embed.view({ B, T, C_, H_, W_ });
    torch::Tensor hid_out = hid.forward(z);
    hid_out = hid_out.view({ -1, C_, H_, W_ });

    if (pastCount > futureCount)
    {
        auto CS = skip.size(1), HS = skip.size(2), WS = skip.size(3);
        torch::Tensor tmp = skip.view({ B, T, -1 });
        tmp = tmp.slice(1, 0, futureCount);
        skip = tmp.view({ -1, CS, HS, WS });
    }

    torch::Tensor Y = dec->forward(hid_out, skip);
    Y = Y.view({ B, -1, C, H, W });
    return Y;
}


std::vector<torch::Tensor> SimVPv2Model::RunForward(DataLoaderData& batch)
{
    auto x = this->forward(batch.input);

    return { x, batch.target };
}
