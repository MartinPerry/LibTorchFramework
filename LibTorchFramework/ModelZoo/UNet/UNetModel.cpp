#include "./UNetModel.h"

#include "../../core/Modules/ModulesOptions.h"
#include "../../core/Modules/UpSample2d.h"

#include "../../InputProcessing/DataLoaderData.h"

#include "../../Utils/TorchImageUtils.h"

using namespace ModelZoo::unet;

SimpleUNetBlockImpl::SimpleUNetBlockImpl(int in_ch,
    int out_ch,
    bool use_padding,
    std::optional<double> dropout_rate)
{
    int padding = 0;
    torch::nn::Conv2dOptions::padding_mode_t padding_mode = torch::kZeros;
    if (use_padding)
    {
        padding = 1;
        padding_mode = torch::kReplicate;
    }

    // conv1
    conv1 = register_module("conv1",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(in_ch, out_ch, 3)
            .padding(padding)
            .padding_mode(padding_mode)
        )
    );

    // ReLU
    relu = register_module("relu", torch::nn::ReLU());

    // conv2
    conv2 = register_module("conv2",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(out_ch, out_ch, 3)
            .padding(padding)
            .padding_mode(padding_mode)
        )
    );

    if (dropout_rate.has_value()) 
    {
        dropout = register_module("dropout", torch::nn::Dropout(torch::nn::DropoutOptions(*dropout_rate)));
        use_dropout = true;
    }
    else 
    {        
        use_dropout = false;
    }
}

torch::Tensor SimpleUNetBlockImpl::forward(torch::Tensor x) 
{
    x = conv1->forward(x);
    x = relu->forward(x);
    if (use_dropout) 
    {
        x = dropout->forward(x);
    }    
    x = conv2->forward(x);
    return x;
}

//===================================================================================

EncoderImpl::EncoderImpl(int inputChannels,
    const std::vector<int>& chs,
    bool use_padding)
{
    // First block
    enc_blocks = register_module("enc_blocks", torch::nn::ModuleList());
    enc_blocks->push_back(SimpleUNetBlock(inputChannels, chs[0], use_padding));

    // Remaining blocks
    for (size_t i = 0; i < chs.size() - 1; i++) {
        enc_blocks->push_back(SimpleUNetBlock(chs[i], chs[i + 1], use_padding));
    }

    // MaxPool2d
    pool = register_module("pool", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2)));
}

std::vector<torch::Tensor> EncoderImpl::forward(torch::Tensor x)
{
    std::vector<torch::Tensor> ftrs;
    for (auto& block_ptr : *enc_blocks) 
    {
        auto block = std::dynamic_pointer_cast<SimpleUNetBlockImpl>(block_ptr);
        x = block->forward(x);
        ftrs.push_back(x);
        x = pool->forward(x);
    }
    return ftrs;
}

//===================================================================================

DecoderImpl::DecoderImpl(const std::vector<int>& chs, bool use_padding) :
    chs(chs)
{
    upconvs = register_module("upconvs", torch::nn::ModuleList());
    dec_blocks = register_module("dec_blocks", torch::nn::ModuleList());

    for (size_t i = 0; i < chs.size() - 1; i++) 
    {
        upconvs->push_back(UpSample2d(ResampleOptions(chs[i], chs[i + 1], 2)));
        dec_blocks->push_back(SimpleUNetBlock(chs[i], chs[i + 1], use_padding));
    }
}

// Manual center crop, equivalent to transforms.CenterCrop in Python
torch::Tensor DecoderImpl::crop(const torch::Tensor& img, const torch::Tensor& target) 
{
    auto h = target.size(2);
    auto w = target.size(3);
    auto Hx = img.size(2);
    auto Wx = img.size(3);

    if (Hx == h && Wx == w) {
        return img;
    }

    auto diffY = Hx - h;
    auto diffX = Wx - w;
    auto startY = diffY / 2;
    auto startX = diffX / 2;

    // slice: keep dimensions [N, C, h, w]
    return img.index({
        torch::indexing::Slice(),
        torch::indexing::Slice(),
        torch::indexing::Slice(startY, startY + h),
        torch::indexing::Slice(startX, startX + w)
    });
}

torch::Tensor DecoderImpl::forward(torch::Tensor x, const std::vector<torch::Tensor>& encoderFeatures) 
{
    for (size_t i = 0; i < chs.size() - 1; i++) 
    {
        auto up = std::dynamic_pointer_cast<UpSample2dImpl>(upconvs[i]);
        x = up->forward(x);

        auto enc_ftrs = crop(encoderFeatures[i], x);
        x = torch::cat({ x, enc_ftrs }, 1);

        auto block = std::dynamic_pointer_cast<SimpleUNetBlockImpl>(dec_blocks[i]);
        x = block->forward(x);
    }

    return x;
}

//===================================================================================

UNetModel::UNetModel(const ImageSize& inputImSize,
    const ImageSize& outputImSize,
    const std::vector<int>& enc_chs,
    const std::vector<int>& dec_chs) : 
    outW(outputImSize.width),
    outH(outputImSize.height)
{
    encoder = Encoder(inputImSize.channels, enc_chs, true);
    decoder = Decoder(dec_chs, true);

    head = torch::nn::Conv2d(torch::nn::Conv2dOptions(dec_chs.back(), outputImSize.channels, 1));

    encoder = register_module("encoder", encoder);
    decoder = register_module("decoder", decoder);
    head = register_module("head", head);
}

const char* UNetModel::GetName() const
{
    return "UNet";
}

torch::Tensor UNetModel::forward(torch::Tensor x)
{
    auto enc_ftrs = encoder(x);
    
    // 2. Reverse the features vector
    std::vector<torch::Tensor> rev_enc_ftrs(enc_ftrs.rbegin(), enc_ftrs.rend());

    // 3. Call decoder: first element is input, rest are skip connections
    torch::Tensor outDec = decoder->forward(rev_enc_ftrs[0],
        std::vector<torch::Tensor>(rev_enc_ftrs.begin() + 1, rev_enc_ftrs.end()));

    auto out = head(outDec);

    bool needInterpolate = ((out.size(2) != outH) || (out.size(3) != outH));

    if (needInterpolate)
    {
        out = torch::nn::functional::interpolate(
            out,
            torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{outH, outW})
            .mode(torch::kBilinear)
            .align_corners(false)
        );
    }

    return out;
}

std::vector<torch::Tensor> UNetModel::RunForward(DataLoaderData& batch)
{
    auto x = this->forward(batch.input);
    
	return {x, batch.target};
}