#include "./exPreCastModel.h"

#include <cmath>
#include <algorithm>
#include <iostream>
#include <optional>

#include "../../InputProcessing/DataLoaderData.h"

#include "../../Utils/TorchUtils.h"
#include "../../Utils/TorchImageUtils.h"

using namespace ModelZoo::exPreCast;

using namespace torch::indexing;


exPreCastModel::exPreCastModel(
    int64_t inputFrames,
    int64_t outputFrames,
    int64_t inChans,
    int64_t outChans,
    std::array<int64_t, 3> patchEmbedSize,
    std::array<int64_t, 3> patchExpandSize,
    std::array<int64_t, 3> upsamplingScale,
    std::array<int64_t, 3> downsamplingScale,
    int64_t embedDim,
    std::vector<int64_t> depths,
    std::vector<int64_t> numHeads,
    std::array<int64_t, 3> windowSize,
    double mlpRatio,
    bool qkvBias,
    double dropRate,
    double attnDropRate,
    double dropPathRate,
    bool patchNorm,
    std::string skipConnection
) :
    inputFrames(inputFrames),
    outputFrames(outputFrames),
    inChans(inChans),
    outChans(outChans),
    embedDim(embedDim),
    patchEmbedSize(patchEmbedSize),
    patchExpandSize(patchExpandSize),
    upsamplingScale(upsamplingScale),
    downsamplingScale(downsamplingScale),
    skipConnection(skipConnection),
    numLayers(depths.size()),
    encoder(register_module("encoder", torch::nn::ModuleList())),
    decoder(register_module("decoder", torch::nn::ModuleList()))
{

    //
    // calculate temporal dimensions
    //
    std::vector<int64_t> encoderTimeDims;
    for (int64_t i = 0; i < numLayers - 1; ++i)
    {
        int64_t value =
            ((inputFrames + 1) / 2) *
            static_cast<int64_t>(std::pow(static_cast<double>(downsamplingScale[0]), i + 1));

        encoderTimeDims.push_back(value);
    }

    std::vector<int64_t> decoderTimeDims;
    if (skipConnection == "concat")
    {
        decoderTimeDims.push_back(encoderTimeDims.back() * upsamplingScale[0] + encoderTimeDims.back());

        for (int64_t i = 0; i < numLayers - 2; ++i)
        {
            decoderTimeDims.push_back(decoderTimeDims.back() * upsamplingScale[0] + encoderTimeDims[encoderTimeDims.size() - (i + 1)]);
        }
    }
    else if (skipConnection == "add")
    {
        decoderTimeDims.push_back(encoderTimeDims.back() * upsamplingScale[0]);

        for (int64_t i = 0; i < numLayers - 2; ++i)
        {
            decoderTimeDims.push_back(decoderTimeDims.back() * upsamplingScale[0]);
        }
    }


    patchEmbed = register_module("patch_embed",
            PatchEmbed3D(
                patchEmbedSize,
                inChans,
                embedDim,
                patchNorm));

    posDrop = register_module("pos_drop", torch::nn::Dropout(dropRate));


    std::vector<float> dpr;
    auto t = torch::linspace(0.0, dropPathRate, std::accumulate(depths.begin(), depths.end(), 0));
    dpr.assign(t.data_ptr<float>(), t.data_ptr<float>() + t.numel());
    


    for (int64_t i = 0; i < numLayers; ++i)
    {
        int begin = std::accumulate(depths.begin(), depths.begin() + i, 0);
        int end = begin + depths[i];

        std::vector<double> layerDpr(
            dpr.begin() + begin,
            dpr.begin() + end
        );

        auto layer = BasicLayerSkip(
                embedDim * (1LL << i),
                depths[i],
                numHeads[i],
                windowSize,
                mlpRatio,
                qkvBias,
                std::nullopt,
                dropRate,
                attnDropRate,
                layerDpr,
                (i < numLayers - 1) ? 
                    BasicLayerSkipImpl::SubsampleType::PatchMergingType : 
                    BasicLayerSkipImpl::SubsampleType::None,
                downsamplingScale
        );

        encoder->push_back(layer);
    }


    const int64_t bottleneckDim = embedDim * (1LL << (numLayers - 1));


    bottleneckUpscale = register_module("bottleneck_upscale",
            CubicDualUpsample( bottleneckDim, upsamplingScale));


    for (int64_t i = numLayers - 2; i >= 0; --i)
    {
        int begin = std::accumulate(depths.begin(), depths.begin() + i, 0);
        int end = begin + depths[i];

        std::vector<double> layerDpr(
            dpr.begin() + begin,
            dpr.begin() + end
        );

        auto layer = BasicLayerSkip(
                embedDim * (1LL << i),
                depths[i],
                numHeads[i],
                windowSize,
                mlpRatio,
                qkvBias,
                std::nullopt,
                dropRate,
                attnDropRate,
                layerDpr,
                (i > 0) ?
                    BasicLayerSkipImpl::SubsampleType::CubicDualUpsampleType : 
                    BasicLayerSkipImpl::SubsampleType::None,
                upsamplingScale
            );

        decoder->push_back(layer);
    }


    patchExpand3D = register_module("patch_expand3d",
        PatchExpanding3D(patchExpandSize, embedDim, outChans));


    // Equivalent of:
    // last_time_dim = decoder_time_dims[-1] * patch_expan_size[0]
    //
    // TODO:
    // Exact temporal dimension depends on encoder/decoder scales.
    lastTimeDim = decoderTimeDims.back() * patchExpandSize[0];



    timeExtractor = register_module("time_extractor",
            torch::nn::Conv3d(
                torch::nn::Conv3dOptions(
                    lastTimeDim,
                    outputFrames,
                    { 3, 3, 1 })
                .stride({ 1, 1, 1 })
                .padding({ 1, 1, 0 })
            )
    );
    
}

const char* exPreCastModel::GetName() const
{
    return "exPreCastModel";
}

torch::Tensor exPreCastModel::forward(torch::Tensor x)
{
    //update loaded shape to match exPrecast input [B, 1, SeqLen, W, H]
    x = x.squeeze(2);
    x = x.unsqueeze(1);

    
    //[4, 1, 12, 256, 256] => [4, 96, 6, 64, 64]
    x = patchEmbed->forward(x);

    x = posDrop->forward(x);


    std::vector<torch::Tensor> skips;
    
    
    for (size_t i = 0; i < encoder->size(); i++)
    //for (auto& module : *encoder)
    {
        auto result = encoder[i]->as<BasicLayerSkipImpl>()->forward(x);
                
        //auto layer = std::dynamic_pointer_cast<BasicLayerSkipImpl>(module.ptr());
        //auto result = layer->forward(x);
                
        x = std::get<0>(result);

        if (i < encoder->size() - 1)
        {
            skips.push_back(std::get<1>(result));
        }
    }


    x = x.permute({ 0, 2, 3, 4, 1 }).contiguous();
    x = bottleneckUpscale->forward(x);
    x = x.permute({ 0, 4, 1, 2, 3}).contiguous();

    for (size_t i = 0; i < decoder->size(); ++i)
    {        
        if (skipConnection == "add")
        {
            x = x + skips[skips.size() - i - 1];
        }
        else 
        {
            //concat
            x = torch::cat({ x, skips[skips.size() - i - 1] }, 2);
        }

        auto result = decoder[i]->as<BasicLayerSkipImpl>()->forward(x);
        //auto layer = std::dynamic_pointer_cast<BasicLayerSkipImpl>(decoder[i].ptr());
        //auto result = layer->forward(x);

        x = std::get<0>(result);
    }


    x = patchExpand3D->forward(x);


    if (lastTimeDim != outputFrames)
    {
        x = x.permute({ 0, 2, 3, 4, 1 }); // B C T H W -> B T H W C
        x = timeExtractor->forward(x);
        x = x.permute({ 0, 4, 1, 2, 3 }); // B T H W C -> B C T H W        
    }

    //update result shape back to match input shape
    x = x.squeeze(1);
    x = x.unsqueeze(2);

    //if (x.isnan().any().item<bool>())
    //{
    //    printf("nan");
    //}
   
    return x;
}

std::vector<torch::Tensor> exPreCastModel::RunForward(DataLoaderData& batch)
{
    //this->eval();
    auto x = this->forward(batch.input);

    return { x };
}
