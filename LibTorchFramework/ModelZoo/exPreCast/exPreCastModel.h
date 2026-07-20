#ifndef EXPRECAST_MODEL_H
#define EXPRECAST_MODEL_H

#include <vector>
#include <array>

#include <torch/torch.h>

#include "../../core/AbstractModel.h"

#include "./PatchEmbed3D.h"
#include "./BasicLayerSkip.h"
#include "./CubicDualUpsample.h"
#include "./PatchExpanding3D.h"

namespace ModelZoo {
    namespace exPreCast {
        class exPreCastModel : public AbstractModel
        {
        public:
            exPreCastModel(
                int64_t inputFrames = 7,
                int64_t outputFrames = 6,
                int64_t inChans = 1,
                int64_t outChans = 1,
                std::array<int64_t, 3> patchEmbedSize = { 2, 4, 4 },
                std::array<int64_t, 3> patchExpandSize = { 2, 4, 4 },
                std::array<int64_t, 3> upsamplingScale = { 1, 2, 2 },
                std::array<int64_t, 3> downsamplingScale = { 1, 2, 2 },
                int64_t embedDim = 96,
                std::vector<int64_t> depths = { 2, 6, 2, 2 },
                std::vector<int64_t> numHeads = { 3, 6, 12, 24 },
                std::array<int64_t, 3> windowSize = { 2, 7, 7 },
                double mlpRatio = 4.0,
                bool qkvBias = true,
                double dropRate = 0.0,
                double attnDropRate = 0.0,
                double dropPathRate = 0.2,
                bool patchNorm = false,                
                std::string skipConnection = "add"
            );

            const char* GetName() const override;

            torch::Tensor forward(torch::Tensor x);
            
            std::vector<torch::Tensor> RunForward(DataLoaderData& batch) override;

        private:
                  
            int64_t inputFrames;
            int64_t outputFrames;

            int64_t inChans;
            int64_t outChans;

            int64_t embedDim;
            int64_t numLayers;

            std::array<int64_t, 3> patchEmbedSize;
            std::array<int64_t, 3> patchExpandSize;

            std::array<int64_t, 3> upsamplingScale;
            std::array<int64_t, 3> downsamplingScale;

            std::string skipConnection;

            int64_t lastTimeDim;

            PatchEmbed3D patchEmbed{ nullptr };
            torch::nn::Dropout posDrop{ nullptr };

            torch::nn::ModuleList encoder;
            torch::nn::ModuleList decoder;

            CubicDualUpsample bottleneckUpscale{ nullptr };

            PatchExpanding3D patchExpand3D{ nullptr };

            torch::nn::Conv3d timeExtractor{ nullptr };                       
        };
    }
}


#endif