#ifndef  METEONET_INPUT_LOADER_H
#define  METEONET_INPUT_LOADER_H

struct DatasetSettings;

#include <optional>
#include <unordered_map>

#include <RasterData/Image2d.h>

#include "../../InputProcessing/InputLoaders/VideoSequenceInputLoader.h"

#include "../../core/Structures.h"

namespace CustomScenarios
{
    namespace exPreCastTraining
    {

        class MeteonetInputLoader : public VideoSequenceInputLoader
        {
        public:
            MeteonetInputLoader(RunMode type, std::weak_ptr<InputLoadersWrapper> parent,
                const std::string& datasetPath,
                int prevSeqLen,
                int futureSeqLen,
                const DatasetSettings& params);

            void Load() override;
            
            void SaveSequence(size_t index, const std::string& outputName,
                std::optional<std::string> colorMappingFileName = std::nullopt);

            void PrecalcVectorField();

        protected:
            int yearFrom;
            int yearTo;
            int seqOverlap;

            void LoadSequenceFiles() override;

            Image2d<float> LoadAsImage(const std::string& p) const;
            std::vector<float> LoadImage(const std::string& p) const override;
        };
    }
}

#endif // ! MRMS_INPUT_LOADER_H

