#ifndef  METEONET_INPUT_LOADER_H
#define  METEONET_INPUT_LOADER_H

#include <optional>

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
                int futureSeqLen);

            void Load() override;

            void SaveSequence(size_t index, const std::string& outputName,
                std::optional<std::string> colorMappingFileName = std::nullopt);

        protected:

            void LoadSequenceFiles() override;

            std::vector<float> LoadImage(const std::string& p) const override;
        };
    }
}

#endif // ! MRMS_INPUT_LOADER_H

