#ifndef  MRMS_INPUT_LOADER_H
#define  MRMS_INPUT_LOADER_H

#include "../../InputProcessing/InputLoaders/VideoSequenceInputLoader.h"

#include "../../core/Structures.h"

namespace CustomScenarios
{
    namespace MrmsTraining
    {

        class MrmsInputLoader : public VideoSequenceInputLoader
        {
        public:
            MrmsInputLoader(RunMode type, std::weak_ptr<InputLoadersWrapper> parent,
                const std::string& datasetPath,
                int prevSeqLen,
                int futureSeqLen);

        protected:

            void LoadSequenceFiles() override;

            std::vector<float> LoadImage(const std::string& p) const override;
        };
    }
}

#endif // ! MRMS_INPUT_LOADER_H

