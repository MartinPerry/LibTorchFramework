#ifndef VIDEO_SEQUENCE_INPUT_LOADER_H
#define VIDEO_SEQUENCE_INPUT_LOADER_H

#include <string>
#include <format>
#include <utility>

#include "../InputLoader.h"

#include "../../core/Structures.h"

class VideoSequenceInputLoader : public InputLoader
{
public:

    VideoSequenceInputLoader(RunMode type, std::weak_ptr<InputLoadersWrapper> parent,
        const std::string& datasetPath,
        int prevSeqLen,
        int futureSeqLen);
    virtual ~VideoSequenceInputLoader() = default;

    size_t GetSize() const override;
    void Load()  override;
    void FillData(size_t index, DataLoaderData& ld)  override;

protected:
    struct Settings
    {
        std::string datasetPath;
        int imgChannelsCount;        
        int imgW;
        int imgH;
        int prevSeqLen;
        int futureSeqLen;
    };

    struct SequenceInfo
    {        
        std::string dirPath;
        std::vector<std::string> sequenceFiles;

        SequenceInfo(const std::string& dirPath) :     
            dirPath(dirPath)
        {
        }

        SequenceInfo(const SequenceInfo& si) :
            dirPath(si.dirPath)            
        {
        }

        SequenceInfo& operator=(const SequenceInfo& si)
        {
            if (this != &si)
            {                
                dirPath = si.dirPath;
            }
            return *this;
        }
    };

    VideoSequenceInputLoader::Settings sets;
    std::vector<SequenceInfo> data;

    virtual void LoadSequenceFiles();

    std::vector<float> CreateEmptySequence(int seqLen) const;

    virtual std::vector<float> LoadImage(const std::string& p) const;

    virtual std::pair<torch::Tensor, torch::Tensor> LoadSequence(const SequenceInfo& si) const;
};


#endif // !VIDEO_SEQUENCE_INPUT_LOADER_H

