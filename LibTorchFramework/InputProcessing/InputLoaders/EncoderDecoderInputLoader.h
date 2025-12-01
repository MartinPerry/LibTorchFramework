#ifndef ENCODER_DECODER_INPUT_LOADER_H
#define ENCODER_DECODER_INPUT_LOADER_H

#include <string>
#include <format>

#include "../InputLoader.h"

#include "../../core/Structures.h"

class EncoderDecoderInputLoader : public InputLoader
{
public:

    EncoderDecoderInputLoader(RunMode type, std::weak_ptr<InputLoadersWrapper> parent,
        const std::string& datasetPath);
    virtual ~EncoderDecoderInputLoader() = default;

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
    };

    struct FileInfo
    {
        std::string fn; //file path
        std::string dirId; //id of directoy (image_id)

        FileInfo(const std::string& fn, const std::string& dirId) :
            fn(fn),
            dirId(dirId)
        {
        }

        FileInfo(const FileInfo& fi) :
            fn(fi.fn),
            dirId(fi.dirId)
        {
        }

        // Copy assignment operator
        FileInfo& operator=(const FileInfo& fi)
        {
            if (this != &fi) // protect against self-assignment
            {
                fn = fi.fn;
                dirId = fi.dirId;
            }
            return *this;
        }
    };

    Settings sets;
    std::vector<FileInfo> data;
      
};

#endif
