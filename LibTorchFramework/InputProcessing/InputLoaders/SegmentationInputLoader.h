#ifndef SEGMENTATION_INPUT_LOADER_H
#define SEGMENTATION_INPUT_LOADER_H

#include <string>
#include <format>

#include "../InputLoader.h"

#include "../../core/Structures.h"

class SegmentationInputLoader : public InputLoader
{
public:
    
    SegmentationInputLoader(RunMode type, std::weak_ptr<InputLoadersWrapper> parent,
        const std::string& datasetPath);
    ~SegmentationInputLoader() = default;

    size_t GetSize() const override;
    void Load()  override;
    void FillData(size_t index, DataLoaderData& ld)  override;

protected:
    struct Settings
    {
        std::string datasetPath;
        int imgChannelsCount;
        int maskChannelsCount = 1;
        int imgW;
        int imgH;
    };

    struct FileInfo
    {
        std::string fn;
        std::string dirId;

        FileInfo(const std::string& fn, const std::string& dirId) :
            fn(fn),
            dirId(dirId)
        {}

        FileInfo(const FileInfo& fi) : 
            fn(fi.fn),
            dirId(fi.dirId)
        {}

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

        std::string GetMaskFileName(const std::string& dataRoot) const
        {
            return std::format("{}/{}.png", dataRoot, dirId);            
        }
    };

    SegmentationInputLoader::Settings sets;
    std::vector<FileInfo> data;

    bool IsNumeric(const std::string& s) const;

    torch::Tensor LoadImageAsTensor(const std::string& p, int reqChannelsCount) const;
};

#endif
