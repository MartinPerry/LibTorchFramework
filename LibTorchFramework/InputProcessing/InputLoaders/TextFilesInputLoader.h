#ifndef TEXT_FILES_INPUT_LOADER_H
#define TEXT_FILES_INPUT_LOADER_H


#include <vector>
#include <string>
#include <memory>

#include "../InputLoader.h"

#include "../../core/Structures.h"
#include "../../core/Tokenizers/Tokenizers.h"

class TextFilesInputLoader : public InputLoader
{
public:
    TextFilesInputLoader(RunMode type,
        std::weak_ptr<InputLoadersWrapper> parent,
        std::shared_ptr<Tokenizer> tokenizer,
        int32_t seqLen,
        const std::string& datasetPath);
    virtual ~TextFilesInputLoader() = default;


    size_t GetSize() const override;
    void Load()  override;
    void FillData(size_t index, DataLoaderData& ld)  override;


protected:
    std::shared_ptr<Tokenizer> tokenizer;
    int32_t seqLen;
    
};

#endif