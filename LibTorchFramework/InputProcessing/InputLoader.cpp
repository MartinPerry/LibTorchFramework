#include "./InputLoader.h"



InputLoader::InputLoader(RunMode type, std::weak_ptr<InputLoadersWrapper> parent) :
    type(type),
    parent(parent)    
{
}

void InputLoader::SetLoaderSettings(const InputLoaderSettings& loaderSets)
{
    this->loaderSets = loaderSets;
}

void InputLoader::ApplyTransform()
{
    
}