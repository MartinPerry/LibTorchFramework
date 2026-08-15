#ifndef INPUT_LOAD_SETTINGS_H
#define INPUT_LOAD_SETTINGS_H

#include <optional>


struct InputLoaderSettings
{
    std::optional<size_t> subsetSize = std::nullopt;
};

#endif
