#ifndef FREEZE_INFO_H
#define FREEZE_INFO_H

#include <string>
#include <vector>

class FreezeInfo 
{
public:
    bool enabled = false;
    std::vector<std::string> unfreezed;

    FreezeInfo(bool enable) : 
        enabled(enable)
    {}

    FreezeInfo(bool enable, const std::vector<std::string>& items) : 
        enabled(enable), 
        unfreezed(items) 
    {
    }

    bool IsFreezeAllEnabled() const
    {
        return enabled && unfreezed.empty();
    }

    bool CanFreeze(const std::string& name) const 
    {
        if (!enabled)
        {
            return false;
        }

        // If any string in unfreezed appears in 'name', then it cannot be frozen
        for (const auto& u : unfreezed) 
        {
            if (name.find(u) != std::string::npos)
            {
                return false;
            }
        }
        return true;
    }
};

#endif