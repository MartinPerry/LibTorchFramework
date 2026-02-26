#ifndef CHANGABLE_MODULE_H
#define CHANGABLE_MODULE_H


#define AUTO_REGISTER_CHANGABLE_MODULE(var) \
    this->RegisterModule(#var, var)

#define AUTO_REGISTER_CHANGABLE_MODULE(var, arg) \
    if constexpr (std::is_same<decltype(var), torch::nn::AnyModule>::value) { \
        var = torch::nn::AnyModule(arg); \
        this->RegisterModule(#var, var); \
    } else { \
        this->RegisterModule(#var, var); \
    }
    

#include <string>
#include <unordered_map>
#include <tuple>

#include <torch/torch.h>

template <typename... HolderTypes>
struct ChangableModule : torch::nn::Module
{
public:

    virtual ~ChangableModule() = default;

    template <typename HolderT>
    auto ReplaceModule(const std::string& name, HolderT& m)
        -> decltype(m.ptr()) // returns shared_ptr<Impl>
    {         
        auto& tmp = get<HolderT>(); 
        
        auto it = tmp.find(name); 
        if (it == tmp.end()) 
        {
            return m.ptr(); 
        } 
        
        *(it->second) = m; 
        return this->replace_module(name, m); 
    }
    
    std::shared_ptr<torch::nn::Module> ReplaceModule(const std::string& name, torch::nn::AnyModule& m)        
    {
        auto& tmp = get<torch::nn::AnyModule>();

        auto it = tmp.find(name);
        if (it == tmp.end())
        {
            return m.ptr();
        }

        *(it->second) = m;
        return this->replace_module(name, m.ptr());
    }
    
protected:
    std::tuple<std::unordered_map<std::string, HolderTypes*>...> modules;

    ChangableModule() = default;

    template <typename T>
    std::unordered_map<std::string, T*>& get() 
    {       
        static_assert((std::is_same_v<T, HolderTypes> || ...),
            "T must be one of ChangableModule<...> HolderTypes");
        return std::get<std::unordered_map<std::string, T*>>(modules);
    }

    
    template <typename HolderT>
    auto RegisterModule(const std::string& name, HolderT& m) 
        -> decltype(m.ptr()) // returns shared_ptr<Impl>
    {
        // HolderT must be in HolderTypes...
        get<HolderT>().try_emplace(name, &m);
        return this->register_module(name, m);
    }
    
    std::shared_ptr<torch::nn::Module> RegisterModule(const std::string& name, torch::nn::AnyModule& m)       
    {       
        get<torch::nn::AnyModule>().try_emplace(name, &m);
        return this->register_module(name, m.ptr());
    }
    
    /*
    template <typename ModuleType>
    std::shared_ptr<ModuleType> RegisterModule(const std::string& name, torch::nn::ModuleHolder<ModuleType>& m)
    {       
        using HolderT = torch::nn::ModuleHolder<ModuleType>; // e.g. torch::nn::Linear
        get<HolderT>().try_emplace(name, &m);
        return register_module(name, m);
    }
    */
};

#endif