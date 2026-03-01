#include "./LoRALinear.h"

#include "./ChangableModule.h"
#include "./Linear.h"

inline bool isLinearModule(std::shared_ptr<torch::nn::Module> m)
{
    return std::dynamic_pointer_cast<torch::nn::Linear>(m) != nullptr;
}

void LoRAWrap(std::shared_ptr<torch::nn::Module> m,
    const std::string& name,
    int64_t rank,
    double alpha,
    double dropout,
    const std::unordered_set<std::string>& targets
)
{
    LoRAWrap(*m.get(), name, rank, alpha, dropout, targets);
}


void LoRAWrap(torch::nn::Module& m,
    const std::string& name,
    int64_t rank,
    double alpha,
    double dropout,
    const std::unordered_set<std::string>& targets) 
{    
    for (const auto& it : m.named_children())
    {   
        const std::string& child_name = it.key();               
        auto child = it.value();

        if (targets.find(child_name) != targets.end()) 
        {            
            auto parent = dynamic_cast<ChangableModule<torch::nn::AnyModule>*>(&m);
            if (parent)
            {
                auto linearImpl = std::dynamic_pointer_cast<torch::nn::LinearImpl>(child);
                if (linearImpl)
                {

                    auto linear = torch::nn::Linear(linearImpl);

                    auto wrapped = std::make_shared<LoRALinearImpl<torch::nn::Linear>>(linear, rank, alpha, dropout);
                    auto wrappedAny = torch::nn::AnyModule(wrapped);

                    parent->ReplaceModule(child_name, wrappedAny);

                    continue;
                }

                auto cLinearImpl = std::dynamic_pointer_cast<CustomLinearImpl>(child);
                if (cLinearImpl)
                {

                    auto linear = CustomLinear(cLinearImpl);

                    auto wrapped = std::make_shared<LoRALinearImpl<CustomLinear>>(linear, rank, alpha, dropout);
                    auto wrappedAny = torch::nn::AnyModule(wrapped);

                    parent->ReplaceModule(child_name, wrappedAny);

                    continue;
                }
            }
        }
        

        const std::string full = name.empty() ? child_name : (name + "." + child_name);

        LoRAWrap(child, full, rank, alpha, dropout, targets);        
    }
}