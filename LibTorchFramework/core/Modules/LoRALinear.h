#ifndef LORA_LINEAR_H
#define LORA_LINEAR_H

#include <unordered_set>
#include <cmath>
#include <memory>
#include <string>

#include <torch/torch.h>

#include "../../Utils/TorchUtils.h"

struct LoRALinearImpl : torch::nn::Module 
{    
    torch::nn::Linear base{ nullptr };      // frozen by default
    int64_t r = 0;
    double alpha = 0.0;
    double scaling = 0.0;

    // dropout is optional; when p<=0 we just skip it in forward
    double dropout_p = 0.0;
    torch::Tensor A; // (r, in)
    torch::Tensor B; // (out, r)

    
    LoRALinearImpl(const torch::nn::Linear& base,
        uint64_t rank,
        double alpha,
        double dropout = 0.0) : 
        base(base),
        r(rank),
        alpha(alpha),
        scaling(alpha / static_cast<double>(r)),
        dropout_p(dropout) 
    {
        if (r <= 0)
        {
            throw std::invalid_argument("LoRA rank r must be > 0");
        }
        
        // Register base as submodule (keeps parameters reachable via this module)
        register_module("base", base);
        
        const int64_t in_features = base->weight.size(1);
        const int64_t out_features = base->weight.size(0);

        AUTO_REGISTER_NEW_PARAMETER(A, torch::empty({ r, in_features }, base->weight.options()));
        AUTO_REGISTER_NEW_PARAMETER(B, torch::empty({ out_features, r }, base->weight.options()));

        // Init: A ~ Kaiming uniform, B = 0
        torch::nn::init::kaiming_uniform_(A, std::sqrt(5.0));
        torch::nn::init::zeros_(B);

        // Freeze base weights by default
        for (auto& p : base->parameters()) 
        {
            p.requires_grad_(false);
        }
    }

    torch::Tensor forward(const torch::Tensor& x) 
    {
        auto y = base->forward(x);

        torch::Tensor x_d = x;
        if ((dropout_p > 0.0) && (is_training()))
        {
            x_d = torch::dropout(x, dropout_p, /*train=*/true);
        }

        // x: (..., in) @ A^T(in, r) -> (..., r)
        auto lora_down = torch::matmul(x_d, A.t());
        
        // (..., r) @ B^T(r, out) -> (..., out)
        auto lora_up = torch::matmul(lora_down, B.t());

        return y + lora_up * scaling;
    }
};

TORCH_MODULE(LoRALinear);

void LoRAWrap(std::shared_ptr<torch::nn::Module> m,
    const std::string& name,
    int64_t rank,
    double alpha,
    double dropout,
    const std::unordered_set<std::string>& targets
);

void LoRAWrap(torch::nn::Module& m,
    const std::string& name,
    int64_t rank,
    double alpha,
    double dropout,
    const std::unordered_set<std::string>& targets
);

#endif
