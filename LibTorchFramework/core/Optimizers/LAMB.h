#ifndef LAMB_OPTIMIZER_H
#define LAMB_OPTIMIZER_H

#include <tuple>
#include <vector>
#include <unordered_map>

#include <torch/torch.h>

struct LambOptions : public torch::optim::OptimizerCloneableOptions<LambOptions>
{    
    LambOptions(double lr = 1e-3);

    typedef std::tuple<double, double> betas_t;

    TORCH_ARG(double, lr) = 1e-3;    
    TORCH_ARG(betas_t, betas) = std::make_pair(0.9, 0.999);
    TORCH_ARG(double, eps) = 1e-8;
    TORCH_ARG(double, weight_decay) = 1e-2;
    TORCH_ARG(bool, adam) = false;

    void serialize(torch::serialize::InputArchive& archive) override;
    void serialize(torch::serialize::OutputArchive& archive) const override;
    TORCH_API friend bool operator==(
        const LambOptions& lhs,
        const LambOptions& rhs);
    double get_lr() const override;
    void set_lr(const double lr) override;
};

//======================================================================

class LAMB : public torch::optim::Optimizer
{
public:        
    explicit LAMB(const std::vector<torch::Tensor>& params, LambOptions defaults = {});
    explicit LAMB(const std::vector<torch::optim::OptimizerParamGroup>& param_groups, LambOptions defaults = {});    

    torch::Tensor step(torch::optim::Optimizer::LossClosure closure = nullptr) override;

    const LambOptions& options() const noexcept
    {
        return *(dynamic_cast<LambOptions*>(this->defaults_.get()));
    }

private:

    struct ParamState
    {
        int64_t step = 0;
        torch::Tensor exp_avg;
        torch::Tensor exp_avg_sq;

        torch::Tensor weight_norm;   // stored for debug/inspection
        torch::Tensor adam_norm;     // stored for debug/inspection
        torch::Tensor trust_ratio;   // stored for debug/inspection       
    };
    
    std::unordered_map<void*, ParamState> state_;

    ParamState& get_or_init_state(const torch::Tensor& param);
};


#endif