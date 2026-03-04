#include "./LAMB.h"

#include <cmath>
#include <stdexcept>

/*
Implements Lamb algorithm.
It has been proposed in `Large Batch Optimization for Deep Learning : Training BERT in 76 minutes`_.

lr(float, optional) : learning rate(default: 1e-3)
betas(Tuple[float, float], optional) : coefficients used for computing
            running averages of gradient and its square(default: (0.9, 0.999))
    eps(float, optional) : term added to the denominator to improve
            numerical stability(default: 1e-8)
    weight_decay(float, optional) : weight decay(L2 penalty) (default: 0)
    adam(bool, optional) : always use trust ratio = 1, which turns this into Adam. Useful for comparison purposes.

.._Large Batch Optimization for Deep Learning : Training BERT in 76 minutes :
https ://arxiv.org/abs/1904.00962

*/

// =======================
// Options
// =======================

LambOptions::LambOptions(double lr) :
    lr_(lr)    
{    
}

double LambOptions::get_lr() const
{
    return this->lr_;
}

void LambOptions::set_lr(const double lr)
{
    this->lr_ = lr;
}

// =======================
// Constructor
// =======================

//https://towardsdatascience.com/an-intuitive-understanding-of-the-lamb-optimizer-46f8c0ae4866
//https ://github.com/cybertronai/pytorch-lamb

//https://github.com/pytorch/pytorch/tree/main/torch/csrc/api/src/optim

LAMB::LAMB(const std::vector<torch::Tensor>& params, LambOptions defaults) :
    LAMB({ torch::optim::OptimizerParamGroup(std::move(params)) }, std::move(defaults))
{

}

LAMB::LAMB(const std::vector<torch::optim::OptimizerParamGroup>& param_groups, LambOptions defaults) :
    torch::optim::Optimizer(
        param_groups,
        std::make_unique<torch::optim::AdamWOptions>(0.01)
    )
{    
    TORCH_CHECK(defaults.lr() >= 0, "Invalid learning rate: ", defaults.lr());
    TORCH_CHECK(defaults.eps() >= 0, "Invalid epsilon value: ", defaults.eps());
    auto betas = defaults.betas();
    TORCH_CHECK(
        0 <= std::get<0>(betas) && std::get<0>(betas) < 1.0,
        "Invalid beta parameter at index 0: ",
        std::get<0>(betas));
    TORCH_CHECK(
        0 <= std::get<1>(betas) && std::get<1>(betas) < 1.0,
        "Invalid beta parameter at index 1: ",
        std::get<1>(betas));
    TORCH_CHECK(
        defaults.weight_decay() >= 0,
        "Invalid weight_decay value: ",
        defaults.weight_decay());
}


// =======================
// Step
// =======================

torch::Tensor LAMB::step(torch::optim::Optimizer::LossClosure closure)
{
    torch::Tensor loss;

    if (closure)
    {
        torch::AutoGradMode enable_grad(true);
        loss = closure();
    }
    torch::NoGradGuard no_grad;

    auto& opt = this->options();

    for (auto& group : param_groups())
    {
        for (auto& p : group.params())
        {
            if (!p.grad().defined())
            {
                continue;
            }

            torch::Tensor grad = p.grad();

            if (grad.is_sparse())
            {
                throw std::runtime_error(
                    "LAMB does not support sparse gradients."
                );
            }

            auto& state = get_or_init_state(p);            
            state.step += 1;

            // m_t
            state.exp_avg.mul_(opt.betas().first)
                .add_(grad, 1.0 - opt.betas().first);

            // v_t
            state.exp_avg_sq.mul_(opt.betas().second)
                .addcmul_(grad, grad, 1.0 - opt.betas().second);

            double step_size = opt.lr();

            torch::Tensor weight_norm =
                p.pow(2).sum().sqrt().clamp(0.0, 10.0);

            torch::Tensor adam_step =
                state.exp_avg /
                (state.exp_avg_sq.sqrt().add(opt.eps()));

            if (opt.weight_decay() != 0.0)
            {
                adam_step.add_(p, opt.weight_decay());
            }

            torch::Tensor adam_norm =
                adam_step.pow(2).sum().sqrt();

            torch::Tensor trust_ratio;

            if (weight_norm.item<double>() == 0.0 ||
                adam_norm.item<double>() == 0.0)
            {
                trust_ratio = torch::ones_like(weight_norm);
            }
            else
            {
                trust_ratio = weight_norm / adam_norm;
            }

            state.weight_norm = weight_norm;
            state.adam_norm = adam_norm;
            state.trust_ratio = trust_ratio;

            if (opt.adam())
            {
                trust_ratio = torch::ones_like(trust_ratio);
            }

            p.add_(
                adam_step,
                -step_size * trust_ratio.item<double>()
            );
        }
    }

    return loss;
}

LAMB::ParamState& LAMB::get_or_init_state(const torch::Tensor& param)
{
    auto* key = static_cast<void*>(param.unsafeGetTensorImpl());
    auto it = state_.find(key);
    if (it != state_.end())
    {
        return it->second;
    }

    ParamState s;
    s.step = 0;
    s.exp_avg = torch::zeros_like(param);
    s.exp_avg_sq = torch::zeros_like(param);
    
    auto [inserted_it, inserted] = state_.emplace(key, std::move(s));
    (void)inserted;
    return inserted_it->second;
}