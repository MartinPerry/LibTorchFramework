#include "./optimizers_tests.h"


//
// Runs three checks:
//  1) Exact match vs torch::optim::AdamW step-by-step when quantization disabled
//  2) Loss decreases on a tiny regression toy problem
//  3) Numerical sanity: no NaN/Inf, finite state
//
// Assumes your MyOptimizer API matches torch::optim::Optimizer (step(closure), options with lr/betas/eps/weight_decay,
// amsgrad, block_size, min_quantized_numel, bf16_stochastic_round)

#include <iostream>
#include <iomanip>
#include <random>

#include <torch/torch.h>

#include "../../core/Optimizers/AdamW8bit.h"
#include "../../core/Optimizers/FusedAdamW8bit.h"

namespace CustomScenarios::_tests_
{

    // ---------- Helpers ----------
    static void assert_allclose(const torch::Tensor& a, const torch::Tensor& b, double rtol, double atol, const char* what) 
    {
        auto diff = (a - b).abs();
        auto max_abs = diff.max().item<double>();
        auto max_ref = b.abs().max().item<double>();
        auto tol = atol + rtol * max_ref;
        if (!(max_abs <= tol) || !std::isfinite(max_abs)) 
        {
            std::cerr << "\n[FAIL] " << what << " allclose failed:\n"
                << "  max_abs_diff=" << std::setprecision(16) << max_abs << "\n"
                << "  tol=" << tol << " (rtol=" << rtol << ", atol=" << atol << ")\n";
            std::exit(1);
        }
    }

    static void assert_finite(const torch::Tensor& t, const char* what) 
    {
        if (!t.defined()) return;
        auto ok = torch::isfinite(t).all().item<bool>();
        if (!ok) {
            std::cerr << "\n[FAIL] " << what << " contains NaN/Inf\n";
            std::exit(1);
        }
    }

    static torch::Device pick_device() 
    {
        if (torch::cuda::is_available()) return torch::kCUDA;
        return torch::kCPU;
    }

    // Deterministic init for reproducibility
    static void seed_all(uint64_t seed) 
    {
        torch::manual_seed(seed);
        if (torch::cuda::is_available()) torch::cuda::manual_seed_all(seed);
    }

    // ---------- Tests ----------

    void test_matches_adamw_when_quant_off() 
    {
        std::cout << "[TEST] matches AdamW when quantization disabled...\n";

        auto dev = pick_device();

        seed_all(123);

        // Disable quantization by setting min_quantized_numel huge
        AdamW8bitOptions myopt(1e-3);
        myopt.betas({ 0.9, 0.999 });
        myopt.eps(1e-8);        
        myopt.weight_decay(0.01);
        myopt.amsgrad(false);
        myopt.block_size(256);
        myopt.min_quantized_numel((int64_t)1e18); // disable
        myopt.bf16_stochastic_round(false);

        // Torch AdamW options
        torch::optim::AdamWOptions awopt(myopt.lr());
        awopt.betas(myopt.betas());
        awopt.eps(myopt.eps());
        awopt.weight_decay(myopt.weight_decay());
        awopt.amsgrad(false);

        // Two params, include non-multiple sizes to ensure no quant path triggers
        auto p1a = torch::randn({ 257, 13 }, torch::TensorOptions().device(dev).dtype(torch::kFloat32).requires_grad(true));
        auto p2a = torch::randn({ 19 }, torch::TensorOptions().device(dev).dtype(torch::kFloat32).requires_grad(true));

        auto p1b = p1a.detach().clone().set_requires_grad(true);
        auto p2b = p2a.detach().clone().set_requires_grad(true);

        std::vector<torch::Tensor> paramsA{ p1a, p2a };
        std::vector<torch::Tensor> paramsB{ p1b, p2b };

        AdamW8bit my(paramsA, myopt);
        torch::optim::AdamW aw(paramsB, awopt);

        // Step-by-step: random grads each step; compare after each update.
        for (int step = 0; step < 20; ++step) 
        {
            // generate grads deterministically
            auto g1 = torch::randn_like(p1a);
            auto g2 = torch::randn_like(p2a);

            p1a.mutable_grad() = g1.clone();
            p2a.mutable_grad() = g2.clone();

            p1b.mutable_grad() = g1.clone();
            p2b.mutable_grad() = g2.clone();

            my.step();
            aw.step();

            assert_allclose(p1a, p1b, /*rtol=*/1e-5, /*atol=*/1e-6, "p1");
            assert_allclose(p2a, p2b, /*rtol=*/1e-5, /*atol=*/1e-6, "p2");
            assert_finite(p1a, "p1");
            assert_finite(p2a, "p2");
        }

        std::cout << "  OK\n";
    }

    //========================================================================================

    
    template <typename Opt, typename OptSets>
    void test_loss_decreases_toy_regression(const OptSets& sets)
    {
        std::cout << "[TEST] loss decreases on toy regression...\n";

        auto dev = pick_device();

        seed_all(42);

        const int64_t N = 512;
        const int64_t D = 1024;

        auto X = torch::randn({ N, D }, torch::TensorOptions().device(dev).dtype(torch::kFloat32));
        auto true_w = torch::randn({ D, 1 }, torch::TensorOptions().device(dev).dtype(torch::kFloat32));
        auto y = X.matmul(true_w) + 0.1 * torch::randn({ N, 1 }, X.options());

        auto w = torch::zeros({ D, 1 }, torch::TensorOptions().device(dev).dtype(torch::kFloat32).requires_grad(true));

        /*
        AdamW8bitOptions myopt(5e-2);
        myopt.betas({ 0.9, 0.999 });
        myopt.eps(1e-8);
        myopt.weight_decay(0.0);
        myopt.amsgrad(false);
        myopt.block_size(4);
        myopt.min_quantized_numel(1);
        myopt.bf16_stochastic_round(false);
        */
        Opt my({ w }, sets);
       

        auto loss_fn = [&]() {
            if (w.grad().defined()) {
                w.grad().zero_();
            }
            auto pred = X.matmul(w);
            auto loss = torch::mse_loss(pred, y);
            loss.backward();
            return loss;
            };

        double loss0 = 0.0;
        double loss_last = 0.0;

        for (int i = 0; i < 100; ++i) 
        {            
            auto L = my.step(loss_fn); // uses closure with enable_grad internally
            auto lv = L.item<double>();
            if (i == 0) loss0 = lv;
            loss_last = lv;

            assert_finite(w, "w");
            assert_finite(w.grad(), "w.grad");
        }

        if (!(loss_last < loss0 * 0.2)) 
        { 
            std::cerr << "\n[FAIL] loss did not decrease enough: loss0=" << loss0 << " last=" << loss_last << "\n";
            std::exit(1);
        }

        std::cout << "  OK (loss0=" << loss0 << " last=" << loss_last << ")\n";
    }
    
    void test_loss_decreases_toy_regression_adamw8()
    {
        AdamW8bitOptions myopt(5e-2);
        myopt.betas({ 0.9, 0.999 });
        myopt.eps(1e-8);
        myopt.weight_decay(0.0);
        myopt.amsgrad(false);
        myopt.block_size(4);
        myopt.min_quantized_numel(1);
        myopt.bf16_stochastic_round(false);

        test_loss_decreases_toy_regression<AdamW8bit>(myopt);
    }

    void test_loss_decreases_toy_regression_fused_adamw8()
    {
        FusedAdamW8bitOptions myopt(5e-2);
        myopt.betas({ 0.9, 0.999 });
        myopt.eps(1e-8);
        myopt.weight_decay(0.0);
        myopt.amsgrad(false);
        myopt.block_size(256);
        myopt.min_quantized_numel(1);
        myopt.bf16_stochastic_round(false);

        test_loss_decreases_toy_regression<FusedAdamW8bit>(myopt);
    }


    /*
    static void test_quant_roundtrip_sanity(torch::Device dev)
    {
        std::cout << "[TEST] quant/dequant sanity (finite, reasonable error)...\n";

        seed_all(7);

        AdamW8bitOptions opt;
        opt.lr = 1e-3;
        opt.betas = { 0.9, 0.999 };
        opt.eps = 1e-8;
        opt.weight_decay = 0.0;
        opt.amsgrad = false;
        opt.block_size = 256;
        opt.min_quantized_numel = 1;
        opt.bf16_stochastic_round = false;

        // dummy param to construct optimizer (so qmaps exist)
        auto p = torch::randn({ 1024 }, torch::TensorOptions().device(dev).dtype(torch::kFloat32).requires_grad(true));
        AdamW8bit my({ p }, opt);

        auto x = torch::randn({ 1024 }, p.options().dtype(torch::kFloat32)) * 0.01;
        // call your public helpers if exposed; otherwise skip this test.
        // If not public, comment out.
        auto q = my.quantize_from_fp32(x, my.qmap_signed_cpu().to(dev), opt.block_size); // adjust accessors if needed
        auto y = my.dequant_to_fp32(q);

        assert_finite(y, "dequant");
        auto rel = (x - y).abs().mean().item<double>() / (x.abs().mean().item<double>() + 1e-12);
        if (!(rel < 0.2))
        {
            // loose bound; 8-bit log-ish map
            std::cerr << "\n[FAIL] quant roundtrip too lossy: rel_mean_err=" << rel << "\n";
            std::exit(1);
        }

        std::cout << "  OK (rel_mean_err=" << rel << ")\n";
    }
    */
}