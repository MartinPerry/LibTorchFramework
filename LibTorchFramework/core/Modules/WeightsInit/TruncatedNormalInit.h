#ifndef TRUNCATED_NORMAL_INIT_H
#define TRUNCATED_NORMAL_INIT_H

#include <torch/torch.h>
#include <cmath>
#include <memory>

class TruncatedNormalInit
{
public:
   
    static void trunc_normal(torch::Tensor tensor, double mean = 0.0, double std = 1.0, double a = -2.0, double b = 2.0);
    static void weights_init(torch::nn::Module& m, double mean, double std, double a, double b);
    
    explicit TruncatedNormalInit(torch::nn::Module& model, double mean = 0.0, double std = 1.0, double a = -2.0, double b = 2.0);

private:
    static double norm_cdf(double x);
};


#endif