#include "./MLP.h"

MlpImpl::MlpImpl(
    int64_t inFeatures,
    std::optional<int64_t> hiddenFeatures,
    std::optional<int64_t> outFeatures,
    double dropProb)
{
    const int64_t hidden = hiddenFeatures.has_value() ? *hiddenFeatures : inFeatures;
    const int64_t output = outFeatures.has_value() ? *outFeatures : inFeatures;

    fc1 = register_module(
        "fc1",
        torch::nn::Linear(inFeatures, hidden));

    act = register_module(
        "act",
        torch::nn::GELU());

    fc2 = register_module(
        "fc2",
        torch::nn::Linear(hidden, output));

    drop = register_module(
        "drop",
        torch::nn::Dropout(torch::nn::DropoutOptions(dropProb)));
}

torch::Tensor MlpImpl::forward(torch::Tensor x)
{
    x = fc1->forward(x);
    x = act->forward(x);
    x = drop->forward(x);
    x = fc2->forward(x);
    x = drop->forward(x);

    return x;
}