#include "./Embedding.h"

CustomEmbeddingImpl::CustomEmbeddingImpl(const torch::nn::EmbeddingOptions& opt) :
    CustomEmbeddingImpl(CustomEmbeddingOptions(opt.num_embeddings(), opt.embedding_dim()))
{
}

CustomEmbeddingImpl::CustomEmbeddingImpl(const CustomEmbeddingOptions& opt) :
    options(opt)
{
    CustomEmbeddingImpl::reset();
}

void CustomEmbeddingImpl::reset()
{
    if (options.padding_idx().has_value()) 
    {
        if (options.padding_idx() > 0) 
        {
            TORCH_CHECK(
                options.padding_idx() < options.num_embeddings(),
                "Padding_idx must be within num_embeddings");
        }
        else if (options.padding_idx() < 0) 
        {
            TORCH_CHECK(
                options.padding_idx() >= -options.num_embeddings(),
                "Padding_idx must be within num_embedding");
            options.padding_idx(
                // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
                options.num_embeddings() + *options.padding_idx());
        }
    }

    if (!options._weight().defined()) 
    {
        weight = register_parameter(
            "weight",
            torch::empty({ options.num_embeddings(), options.embedding_dim() }));

        if (options.init_params())
        {
            reset_parameters();
        }
    }
    else
    {
        TORCH_CHECK(
            options._weight().sizes() ==
            torch::IntArrayRef(
                { options.num_embeddings(), options.embedding_dim() }),
            "Shape of _weight does not match num_embeddings and embedding_dim");
        weight = register_parameter("weight", options._weight());
    }
}

void CustomEmbeddingImpl::reset_parameters()
{
    torch::nn::init::normal_(weight);

    if (options.padding_idx().has_value()) 
    {
        torch::NoGradGuard no_grad;
        // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
        weight[*options.padding_idx()].fill_(0);
    }
}


void CustomEmbeddingImpl::pretty_print(std::ostream& stream) const
{
    stream << "torch::nn::Embedding(num_embeddings=" << options.num_embeddings()
        << ", embedding_dim=" << options.embedding_dim();
    auto const& padding_idx_opt = options.padding_idx();
    if (padding_idx_opt.has_value()) 
    {
        stream << ", padding_idx=" << padding_idx_opt.value();
    }
    auto const& max_norm_opt = options.max_norm();
    if (max_norm_opt.has_value()) 
    {
        stream << ", max_norm=" << max_norm_opt.value();
    }
    if (options.norm_type() != 2) 
    {
        stream << ", norm_type=" << options.norm_type();
    }
    if (options.scale_grad_by_freq()) 
    {
        stream << ", scale_grad_by_freq=" << std::boolalpha
            << options.scale_grad_by_freq();
    }
    if (options.sparse()) 
    {
        stream << ", sparse=" << std::boolalpha << options.sparse();
    }
    stream << ')';
}

torch::Tensor CustomEmbeddingImpl::forward(const torch::Tensor& input)
{
    return torch::nn::functional::detail::embedding(
        input,
        weight,
        options.padding_idx(),
        options.max_norm(),
        options.norm_type(),
        options.scale_grad_by_freq(),
        options.sparse());
}
