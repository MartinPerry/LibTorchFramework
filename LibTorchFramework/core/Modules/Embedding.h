#ifndef CUSTOM_EMBEDDING_MODULE_H
#define CUSTOM_EMBEDDING_MODULE_H

#include <torch/torch.h>

struct CustomEmbeddingOptions
{
    CustomEmbeddingOptions(int64_t num_embeddings, int64_t embedding_dim) :
        num_embeddings_(num_embeddings),
        embedding_dim_(embedding_dim)
    {
    }

    /// The size of the dictionary of embeddings.
    TORCH_ARG(int64_t, num_embeddings);
    /// The size of each embedding vector.
    TORCH_ARG(int64_t, embedding_dim);
    /// If specified, the entries at `padding_idx` do not contribute to the
    /// gradient; therefore, the embedding vector at `padding_idx` is not updated
    /// during training, i.e. it remains as a fixed "pad". For a newly constructed
    /// Embedding, the embedding vector at `padding_idx` will default to all
    /// zeros, but can be updated to another value to be used as the padding
    /// vector.
    TORCH_ARG(std::optional<int64_t>, padding_idx) = std::nullopt;
    /// If given, each embedding vector with norm larger than `max_norm` is
    /// renormalized to have norm `max_norm`.
    TORCH_ARG(std::optional<double>, max_norm) = std::nullopt;
    /// The p of the p-norm to compute for the `max_norm` option. Default ``2``.
    TORCH_ARG(double, norm_type) = 2.;
    /// If given, this will scale gradients by the inverse of frequency of the
    /// words in the mini-batch. Default ``false``.
    TORCH_ARG(bool, scale_grad_by_freq) = false;
    /// If ``true``, gradient w.r.t. `weight` matrix will be a sparse tensor.
    TORCH_ARG(bool, sparse) = false;
    /// The learnable weights of the module of shape (num_embeddings,
    /// embedding_dim)
    TORCH_ARG(torch::Tensor, _weight);

    /// If set to false, the layer will not init weights to random during creations. Default: true
    TORCH_ARG(bool, init_params) = true;
};

class CustomEmbeddingImpl : public torch::nn::Cloneable<CustomEmbeddingImpl>
{
public:
    /// The options used to configure this module.
    CustomEmbeddingOptions options;

    /// The learned weight.
    torch::Tensor weight;

    /// The learned bias. If `bias` is false in the `options`, this tensor is
    /// undefined.
    torch::Tensor bias;

    CustomEmbeddingImpl(int64_t in_features, int64_t out_features) :
        CustomEmbeddingImpl(CustomEmbeddingOptions(in_features, out_features))
    {
    }

    CustomEmbeddingImpl(const torch::nn::EmbeddingOptions& opt);

    explicit CustomEmbeddingImpl(const CustomEmbeddingOptions& opt);

    void reset() override;

    void reset_parameters();

    void pretty_print(std::ostream& stream) const override;

    torch::Tensor forward(const torch::Tensor& x);
};

//===========================================================================

/// Options for the `Embedding::from_pretrained` function.
struct TORCH_API CustomEmbeddingFromPretrainedOptions {
    /// If ``true``, the tensor does not get updated in the learning process.
    /// Equivalent to ``embedding.weight.requires_grad_(false)``. Default:
    /// ``true``
    TORCH_ARG(bool, freeze) = true;
    /// If specified, the entries at `padding_idx` do not contribute to the
    /// gradient; therefore, the embedding vector at `padding_idx` is not updated
    /// during training, i.e. it remains as a fixed "pad".
    TORCH_ARG(std::optional<int64_t>, padding_idx) = std::nullopt;
    /// If given, each embedding vector with norm larger than `max_norm` is
    /// renormalized to have norm `max_norm`.
    TORCH_ARG(std::optional<double>, max_norm) = std::nullopt;
    /// The p of the p-norm to compute for the `max_norm` option. Default ``2``.
    TORCH_ARG(double, norm_type) = 2.;
    /// If given, this will scale gradients by the inverse of frequency of the
    /// words in the mini-batch. Default ``false``.
    TORCH_ARG(bool, scale_grad_by_freq) = false;
    /// If ``true``, gradient w.r.t. `weight` matrix will be a sparse tensor.
    TORCH_ARG(bool, sparse) = false;
};


/// A `ModuleHolder` subclass for `EmbeddingImpl`.
/// See the documentation for `EmbeddingImpl` class to learn what methods it
/// provides, and examples of how to use `Embedding` with
/// `torch::nn::EmbeddingOptions`. See the documentation for `ModuleHolder` to
/// learn about PyTorch's module storage semantics.
class CustomEmbedding : public torch::nn::ModuleHolder<CustomEmbeddingImpl> {
public:
    using torch::nn::ModuleHolder<CustomEmbeddingImpl>::ModuleHolder;

    /// See the documentation for `torch::nn::EmbeddingFromPretrainedOptions`
    /// class to learn what optional arguments are supported for this function.
    static CustomEmbedding from_pretrained(
        const torch::Tensor& embeddings,
        const CustomEmbeddingFromPretrainedOptions& options = {}) {
        TORCH_CHECK(
            embeddings.dim() == 2,
            "Embeddings parameter is expected to be 2-dimensional");

        auto rows = embeddings.size(0);
        auto cols = embeddings.size(1);

        CustomEmbedding embedding(CustomEmbeddingOptions(rows, cols)
            ._weight(embeddings)
            .padding_idx(options.padding_idx())
            .max_norm(options.max_norm())
            .norm_type(options.norm_type())
            .scale_grad_by_freq(options.scale_grad_by_freq())
            .sparse(options.sparse()));
        embedding->weight.set_requires_grad(!options.freeze());
        return embedding;
    }
};

#endif

