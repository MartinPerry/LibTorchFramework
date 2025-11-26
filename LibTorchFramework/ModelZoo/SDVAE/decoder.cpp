#include "./decoder.h"

using namespace ModelZoo::sdvae;

//
// VAE_AttentionBlockImpl
//
VAE_AttentionBlockImpl::VAE_AttentionBlockImpl(int64_t channels)
{
    // groupnorm with 32 groups
    groupnorm = register_module("groupnorm", torch::nn::GroupNorm(torch::nn::GroupNormOptions(32, channels)));

    // SelfAttention expects (n_heads, d_embed)
    attention = register_module("attention", SelfAttention(1, channels));
}

torch::Tensor VAE_AttentionBlockImpl::forward(torch::Tensor x)
{
    // x: (Batch_Size, Features, Height, Width)
    torch::Tensor residue = x;

    // Apply GroupNorm
    x = groupnorm->forward(x);

    // Get shape
    auto sizes = x.sizes();
    int64_t n = sizes[0];
    int64_t c = sizes[1];
    int64_t h = sizes[2];
    int64_t w = sizes[3];

    // Reshape: (n, c, h*w)
    x = x.view({ n, c, h * w });

    // Transpose to (n, h*w, c)
    x = x.transpose(-1, -2);

    // Self-attention expects (Batch, SeqLen, Dim)
    x = attention->forward(x, /*causal_mask=*/false);

    // Back to (n, c, h*w)
    x = x.transpose(-1, -2);

    // Back to (n, c, h, w)
    x = x.view({ n, c, h, w });

    // Residual connection
    x = x + residue;

    return x;
}

//
// VAE_ResidualBlockImpl
//
VAE_ResidualBlockImpl::VAE_ResidualBlockImpl(int64_t in_channels, int64_t out_channels)
{
    groupnorm_1 = register_module("groupnorm_1", torch::nn::GroupNorm(torch::nn::GroupNormOptions(32, in_channels)));
    conv_1 = register_module("conv_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, /*kernel_size=*/3).padding(1)));

    groupnorm_2 = register_module("groupnorm_2", torch::nn::GroupNorm(torch::nn::GroupNormOptions(32, out_channels)));
    conv_2 = register_module("conv_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels, out_channels, /*kernel_size=*/3).padding(1)));

    if (in_channels == out_channels)
    {
        // Use Identity module to preserve residual when channels match
        residual_layer = register_module("residual_layer", torch::nn::Identity());
    }
    else
    {
        // Use 1x1 conv to match channels
        residual_layer = register_module("residual_layer", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, /*kernel_size=*/1)));
    }
}

torch::Tensor VAE_ResidualBlockImpl::forward(torch::Tensor x)
{
    // x: (Batch_Size, In_Channels, Height, Width)
    torch::Tensor residue = x;

    x = groupnorm_1->forward(x);
    x = torch::nn::functional::silu(x);

    x = conv_1->forward(x);

    x = groupnorm_2->forward(x);
    x = torch::nn::functional::silu(x);

    x = conv_2->forward(x);

    torch::Tensor res_out = residual_layer.forward(residue);
   
    x = x + res_out;

    return x;
}

//
// VAE_DecoderImpl
//
VAE_DecoderImpl::VAE_DecoderImpl()
{
    seq = torch::nn::Sequential(

        // (Batch_Size, 4, H/8, W/8) -> (Batch_Size, 4, H/8, W/8)
        torch::nn::Conv2d(torch::nn::Conv2dOptions(4, 4, /*kernel_size=*/1).padding(0)),

        // (Batch_Size, 4, H/8, W/8) -> (Batch_Size, 512, H/8, W/8)
        torch::nn::Conv2d(torch::nn::Conv2dOptions(4, 512, /*kernel_size=*/3).padding(1)),

        // Residual + Attention + many blocks...
        VAE_ResidualBlock(512, 512),
        VAE_AttentionBlock(512),
        VAE_ResidualBlock(512, 512),
        VAE_ResidualBlock(512, 512),
        VAE_ResidualBlock(512, 512),
        VAE_ResidualBlock(512, 512),

        // Upsample x2 (H/8 -> H/4)
        torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>{2.0, 2.0}).mode(torch::kNearest)),

        torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, /*kernel_size=*/3).padding(1)),

        VAE_ResidualBlock(512, 512),
        VAE_ResidualBlock(512, 512),
        VAE_ResidualBlock(512, 512),

        // Upsample x2 (H/4 -> H/2)
        torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>{2.0, 2.0}).mode(torch::kNearest)),

        torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, /*kernel_size=*/3).padding(1)),

        // Reduce channels 512 -> 256
        VAE_ResidualBlock(512, 256),
        VAE_ResidualBlock(256, 256),
        VAE_ResidualBlock(256, 256),

        // Upsample x2 (H/2 -> H)
        torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>{2.0, 2.0}).mode(torch::kNearest)),

        torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, /*kernel_size=*/3).padding(1)),

        VAE_ResidualBlock(256, 128),
        VAE_ResidualBlock(128, 128),
        VAE_ResidualBlock(128, 128),

        // Final normalization + activation + conv to 3 channels
        torch::nn::GroupNorm(torch::nn::GroupNormOptions(32, 128)),
        torch::nn::SiLU(),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 3, /*kernel_size=*/3).padding(1))
    );


    seq = register_module("seq", seq);
}


torch::Tensor VAE_DecoderImpl::forward(torch::Tensor x)
{
    // x: (Batch_Size, 4, Height / 8, Width / 8)
    // Remove the scaling added by the Encoder.
    x = x / 0.18215;

    // Forward through Sequential
    x = seq->forward(x);

    // (Batch_Size, 3, Height, Width)
    return x;
}