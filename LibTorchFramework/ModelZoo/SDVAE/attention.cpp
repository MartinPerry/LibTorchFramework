#include "attention.h"

using namespace ModelZoo::sdvae;

SelfAttentionImpl::SelfAttentionImpl(int64_t n_heads_, int64_t d_embed, bool in_proj_bias, bool out_proj_bias)
    : n_heads(n_heads_)
{
    // Basic validation
    if (d_embed % n_heads != 0)
    {
        // For now, throw. Could also floor/divide and warn in future.
        throw std::runtime_error("d_embed must be divisible by n_heads");
    }

    d_head = d_embed / n_heads;

    in_proj = register_module("in_proj", torch::nn::Linear(torch::nn::LinearOptions(d_embed, 3 * d_embed).bias(in_proj_bias)));
    out_proj = register_module("out_proj", torch::nn::Linear(torch::nn::LinearOptions(d_embed, d_embed).bias(out_proj_bias)));
}

torch::Tensor SelfAttentionImpl::forward(torch::Tensor x, bool causal_mask)
{
    // Expect x shape: (batch, seq_len, dim)
    auto input_shape = x.sizes();
    int64_t batch_size = input_shape[0];
    int64_t sequence_length = input_shape[1];
    int64_t d_embed = input_shape[2];

    // in_proj -> 3 tensors along last dim
    torch::Tensor qkv = in_proj->forward(x);
    std::vector<torch::Tensor> parts = qkv.chunk(3, -1);
    torch::Tensor q = parts[0];
    torch::Tensor k = parts[1];
    torch::Tensor v = parts[2];

    // reshape into (batch, heads, seq_len, d_head)
    q = q.view({ batch_size, sequence_length, n_heads, d_head }).transpose(1, 2);
    k = k.view({ batch_size, sequence_length, n_heads, d_head }).transpose(1, 2);
    v = v.view({ batch_size, sequence_length, n_heads, d_head }).transpose(1, 2);

    // weight = q @ k^T -> (batch, heads, seq_len, seq_len)
    torch::Tensor weight = torch::matmul(q, k.transpose(-1, -2));

    if (causal_mask)
    {
        // create upper-triangular mask (1s above diagonal)
        {
            torch::Tensor tri = torch::triu(torch::ones({ sequence_length, sequence_length }, torch::kBool), 1);
            tri = tri.to(weight.device());
            tri = tri.unsqueeze(0).unsqueeze(0); // (1,1,seq,seq) to broadcast
            // masked_fill_ with -inf where mask==1
            weight.masked_fill_(tri, -std::numeric_limits<float>::infinity());
        }
    }

    // scale
    weight = weight / std::sqrt(static_cast<double>(d_head));

    // softmax
    weight = torch::softmax(weight, -1);

    // output = weight @ v -> (batch, heads, seq_len, d_head)
    torch::Tensor output = torch::matmul(weight, v);

    // transpose & reshape back to (batch, seq_len, dim)
    output = output.transpose(1, 2).contiguous();
    output = output.view({ batch_size, sequence_length, d_embed });

    // final linear
    output = out_proj->forward(output);

    return output;
}


// -------------------- CrossAttention --------------------

CrossAttentionImpl::CrossAttentionImpl(int64_t n_heads_, int64_t d_embed, int64_t d_cross, bool in_proj_bias, bool out_proj_bias)
    : n_heads(n_heads_)
{
    if (d_embed % n_heads != 0)
    {
        throw std::runtime_error("d_embed must be divisible by n_heads");
    }

    d_head = d_embed / n_heads;

    q_proj = register_module("q_proj", torch::nn::Linear(torch::nn::LinearOptions(d_embed, d_embed).bias(in_proj_bias)));
    k_proj = register_module("k_proj", torch::nn::Linear(torch::nn::LinearOptions(d_cross, d_embed).bias(in_proj_bias)));
    v_proj = register_module("v_proj", torch::nn::Linear(torch::nn::LinearOptions(d_cross, d_embed).bias(in_proj_bias)));
    out_proj = register_module("out_proj", torch::nn::Linear(torch::nn::LinearOptions(d_embed, d_embed).bias(out_proj_bias)));
}

torch::Tensor CrossAttentionImpl::forward(torch::Tensor x, torch::Tensor y)
{
    // x: (batch, seq_len_q, dim_q == d_embed)
    // y: (batch, seq_len_kv, dim_kv == d_cross)
    auto x_shape = x.sizes();
    int64_t batch_size = x_shape[0];
    int64_t seq_len_q = x_shape[1];
    int64_t d_embed = x_shape[2];

    // project
    torch::Tensor q = q_proj->forward(x); // (batch, seq_q, d_embed)
    torch::Tensor k = k_proj->forward(y); // (batch, seq_kv, d_embed)
    torch::Tensor v = v_proj->forward(y); // (batch, seq_kv, d_embed)

    // reshape into (batch, heads, seq, d_head)
    q = q.view({ batch_size, seq_len_q, n_heads, d_head }).transpose(1, 2);
    int64_t seq_len_kv = y.size(1);
    k = k.view({ batch_size, seq_len_kv, n_heads, d_head }).transpose(1, 2);
    v = v.view({ batch_size, seq_len_kv, n_heads, d_head }).transpose(1, 2);

    // attention weights (batch, heads, seq_q, seq_kv)
    torch::Tensor weight = torch::matmul(q, k.transpose(-1, -2));

    // scale
    weight = weight / std::sqrt(static_cast<double>(d_head));

    // softmax
    weight = torch::softmax(weight, -1);

    // weighted sum
    torch::Tensor output = torch::matmul(weight, v); // (batch, heads, seq_q, d_head)

    // back to (batch, seq_q, d_embed)
    output = output.transpose(1, 2).contiguous();
    output = output.view({ batch_size, seq_len_q, d_embed });

    // final linear
    output = out_proj->forward(output);

    return output;
}