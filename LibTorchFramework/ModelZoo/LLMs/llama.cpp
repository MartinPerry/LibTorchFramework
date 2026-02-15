#include "./llama.h"

#include <cmath>
#include <cstdint>
#include <limits>

#include <FileUtils/Reading/TextFileReader.h>
#include <Utils/cJSON.h>

#include "../../InputProcessing/DataLoaderData.h"

#include "../../Utils/TorchUtils.h"

using namespace ModelZoo::llama;


using torch::indexing::Slice;

//========================================================================

bool TryReadInt64(cJSON* root, const char* key, int64_t& outValue)
{
	cJSON* node = cJSON_GetObjectItemCaseSensitive(root, key);
	if (!cJSON_IsNumber(node))
	{
		return false;
	}
	outValue = static_cast<int64_t>(node->valuedouble);
	return true;
}

bool TryReadDouble(cJSON* root, const char* key, double& outValue)
{
	cJSON* node = cJSON_GetObjectItemCaseSensitive(root, key);
	if (!cJSON_IsNumber(node))
	{
		return false;
	}
	outValue = node->valuedouble;
	return true;
}

bool TryReadBool(cJSON* root, const char* key, bool& outValue)
{
	cJSON* node = cJSON_GetObjectItemCaseSensitive(root, key);
	if (!cJSON_IsBool(node))
	{
		return false;
	}
	outValue = cJSON_IsTrue(node);
	return true;
}

LlamaConfig LlamaConfig::FromJsonString(const std::string& jsonText)
{
	cJSON* root = cJSON_Parse(jsonText.c_str());
	if (root == nullptr)
	{
		throw std::runtime_error("Failed to parse LlamaConfig JSON");
	}

	LlamaConfig cfg;
	try
	{
		int64_t intValue = 0;
		double doubleValue = 0.0;
		bool boolValue = false;

		if (TryReadInt64(root, "vocab_size", intValue))
		{
			cfg.vocab_size = intValue;
		}
		if (TryReadInt64(root, "hidden_size", intValue))
		{
			cfg.hidden_size = intValue;
		}
		if (TryReadInt64(root, "num_hidden_layers", intValue))
		{
			cfg.num_hidden_layers = intValue;
		}
		if (TryReadInt64(root, "num_attention_heads", intValue))
		{
			cfg.num_attention_heads = intValue;
		}
		if (TryReadInt64(root, "num_key_value_heads", intValue))
		{
			cfg.num_key_value_heads = intValue;
		}
		if (TryReadInt64(root, "intermediate_size", intValue))
		{
			cfg.intermediate_size = intValue;
		}
		if (TryReadDouble(root, "rms_norm_eps", doubleValue))
		{
			cfg.rms_norm_eps = doubleValue;
		}
		if (TryReadDouble(root, "rope_theta", doubleValue))
		{
			cfg.rope_theta = doubleValue;
		}
		if (TryReadBool(root, "tie_word_embeddings", boolValue))
		{
			cfg.tie_word_embeddings = boolValue;
		}
	}
	catch (...)
	{
		cJSON_Delete(root);
		throw;
	}

	cJSON_Delete(root);
	return cfg;
}

LlamaConfig LlamaConfig::FromJsonFile(const std::string& filePath)
{
	TextFileReader tf(filePath.c_str());
	auto data = tf.GetText();
	tf.Close();

	return FromJsonString(data.c_str());
}

//========================================================================

// ---- Layers ----
RMSNormImpl::RMSNormImpl(int64_t dim, double eps) : 
	eps(eps) 
{
	AUTO_REGISTER_NEW_PARAMETER(weight, torch::ones({ dim }));
}

torch::Tensor RMSNormImpl::forward(const torch::Tensor& x) 
{
	auto norm = x.pow(2).mean(-1, true);
	auto y = x * torch::rsqrt(norm + eps);
	return y * weight;
}

//========================================================================

MLPImpl::MLPImpl(int64_t dim, int64_t hidden_dim) 
{
	AUTO_REGISTER_NEW_MODULE(gate_proj, torch::nn::Linear(torch::nn::LinearOptions(dim, hidden_dim).bias(false)));
	AUTO_REGISTER_NEW_MODULE(up_proj, torch::nn::Linear(torch::nn::LinearOptions(dim, hidden_dim).bias(false)));
	AUTO_REGISTER_NEW_MODULE(down_proj, torch::nn::Linear(torch::nn::LinearOptions(hidden_dim, dim).bias(false)));
}

torch::Tensor MLPImpl::forward(const torch::Tensor& x) 
{
	return down_proj(torch::silu(gate_proj(x)) * up_proj(x));
}

//========================================================================

AttentionImpl::AttentionImpl(int64_t dim, int64_t n_heads, std::optional<int64_t> n_kv_heads_opt) :
	dim(dim),
	n_heads(n_heads),
	n_kv_heads(n_kv_heads_opt.has_value() ? n_kv_heads_opt.value() : n_heads),
	head_dim(dim / n_heads) 
{
	TORCH_CHECK(dim % n_heads == 0, "dim must be divisible by n_heads");

	AUTO_REGISTER_NEW_MODULE(q_proj, torch::nn::Linear(torch::nn::LinearOptions(dim, n_heads * head_dim).bias(false)));
	AUTO_REGISTER_NEW_MODULE(k_proj, torch::nn::Linear(torch::nn::LinearOptions(dim, n_kv_heads * head_dim).bias(false)));
	AUTO_REGISTER_NEW_MODULE(v_proj, torch::nn::Linear(torch::nn::LinearOptions(dim, n_kv_heads * head_dim).bias(false)));
	AUTO_REGISTER_NEW_MODULE(o_proj, torch::nn::Linear(torch::nn::LinearOptions(n_heads * head_dim, dim).bias(false)));
}

torch::Tensor AttentionImpl::apply_rope(const torch::Tensor& x, 
	const torch::Tensor& cos,
	const torch::Tensor& sin)
{
	// x: (B, T, H, D), cos/sin: (T, D/2)
	auto B = x.size(0);
	auto T = x.size(1);
	auto H = x.size(2);
	auto D = x.size(3);
	
	auto x1 = x.index({ Slice(), Slice(), Slice(), Slice(0, torch::indexing::None, 2) });
	auto x2 = x.index({ Slice(), Slice(), Slice(), Slice(1, torch::indexing::None, 2) });

	auto cos_t = cos.index({ Slice(0, T) }).unsqueeze(0).unsqueeze(2);  // (1, T, 1, D/2)
	auto sin_t = sin.index({ Slice(0, T) }).unsqueeze(0).unsqueeze(2);

	auto y1 = x1 * cos_t - x2 * sin_t;
	auto y2 = x1 * sin_t + x2 * cos_t;

	auto y = torch::empty_like(x);
	y.index_put_({ Slice(), Slice(), Slice(), Slice(0, torch::indexing::None, 2) }, y1);
	y.index_put_({ Slice(), Slice(), Slice(), Slice(1, torch::indexing::None, 2) }, y2);
	return y;
}

torch::Tensor AttentionImpl::forward(const torch::Tensor& x, 
	const torch::Tensor& cos, 
	const torch::Tensor& sin,
	const torch::Tensor& attn_mask) 
{
	auto B = x.size(0);
	auto T = x.size(1);

	auto q = q_proj(x).view({ B, T, n_heads, head_dim });
	auto k = k_proj(x).view({ B, T, n_kv_heads, head_dim });
	auto v = v_proj(x).view({ B, T, n_kv_heads, head_dim });

	q = apply_rope(q, cos, sin);
	k = apply_rope(k, cos, sin);

	if (n_kv_heads != n_heads) 
	{
		auto repeat = n_heads / n_kv_heads;
		k = k.repeat_interleave(repeat, 2);
		v = v.repeat_interleave(repeat, 2);
	}

	q = q.transpose(1, 2);  // (B, H, T, D)
	k = k.transpose(1, 2);
	v = v.transpose(1, 2);

	auto att = torch::matmul(q, k.transpose(-2, -1)) / std::sqrt(static_cast<double>(head_dim));
	att = att + attn_mask;
	att = torch::softmax(att, -1);
	auto out = torch::matmul(att, v);

	out = out.transpose(1, 2).contiguous().view({ B, T, n_heads * head_dim });
	return o_proj(out);
}

//========================================================================

BlockImpl::BlockImpl(int64_t dim, int64_t n_heads, int64_t hidden_dim, 
	std::optional<int64_t> n_kv_heads,
	double rms_eps) 
{
	AUTO_REGISTER_NEW_MODULE(attn_norm, RMSNorm(dim, rms_eps));
	AUTO_REGISTER_NEW_MODULE(ffn_norm, RMSNorm(dim, rms_eps));
	AUTO_REGISTER_NEW_MODULE(attn, Attention(dim, n_heads, n_kv_heads));
	AUTO_REGISTER_NEW_MODULE(mlp, MLP(dim, hidden_dim));
}

torch::Tensor BlockImpl::forward(const torch::Tensor& x, const torch::Tensor& cos, const torch::Tensor& sin,
	const torch::Tensor& attn_mask, bool use_ckpt) 
{
	(void)use_ckpt;
	auto h = x + attn(attn_norm(x), cos, sin, attn_mask);
	h = h + mlp(ffn_norm(h));
	return h;
}

//========================================================================

LlamaForCausalLM::LlamaForCausalLM(const LlamaConfig& cfg) : 
	cfg(cfg) 
{
	return;
	vocab_size = cfg.vocab_size;
	dim = cfg.hidden_size;
	n_layers = cfg.num_hidden_layers;
	n_heads = cfg.num_attention_heads;
	n_kv_heads = cfg.num_key_value_heads.has_value() ? cfg.num_key_value_heads.value() : n_heads;
	hidden_dim = cfg.intermediate_size.has_value() ? cfg.intermediate_size.value() : 4 * dim;
	rms_eps = cfg.rms_norm_eps;

	AUTO_REGISTER_NEW_MODULE(tok_emb, torch::nn::Embedding(vocab_size, dim));
	
	AUTO_REGISTER_NEW_MODULE(layers, torch::nn::ModuleList());
	for (int64_t i = 0; i < n_layers; ++i) 
	{
		layers->push_back(Block(dim, n_heads, hidden_dim, n_kv_heads, rms_eps));
	}

	AUTO_REGISTER_NEW_MODULE(norm, RMSNorm(dim, rms_eps));

	AUTO_REGISTER_NEW_MODULE(lm_head, torch::nn::Linear(torch::nn::LinearOptions(dim, vocab_size).bias(false)));

	if (cfg.tie_word_embeddings) 
	{
		lm_head->weight = tok_emb->weight;
	}

	AUTO_REGISTER_NEW_BUFFER(_attn_mask_cache, torch::empty({ 0 }));
	AUTO_REGISTER_NEW_BUFFER(_rope_cos, torch::empty({ 0 }));
	AUTO_REGISTER_NEW_BUFFER(_rope_sin, torch::empty({ 0 }));
}

const char* LlamaForCausalLM::GetName() const
{
	return "LlamaForCausalLM";
}

const LlamaConfig& LlamaForCausalLM::GetConfig() const
{
	return this->cfg;
}

torch::Tensor LlamaForCausalLM::get_attn_mask(int64_t T, const torch::Device& device, 
	torch::ScalarType dtype) 
{
	if (_mask_len < T || _attn_mask_cache.device() != device) 
	{
		auto m = torch::full({ T, T }, -std::numeric_limits<float>::infinity(),
			torch::TensorOptions().device(device).dtype(dtype));
		m = torch::triu(m, 1);
		_attn_mask_cache = m.view({ 1, 1, T, T });
		_mask_len = T;
	}
	return _attn_mask_cache.index({ Slice(), Slice(), Slice(0, T), Slice(0, T) });
}


std::pair<torch::Tensor, torch::Tensor> LlamaForCausalLM::precompute_rope_frequencies(
	int64_t dim,
	int64_t max_seq_len,
	double base,
	torch::Device device,
	std::optional<torch::ScalarType> dtype)
{
	auto inv_freq = 1.0 / torch::pow(
		torch::tensor(base, torch::TensorOptions().device(device)),
		torch::arange(0, dim, 2, torch::TensorOptions().device(device).dtype(torch::kFloat)) / static_cast<double>(dim));
	auto t = torch::arange(max_seq_len, torch::TensorOptions().device(device).dtype(torch::kFloat));
	auto freqs = torch::outer(t, inv_freq);
	auto cos = torch::cos(freqs);
	auto sin = torch::sin(freqs);
	if (dtype.has_value())
	{
		cos = cos.to(dtype.value());
		sin = sin.to(dtype.value());
	}
	return { cos, sin };
}


std::pair<torch::Tensor, torch::Tensor> LlamaForCausalLM::get_rope(int64_t T, 
	const torch::Device& device,
	torch::ScalarType dtype) 
{
	if (_rope_len < T || _rope_cos.device() != device) 
	{
		auto head_dim = dim / n_heads;
		auto rope = precompute_rope_frequencies(head_dim, T, cfg.rope_theta, device, dtype);
		_rope_cos = rope.first;
		_rope_sin = rope.second;
		_rope_len = T;
	}

	return { _rope_cos.index({Slice(0, T)}), _rope_sin.index({Slice(0, T)}) };
}

torch::Tensor LlamaForCausalLM::forward(const torch::Tensor& input_ids, bool use_ckpt) 
{
	(void)use_ckpt;
	auto device = input_ids.device();
	auto T = input_ids.size(1);

	auto x = tok_emb(input_ids);
	auto attn_mask = get_attn_mask(T, device, x.scalar_type());
	auto rope = get_rope(T, device, x.scalar_type());

	for (const auto& layer : *layers) 
	{
		x = layer->as<Block>()->forward(x, rope.first, rope.second, attn_mask, use_ckpt);
	}

	x = norm(x);
	auto logits = lm_head(x);
	return logits;
}

std::vector<torch::Tensor> LlamaForCausalLM::RunForward(DataLoaderData& batch)
{
	//input size must be w >= 256 and h >= 256

	auto x = this->forward(batch.input);

	return { x, batch.target };
}

