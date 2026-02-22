#include "./llama.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <algorithm>
#include <execution>

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

	int64_t intValue = 0;
	
	TryReadInt64(root, "vocab_size", cfg.vocab_size);
	TryReadInt64(root, "hidden_size", cfg.hidden_size);
	TryReadInt64(root, "num_hidden_layers", cfg.num_hidden_layers);
	TryReadInt64(root, "num_attention_heads", cfg.num_attention_heads);
	
	if (TryReadInt64(root, "num_key_value_heads", intValue))
	{
		cfg.num_key_value_heads = intValue;
	}
	if (TryReadInt64(root, "intermediate_size", intValue))
	{
		cfg.intermediate_size = intValue;
	}

	TryReadDouble(root, "rms_norm_eps", cfg.rms_norm_eps);	
	TryReadDouble(root, "rope_theta", cfg.rope_theta);	
	TryReadBool(root, "tie_word_embeddings", cfg.tie_word_embeddings);
	
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

std::u8string LlamaConfig::InstructPrompt(std::u8string_view userText,
	std::u8string_view systemText)
{
	std::u8string out;
	out.reserve(
		128 + userText.size() + systemText.size()
	);

	out += u8"<|begin_of_text|>";
	out += u8"<|start_header_id|>system<|end_header_id|>\n";
	out += systemText;
	out += u8"\n";
	out += u8"<|eot_id|>";
	out += u8"<|start_header_id|>user<|end_header_id|>\n";
	out += userText;
	out += u8"\n";
	out += u8"<|eot_id|>";
	out += u8"<|start_header_id|>assistant<|end_header_id|>\n";

	return out;
}


//========================================================================
//========================================================================
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
	const torch::Tensor& cos, const torch::Tensor& sin,
	int startPos)
{
	// x: (B, T, H, D), cos/sin: (T, D/2)
	auto B = x.size(0);
	auto T = x.size(1);
	auto H = x.size(2);
	auto D = x.size(3);

	auto x1 = x.index({ Slice(), Slice(), Slice(), Slice(0, torch::indexing::None, 2) });
	auto x2 = x.index({ Slice(), Slice(), Slice(), Slice(1, torch::indexing::None, 2) });

	auto cos_t = cos.index({ Slice(startPos, startPos + T) }).unsqueeze(0).unsqueeze(2);  // (1, T, 1, D/2)
	auto sin_t = sin.index({ Slice(startPos, startPos + T) }).unsqueeze(0).unsqueeze(2);

	auto y1 = x1 * cos_t - x2 * sin_t;
	auto y2 = x1 * sin_t + x2 * cos_t;

	auto y = torch::empty_like(x);
	y.index_put_({ Slice(), Slice(), Slice(), Slice(0, torch::indexing::None, 2) }, y1);
	y.index_put_({ Slice(), Slice(), Slice(), Slice(1, torch::indexing::None, 2) }, y2);
	return y;
}

std::pair<torch::Tensor, std::optional<KVCache>> AttentionImpl::forward(const torch::Tensor& x,
	const torch::Tensor& cos,
	const torch::Tensor& sin,
	const torch::Tensor& attn_mask,
	const std::optional<KVCache>& past_kv,
	bool use_cache,
	int64_t cache_position)
{
	auto B = x.size(0);
	auto T = x.size(1);

	auto q = q_proj(x).view({ B, T, n_heads, head_dim });
	auto k = k_proj(x).view({ B, T, n_kv_heads, head_dim });
	auto v = v_proj(x).view({ B, T, n_kv_heads, head_dim });

	q = apply_rope(q, cos, sin, static_cast<int>(cache_position));
	k = apply_rope(k, cos, sin, static_cast<int>(cache_position));

	q = q.transpose(1, 2);  // (B, H, T, D)
	k = k.transpose(1, 2);  // (B, H_kv, T, D)
	v = v.transpose(1, 2);  // (B, H_kv, T, D)

	if (past_kv.has_value())
	{
		k = torch::cat({ past_kv->k, k }, 2);
		v = torch::cat({ past_kv->v, v }, 2);
	}

	std::optional<KVCache> present_kv = std::nullopt;
	if (use_cache)
	{
		present_kv = KVCache(k, v);
	}

	if (n_kv_heads != n_heads)
	{
		auto repeat = n_heads / n_kv_heads;
		k = k.repeat_interleave(repeat, 1);
		v = v.repeat_interleave(repeat, 1);
	}

	auto att = torch::matmul(q, k.transpose(-2, -1)) / std::sqrt(static_cast<double>(head_dim));
	att = att + attn_mask;
	att = torch::softmax(att, -1);
	auto out = torch::matmul(att, v);

	out = out.transpose(1, 2).contiguous().view({ B, T, n_heads * head_dim });
	return { o_proj(out), present_kv };
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

std::pair<torch::Tensor, std::optional<KVCache>> BlockImpl::forward(const torch::Tensor& x,
	const torch::Tensor& cos, const torch::Tensor& sin,
	const torch::Tensor& attn_mask,
	const std::optional<KVCache>& past_kv,
	bool use_cache,
	int64_t cache_position)
{
	auto normed = attn_norm(x);
	auto attn_out = attn(normed, cos, sin, attn_mask, past_kv, use_cache, cache_position);
	auto h = x + attn_out.first;
	h = h + mlp(ffn_norm(h));
	return { h, attn_out.second };
}

//========================================================================

LlamaForCausalLM::LlamaForCausalLM(const LlamaConfig& cfg) :
	cfg(cfg)
{

	int64_t n_kv_heads = cfg.num_key_value_heads.has_value() ? cfg.num_key_value_heads.value() : cfg.num_attention_heads;
	int64_t hidden_dim = cfg.intermediate_size.has_value() ? cfg.intermediate_size.value() : 4 * cfg.hidden_size;

	AUTO_REGISTER_NEW_MODULE(tok_emb, torch::nn::Embedding(cfg.vocab_size, cfg.hidden_size));

	AUTO_REGISTER_NEW_MODULE(layers, torch::nn::ModuleList());

	std::vector<Block> tmp(cfg.num_hidden_layers, nullptr);
	std::vector<int64_t> indices(cfg.num_hidden_layers);
	std::iota(indices.begin(), indices.end(), 0);

	std::for_each(std::execution::par, indices.begin(), indices.end(), [&](int64_t i) {
		tmp[i] = Block(cfg.hidden_size, cfg.num_attention_heads, hidden_dim, n_kv_heads, cfg.rms_norm_eps);
		});

	for (int64_t i = 0; i < cfg.num_hidden_layers; ++i)
	{
		//layers->push_back(Block(cfg.hidden_size, cfg.num_attention_heads, hidden_dim, n_kv_heads, cfg.rms_norm_eps));
		layers->push_back(tmp[i]);
	}

	AUTO_REGISTER_NEW_MODULE(norm, RMSNorm(cfg.hidden_size, cfg.rms_norm_eps));

	AUTO_REGISTER_NEW_MODULE(lm_head, torch::nn::Linear(torch::nn::LinearOptions(cfg.hidden_size, cfg.vocab_size).bias(false)));

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

torch::Tensor LlamaForCausalLM::get_attn_mask(int64_t q_len, int64_t k_len,
	torch::ScalarType dtype, int64_t past_len)
{
	constexpr float minValue = std::numeric_limits<float>::lowest();

	if (q_len == k_len && past_len == 0)
	{
		if ((_mask_len < q_len) || (_attn_mask_cache.device() != tOptDevice.device()) ||
			(_attn_mask_cache.scalar_type() != dtype))
		{
			auto m = torch::full({ q_len, q_len }, minValue, tOptDevice.dtype(dtype));
			m = torch::triu(m, 1);
			_attn_mask_cache = m.view({ 1, 1, q_len, q_len });
			_mask_len = q_len;
		}
		return _attn_mask_cache.index({ Slice(), Slice(), Slice(0, q_len), Slice(0, q_len) });
	}

	auto q_pos = (past_len + torch::arange(q_len, tOptDevice)).unsqueeze(1);
	auto k_pos = torch::arange(k_len, tOptDevice).unsqueeze(0);
	auto m = torch::zeros({ q_len, k_len }, tOptDevice.dtype(dtype));
	m = m.masked_fill(k_pos > q_pos, minValue);
	return m.view({ 1, 1, q_len, k_len });
}


std::pair<torch::Tensor, torch::Tensor> LlamaForCausalLM::precompute_rope_frequencies(
	int64_t dim,
	int64_t max_seq_len,
	double base,
	torch::ScalarType dtype)
{
	auto inv_freq = 1.0 / torch::pow(
		torch::tensor(base, tOptDevice),
		torch::arange(0, dim, 2, tOptDevice.dtype(torch::kFloat)) / static_cast<double>(dim)
	);
	auto t = torch::arange(max_seq_len, tOptDevice.dtype(torch::kFloat));
	auto freqs = torch::outer(t, inv_freq);
	auto cos = torch::cos(freqs);
	auto sin = torch::sin(freqs);
	cos = cos.to(dtype);
	sin = sin.to(dtype);

	return { cos, sin };
}


std::pair<torch::Tensor, torch::Tensor> LlamaForCausalLM::get_rope(int64_t T,
	torch::ScalarType dtype)
{
	if ((_rope_len < T) || (_rope_cos.device() != tOptDevice.device()) ||
		(_rope_cos.scalar_type() != dtype))
	{
		auto head_dim = cfg.hidden_size / cfg.num_attention_heads;
		auto rope = precompute_rope_frequencies(head_dim, T, cfg.rope_theta, dtype);
		_rope_cos = rope.first;
		_rope_sin = rope.second;
		_rope_len = T;
	}

	return { _rope_cos.index({Slice(0, T)}), _rope_sin.index({Slice(0, T)}) };
}

torch::Tensor LlamaForCausalLM::forward(const torch::Tensor& input_ids)
{
	return forward_with_cache(input_ids, {}, false).first;
}

std::pair<torch::Tensor, std::vector<KVCache>> LlamaForCausalLM::forward_with_cache(
	const torch::Tensor& input_ids,
	const std::vector<KVCache>& past_key_values,
	bool use_cache)
{
	auto device = input_ids.device();
	tOptDevice = torch::TensorOptions().device(device);

	//auto B = input_ids.size(0);	
	auto T = input_ids.size(1);

	int64_t past_len = 0;
	if ((use_cache) && (past_key_values.empty() == false))
	{
		TORCH_CHECK(static_cast<int64_t>(past_key_values.size()) == cfg.num_hidden_layers,
			"past_key_values has ", static_cast<int64_t>(past_key_values.size()),
			" layers, expected ", cfg.num_hidden_layers);

		past_len = past_key_values[0].k.size(2);
	}

	auto x = tok_emb(input_ids);
	auto total_k_len = past_len + T;
	auto attn_mask = get_attn_mask(T, total_k_len, x.scalar_type(), past_len);
	auto rope = get_rope(total_k_len, x.scalar_type());

	std::vector<KVCache> next_past;
	if (use_cache)
	{
		next_past.reserve(static_cast<size_t>(cfg.num_hidden_layers));
	}

	for (int64_t layer_i = 0; layer_i < cfg.num_hidden_layers; ++layer_i)
	{
		std::optional<KVCache> layer_past = std::nullopt;
		if ((use_cache) && (past_key_values.empty() == false))
		{
			layer_past = past_key_values[static_cast<size_t>(layer_i)];
		}

		auto layer = layers[layer_i]->as<Block>();
		auto layer_out = layer->forward(x, rope.first, rope.second, attn_mask, layer_past, use_cache, past_len);
		x = layer_out.first;
		if (use_cache)
		{
			TORCH_CHECK(layer_out.second.has_value(), "Layer cache missing while use_cache=true");
			next_past.push_back(layer_out.second.value());
		}
	}

	x = norm(x);
	auto logits = lm_head(x);
	return { logits, next_past };
}



std::vector<torch::Tensor> LlamaForCausalLM::RunForward(DataLoaderData& batch)
{
	auto x = this->forward(batch.input);

	return { x, batch.target };
}

