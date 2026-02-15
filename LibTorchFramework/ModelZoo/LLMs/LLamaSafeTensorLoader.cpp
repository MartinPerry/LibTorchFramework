#include "./LLamaSafeTensorLoader.h"

#include "./llama.h"

using namespace ModelZoo::llama;

LoadStateDictReport LLamaSafeTensorLoader::LoadFromHfSafetensors(
	LlamaForCausalLM& model,
	const std::filesystem::path& modelDir,
	bool strict)
{
	TensorMap rawStateDict = LoadSafetensorsSharded(modelDir);
	TensorMap mappedStateDict = MapHfKeysToOurs(rawStateDict, model.GetConfig());
	return LoadMappedStateDict(model, mappedStateDict, strict);
}


TensorMap LLamaSafeTensorLoader::MapHfKeysToOurs(
	const TensorMap& rawStateDict,
	const LlamaConfig& cfg)
{
	TensorMap out;

	if (auto it = rawStateDict.find("model.embed_tokens.weight"); it != rawStateDict.end())
	{
		out["tok_emb.weight"] = it->second;
	}
	else if (auto it = rawStateDict.find("tok_emb.weight"); it != rawStateDict.end())
	{
		out["tok_emb.weight"] = it->second;
	}
	
	if (auto it = rawStateDict.find("model.norm.weight"); it != rawStateDict.end())		
	{
		out["norm.weight"] = it->second;
	}

	if (auto it = rawStateDict.find("lm_head.weight"); it != rawStateDict.end())		
	{
		out["lm_head.weight"] = it->second;
	}

	for (int64_t i = 0; i < cfg.num_hidden_layers; ++i)
	{
		const std::string prefixHf = "model.layers." + std::to_string(i) + ".";
		const std::string prefixOurs = "layers." + std::to_string(i) + ".";

		out[prefixOurs + "attn_norm.weight"] = rawStateDict.at(prefixHf + "input_layernorm.weight");
		out[prefixOurs + "ffn_norm.weight"] = rawStateDict.at(prefixHf + "post_attention_layernorm.weight");

		out[prefixOurs + "attn.q_proj.weight"] = rawStateDict.at(prefixHf + "self_attn.q_proj.weight");
		out[prefixOurs + "attn.k_proj.weight"] = rawStateDict.at(prefixHf + "self_attn.k_proj.weight");
		out[prefixOurs + "attn.v_proj.weight"] = rawStateDict.at(prefixHf + "self_attn.v_proj.weight");
		out[prefixOurs + "attn.o_proj.weight"] = rawStateDict.at(prefixHf + "self_attn.o_proj.weight");

		out[prefixOurs + "mlp.gate_proj.weight"] = rawStateDict.at(prefixHf + "mlp.gate_proj.weight");
		out[prefixOurs + "mlp.up_proj.weight"] = rawStateDict.at(prefixHf + "mlp.up_proj.weight");
		out[prefixOurs + "mlp.down_proj.weight"] = rawStateDict.at(prefixHf + "mlp.down_proj.weight");
	}

	return out;
}