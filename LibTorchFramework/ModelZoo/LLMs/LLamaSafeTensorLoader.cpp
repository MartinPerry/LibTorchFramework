#include "./LLamaSafeTensorLoader.h"

#include "./llama.h"

using namespace ModelZoo::llama;

LoadStateDictReport LLamaSafeTensorLoader::LoadFromHfSafetensors(
	LlamaForCausalLM& model,
	const std::filesystem::path& modelDir,
	bool strict)
{
	this->CreateMapping(model.GetConfig());

	TensorMap stateDict = this->LoadSafetensorsSharded(modelDir, [&](const std::string& hfName) -> std::string {
		return this->MappingHfKeysToOurs(hfName);
	});
	//TensorMap mappedStateDict = MapHfKeysToOurs(stateDict, model.GetConfig());
	return FillModelStateDict(model, stateDict, strict);
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

void LLamaSafeTensorLoader::CreateMapping(const LlamaConfig& cfg)
{
	mapping.clear();

	// layers
	for (int64_t i = 0; i < cfg.num_hidden_layers; ++i)
	{
		const std::string prefixHf = "model.layers." + std::to_string(i) + ".";
		const std::string prefixOurs = "layers." + std::to_string(i) + ".";

		mapping[prefixOurs + "attn_norm.weight"] = prefixHf + "input_layernorm.weight";
		mapping[prefixOurs + "ffn_norm.weight"] = prefixHf + "post_attention_layernorm.weight";
		mapping[prefixOurs + "attn.q_proj.weight"] = prefixHf + "self_attn.q_proj.weight";
		mapping[prefixOurs + "attn.k_proj.weight"] = prefixHf + "self_attn.k_proj.weight";
		mapping[prefixOurs + "attn.v_proj.weight"] = prefixHf + "self_attn.v_proj.weight";
		mapping[prefixOurs + "attn.o_proj.weight"] = prefixHf + "self_attn.o_proj.weight";
		mapping[prefixOurs + "mlp.gate_proj.weight"] = prefixHf + "mlp.gate_proj.weight";
		mapping[prefixOurs + "mlp.up_proj.weight"] = prefixHf + "mlp.up_proj.weight";
		mapping[prefixOurs + "mlp.down_proj.weight"] = prefixHf + "mlp.down_proj.weight";
	}

}

std::string LLamaSafeTensorLoader::MappingHfKeysToOurs(const std::string& hfName)
{
	
	// embeddings
	if (hfName == "model.embed_tokens.weight")
	{
		return "tok_emb.weight";
	}
	else if (hfName == "tok_emb.weight")
	{
		return "tok_emb.weight";
	}

	// final norm
	if (hfName == "model.norm.weight")
	{
		return "norm.weight";
	}

	// lm head
	if (hfName == "lm_head.weight")
	{
		return "lm_head.weight";
	}

	auto it = mapping.find(hfName);
	if (it != mapping.end())
	{
		return it->second;
	}
	
	return hfName;
}