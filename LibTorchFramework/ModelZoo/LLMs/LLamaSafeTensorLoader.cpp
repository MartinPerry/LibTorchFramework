#include "./LLamaSafeTensorLoader.h"

#include "./llama.h"

#include <Utils/Logger.h>

using namespace ModelZoo::llama;

LoadStateDictReport LLamaSafeTensorLoader::LoadFromHfSafetensors(
	LlamaForCausalLM& model,
	const std::filesystem::path& modelDir,
	bool strict)
{
	this->CreateMapping(model.GetConfig());

	return this->LoadSafetensors(modelDir, model, strict, [&](const std::string& hfName) -> std::string {
		return this->MappingHfKeysToOurs(hfName);
	});
	
	/*
	TensorMap stateDict = this->LoadSafetensorsSharded(modelDir, [&](const std::string& hfName) -> std::string {
		return this->MappingHfKeysToOurs(hfName);
	});
	
	return this->FillModelStateDict(model, stateDict, strict);
	*/
}



void LLamaSafeTensorLoader::CreateMapping(const LlamaConfig& cfg)
{
	mapping.clear();

	// layers
	for (int64_t i = 0; i < cfg.num_hidden_layers; ++i)
	{
		const std::string prefixHf = "model.layers." + std::to_string(i) + ".";
		const std::string prefixOurs = "layers." + std::to_string(i) + ".";

		mapping[prefixHf + "input_layernorm.weight"] = prefixOurs + "attn_norm.weight";
		mapping[prefixHf + "post_attention_layernorm.weight"] = prefixOurs + "ffn_norm.weight";
		mapping[prefixHf + "self_attn.q_proj.weight"] = prefixOurs + "attn.q_proj.weight";
		mapping[prefixHf + "self_attn.k_proj.weight"] = prefixOurs + "attn.k_proj.weight";
		mapping[prefixHf + "self_attn.v_proj.weight"] = prefixOurs + "attn.v_proj.weight";
		mapping[prefixHf + "self_attn.o_proj.weight"] = prefixOurs + "attn.o_proj.weight";
		mapping[prefixHf + "mlp.gate_proj.weight"] = prefixOurs + "mlp.gate_proj.weight";
		mapping[prefixHf + "mlp.up_proj.weight"] = prefixOurs + "mlp.up_proj.weight";
		mapping[prefixHf + "mlp.down_proj.weight"] = prefixOurs + "mlp.down_proj.weight";

		/*
		mapping[prefixOurs + "attn_norm.weight"] = prefixHf + "input_layernorm.weight";
		mapping[prefixOurs + "ffn_norm.weight"] = prefixHf + "post_attention_layernorm.weight";
		mapping[prefixOurs + "attn.q_proj.weight"] = prefixHf + "self_attn.q_proj.weight";
		mapping[prefixOurs + "attn.k_proj.weight"] = prefixHf + "self_attn.k_proj.weight";
		mapping[prefixOurs + "attn.v_proj.weight"] = prefixHf + "self_attn.v_proj.weight";
		mapping[prefixOurs + "attn.o_proj.weight"] = prefixHf + "self_attn.o_proj.weight";
		mapping[prefixOurs + "mlp.gate_proj.weight"] = prefixHf + "mlp.gate_proj.weight";
		mapping[prefixOurs + "mlp.up_proj.weight"] = prefixHf + "mlp.up_proj.weight";
		mapping[prefixOurs + "mlp.down_proj.weight"] = prefixHf + "mlp.down_proj.weight";
		*/
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