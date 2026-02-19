#include "./TokenizerJsonLoader.h"

#include <stdexcept>

#include <Utils/cJSON.h>
#include <Utils/Strings/StringUtils.h>

#include <FileUtils/FileMacros.h>


TokenizerJsonLoader::TokenizerJsonLoader(const std::string& jsonPath) : 
	jsonPath(jsonPath)
{
}

TokenizerJsonLoader::~TokenizerJsonLoader()
{
}

const TokenizerJsonLoader::ModelInfo& TokenizerJsonLoader::GetModelInfo() const
{
	return this->mi;
}

std::shared_ptr<TokenizerJsonLoader::IType> TokenizerJsonLoader::GetNormalizer() const
{
	return this->normalizer;
}

const std::vector<TokenizerJsonLoader::AddedToken>& TokenizerJsonLoader::AddedTokens() const
{
	return this->addedTokens;
}

const TokenHashMap& TokenizerJsonLoader::GetVocab() const
{
	return this->vocabData;
}

const ReverseTokenMap& TokenizerJsonLoader::GetVocabReversed() const
{
	return this->vocabDataReverse;
}

const std::vector<TokenizerJsonLoader::MergeInfo>& TokenizerJsonLoader::GetMerges() const
{
	return this->merges;
}

const std::vector<std::shared_ptr<TokenizerJsonLoader::IType>>& TokenizerJsonLoader::GetPreTokenizers() const
{
	return this->preTokenizers;
}

const std::vector<std::shared_ptr<TokenizerJsonLoader::IType>>& TokenizerJsonLoader::GetPostProcessors() const
{
	return this->postProcessors;
}

std::vector<char> TokenizerJsonLoader::LoadFile()
{	
	FILE* fp;
	my_fopen(&fp, jsonPath.c_str(), "r");
	if (fp == nullptr)
	{
		return {};
	}

	my_fseek(fp, 0L, SEEK_END);
	size_t size = static_cast<size_t>(my_ftell(fp));
	my_fseek(fp, 0L, SEEK_SET);

	std::vector<char> buffer(size + 1);
	fread(buffer.data(), sizeof(char), size, fp);
	buffer[size] = 0;
	
	fclose(fp);

	return buffer;
}


void TokenizerJsonLoader::Load()
{
	std::vector<char> jsonData = this->LoadFile();

	if (cJSON* json = cJSON_ParseWithLength(jsonData.data(), jsonData.size()))
	{
		auto addedTokens = cJSON_GetObjectItemCaseSensitive(json, "added_tokens");
		this->LoadAddedTokens(addedTokens);

		auto normalizer = cJSON_GetObjectItemCaseSensitive(json, "normalizer");
		this->LoadNormalizer(normalizer);

		auto preTokenizer = cJSON_GetObjectItemCaseSensitive(json, "pre_tokenizer");
		this->LoadPreTokenizer(preTokenizer);

		auto postProcessor = cJSON_GetObjectItemCaseSensitive(json, "post_processor");
		this->LoadPostProcessor(postProcessor);

		auto model = cJSON_GetObjectItemCaseSensitive(json, "model");
		this->LoadModelInfo(model);

		auto vocab = cJSON_GetObjectItemCaseSensitive(model, "vocab");
		this->LoadModelVocab(vocab);

		auto merges = cJSON_GetObjectItemCaseSensitive(model, "merges");
		this->LoadModelMerges(merges);

		cJSON_Delete(json);
	}
	else 
	{
		const char* err = cJSON_GetErrorPtr();
		printf("JSON parse error at: %s", (err ? err : "(unknown)"));
	}
	
}

//=============================================================================
// Load added tokens
//=============================================================================

/// <summary>
/// Load content of "added_tokens" elements:
///		"id": 128000,
///		"content": "<|begin_of_text|>",
///		"single_word" : false,
///		"lstrip" : false,
///		"rstrip" : false,
///		"normalized" : false,
///		"special" : true
/// </summary>
/// <param name="json"></param>
void TokenizerJsonLoader::LoadAddedTokens(cJSON* json)
{
	cJSON* item = nullptr;
	cJSON_ArrayForEach(item, json)
	{
		auto id = cJSON_GetObjectItemCaseSensitive(item, "id")->valueint;
		auto content = cJSON_GetObjectItemCaseSensitive(item, "content")->valuestring;
		
		auto single_word = (cJSON_GetObjectItemCaseSensitive(item, "single_word")->valueint == 1);
		auto lstrip = (cJSON_GetObjectItemCaseSensitive(item, "lstrip")->valueint == 1);
		auto rstrip = (cJSON_GetObjectItemCaseSensitive(item, "rstrip")->valueint == 1);
		auto normalized = (cJSON_GetObjectItemCaseSensitive(item, "normalized")->valueint == 1);
		auto special = (cJSON_GetObjectItemCaseSensitive(item, "special")->valueint == 1);

		this->addedTokens.emplace_back((char8_t*)content, id);
	}
}

//=============================================================================
// Normalizer
//=============================================================================

void TokenizerJsonLoader::LoadNormalizer(cJSON* json)
{
	if (json == nullptr)
	{
		this->normalizer = nullptr;
		return;
	}

	auto type = cJSON_GetObjectItemCaseSensitive(json, "type")->valuestring;
	if (strcmp(type, "Replace") == 0)
	{		
		this->normalizer = this->LoadReplaceType(json);
	}
}

//=============================================================================
// Pretokenizer
//=============================================================================

/// <summary>
/// Load "pre_tokenizer" setup
/// </summary>
/// <param name="json"></param>
void TokenizerJsonLoader::LoadPreTokenizer(cJSON* json)
{
	if (json == nullptr)
	{
		this->preTokenizers.clear();
		return;
	}

	auto type = cJSON_GetObjectItemCaseSensitive(json, "type")->valuestring;
	if (strcmp(type, "Sequence") == 0)
	{
		auto pretokenizers = cJSON_GetObjectItemCaseSensitive(json, "pretokenizers");

		this->preTokenizers = this->LoadSequenceType(pretokenizers);
	}	
	else if (strcmp(type, "Split") == 0)
	{
		this->preTokenizers = { this->LoadSplitType(json) };
	}
	else if (strcmp(type, "ByteLevel") == 0)
	{
		this->preTokenizers = { this->LoadByteLevelType(json) };
	}
	else if (strcmp(type, "Metaspace") == 0)
	{
		this->preTokenizers = { this->LoadMetaspaceType(json) };
	}
}


//=============================================================================
// Post processor
//=============================================================================

/// <summary>
/// Load "post_processor" setup
/// </summary>
/// <param name="json"></param>
void TokenizerJsonLoader::LoadPostProcessor(cJSON* json)
{	
	if (json == nullptr)
	{
		this->postProcessors.clear();
		return;
	}

	auto type = cJSON_GetObjectItemCaseSensitive(json, "type")->valuestring;

	if (strcmp(type, "Sequence") == 0)
	{
		auto processors = cJSON_GetObjectItemCaseSensitive(json, "processors");

		this->postProcessors = this->LoadSequenceType(processors);
	}	
	else if (strcmp(type, "TemplateProcessing") == 0)
	{		
		this->postProcessors = { this->LoadTemplateProcessingType(json) };
	}
}

//=============================================================================
// Load types
//=============================================================================

std::vector<std::shared_ptr<TokenizerJsonLoader::IType>> TokenizerJsonLoader::LoadSequenceType(cJSON* json)
{
	std::vector<std::shared_ptr<IType>> seq;

	cJSON* item = nullptr;
	cJSON_ArrayForEach(item, json)
	{
		auto type = cJSON_GetObjectItemCaseSensitive(item, "type")->valuestring;
		if (strcmp(type, "Split") == 0)
		{
			seq.emplace_back(this->LoadSplitType(item));
		}
		else if (strcmp(type, "ByteLevel") == 0)
		{
			seq.emplace_back(this->LoadByteLevelType(item));
		}
		else if (strcmp(type, "Metaspace") == 0)
		{
			seq.emplace_back(this->LoadMetaspaceType(item));
		}
		else if (strcmp(type, "TemplateProcessing") == 0)
		{
			seq.emplace_back(this->LoadTemplateProcessingType(item));
		}
	}

	return seq;
}

std::shared_ptr<TokenizerJsonLoader::ReplaceType> TokenizerJsonLoader::LoadReplaceType(cJSON* json)
{
	auto pattern = cJSON_GetObjectItemCaseSensitive(json, "pattern");
	auto stringPattern = cJSON_GetObjectItemCaseSensitive(pattern, "String")->valuestring;

	auto content = cJSON_GetObjectItemCaseSensitive(json, "content")->valuestring;

	return std::make_shared<ReplaceType>(stringPattern, content);
}

std::shared_ptr<TokenizerJsonLoader::SplitType> TokenizerJsonLoader::LoadSplitType(cJSON* json)
{
	const char* patternInfo = nullptr;
	SplitType::SplitDataType splitInfo = SplitType::SplitDataType::Unknown;

	auto pattern = cJSON_GetObjectItemCaseSensitive(json, "pattern");
	if (auto r = cJSON_GetObjectItemCaseSensitive(pattern, "Regex"))
	{
		patternInfo = r->valuestring;
		splitInfo = SplitType::SplitDataType::Regex;
	}
	else if (auto r = cJSON_GetObjectItemCaseSensitive(pattern, "String"))
	{
		patternInfo = r->valuestring;
		splitInfo = SplitType::SplitDataType::String;
	}
	

	auto behavior = cJSON_GetObjectItemCaseSensitive(json, "behavior")->valuestring;
	auto invert = (cJSON_GetObjectItemCaseSensitive(json, "invert")->valueint == 1);

	return std::make_shared<SplitType>(patternInfo, splitInfo, behavior, invert);
}

std::shared_ptr<TokenizerJsonLoader::ByteLevelType> TokenizerJsonLoader::LoadByteLevelType(cJSON* json)
{
	auto add_prefix_space = (cJSON_GetObjectItemCaseSensitive(json, "add_prefix_space")->valueint == 1);
	auto trim_offsets = (cJSON_GetObjectItemCaseSensitive(json, "trim_offsets")->valueint == 1);
	auto use_regex = (cJSON_GetObjectItemCaseSensitive(json, "use_regex")->valueint == 1);

	return std::make_shared<ByteLevelType>(add_prefix_space, trim_offsets, use_regex);
}

std::shared_ptr<TokenizerJsonLoader::MetaspaceType> TokenizerJsonLoader::LoadMetaspaceType(cJSON* json)
{
	const char* replacement = " ";
	const char* prepend_scheme = "always";
	bool split = true;

	if (auto p = cJSON_GetObjectItemCaseSensitive(json, "replacement"))
	{
		replacement = p->valuestring;
	}
	if (auto p = cJSON_GetObjectItemCaseSensitive(json, "prepend_scheme"))
	{
		prepend_scheme = p->valuestring;
	}
	if (auto p = cJSON_GetObjectItemCaseSensitive(json, "split"))
	{
		split = (p->valueint == 1);
	}

	return std::make_shared<MetaspaceType>(replacement, prepend_scheme, split);
}

std::shared_ptr<TokenizerJsonLoader::TemplateProcessingType> TokenizerJsonLoader::LoadTemplateProcessingType(cJSON* json)
{
	auto tt = std::make_shared<TemplateProcessingType>();

	auto single = cJSON_GetObjectItemCaseSensitive(json, "single");

	cJSON* item = nullptr;
	cJSON_ArrayForEach(item, single)
	{
		auto tmp = this->LoadSpecialTokenOrSequence(item);
		tt->single.try_emplace(tmp.content, tmp.id);
	}

	auto pair = cJSON_GetObjectItemCaseSensitive(json, "pair");

	item = nullptr;
	cJSON_ArrayForEach(item, pair)
	{
		auto tmp = this->LoadSpecialTokenOrSequence(item);
		tt->pair.try_emplace(tmp.content, tmp.id);
	}

	auto special_tokens = cJSON_GetObjectItemCaseSensitive(json, "special_tokens");

	item = nullptr;
	cJSON_ArrayForEach(item, special_tokens)
	{
		auto name = item->string;
		
		auto id = cJSON_GetObjectItemCaseSensitive(item, "id")->valuestring;

		std::vector<TokenId> ids;
		auto tmpIds = cJSON_GetObjectItemCaseSensitive(item, "ids");
		cJSON* tmpItem = nullptr;
		cJSON_ArrayForEach(tmpItem, tmpIds)
		{
			ids.emplace_back(tmpItem->valueint);
		}

		std::vector<char8_t*> tokens;
		auto tmpTokens = cJSON_GetObjectItemCaseSensitive(item, "tokens");
		tmpItem = nullptr;
		cJSON_ArrayForEach(tmpItem, tmpTokens)
		{
			tokens.emplace_back((char8_t*)tmpItem->valuestring);
		}

		for (size_t i = 0; i < ids.size(); i++)
		{
			auto it = tt->special.try_emplace(tokens[i]);
			it.first->second.emplace_back(ids[i]);
		}
	}

	return tt;
}

/// <summary>
/// Load content of "special_token" ior "sequence"
/// They both have same structure
///		"id": "<|begin_of_text|>",
///		"type_id": 0
/// </summary>
/// <param name="json"></param>
Token TokenizerJsonLoader::LoadSpecialTokenOrSequence(cJSON* json)
{
	cJSON* tmp = nullptr;
	tmp = cJSON_GetObjectItemCaseSensitive(json, "SpecialToken");
	if (tmp == nullptr)
	{
		tmp = cJSON_GetObjectItemCaseSensitive(json, "Sequence");
	}

	auto id = cJSON_GetObjectItemCaseSensitive(tmp, "id")->valuestring;
	auto type_id = cJSON_GetObjectItemCaseSensitive(tmp, "type_id")->valueint;

	return Token((char8_t*)id, type_id);
}

//=============================================================================
// Load model
//=============================================================================

/// <summary>
/// Load basic info from "model" element
///		"type": "BPE",
///		"dropout": null,
///		"unk_token" : null,
///		"continuing_subword_prefix" : null,
///		"end_of_word_suffix" : null,
///		"fuse_unk" : false,
///		"byte_fallback" : false,
///		"ignore_merges" : true,
/// </summary>
/// <param name="json"></param>
void TokenizerJsonLoader::LoadModelInfo(cJSON* json)
{
	auto tokenType = cJSON_GetObjectItemCaseSensitive(json, "type")->valuestring;
	auto dropout = cJSON_GetObjectItemCaseSensitive(json, "dropout")->valuestring;
	auto unk_token = cJSON_GetObjectItemCaseSensitive(json, "unk_token")->valuestring;
	auto continuing_subword_prefix = cJSON_GetObjectItemCaseSensitive(json, "continuing_subword_prefix")->valuestring;
	auto end_of_word_suffix = cJSON_GetObjectItemCaseSensitive(json, "end_of_word_suffix")->valuestring;
	

	mi.tokenType = tokenType;
	mi.dropout = (dropout != nullptr) ? dropout : "";
	mi.unk_token = (unk_token != nullptr) ? unk_token : "";
	mi.continuing_subword_prefix = (continuing_subword_prefix != nullptr) ? continuing_subword_prefix : "";
	mi.end_of_word_suffix = (end_of_word_suffix != nullptr) ? end_of_word_suffix : "";


	mi.fuse_unk = (cJSON_GetObjectItemCaseSensitive(json, "fuse_unk")->valueint == 1);
	mi.byte_fallback = (cJSON_GetObjectItemCaseSensitive(json, "byte_fallback")->valueint == 1);
	mi.ignore_merges = (cJSON_GetObjectItemCaseSensitive(json, "ignore_merges")->valueint == 1);
}

/// <summary>
/// Load vocabulary
/// </summary>
/// <param name="json"></param>
void TokenizerJsonLoader::LoadModelVocab(cJSON* json)
{
	cJSON* item = nullptr;
	cJSON_ArrayForEach(item, json)
	{				
		StringUtf8 str = (char8_t*)item->string;

		uint64_t tokenHash = Token::CalcHash(str);
		
		auto it = this->vocabData.try_emplace(tokenHash, item->valueint);
		if (it.second == false)
		{
			//Collision - should not happen
			throw std::runtime_error("Hash collision - tokenizer broken");
		}
		
		this->vocabDataReverse.try_emplace(item->valueint, std::move(str));
	}
}

/// <summary>
/// Load merges section
/// loads only if ignore_merges is false
/// </summary>
/// <param name="json"></param>
void TokenizerJsonLoader::LoadModelMerges(cJSON* json)
{	
	cJSON* item = nullptr;
	cJSON_ArrayForEach(item, json)
	{
		if (cJSON_IsString(item))
		{
			this->ProcessStringMerges(item->valuestring);
		}
		else if (cJSON_IsArray(item))
		{
			this->ProcessArrayMerges(item);
		}
				
	}
}

void TokenizerJsonLoader::ProcessStringMerges(const char* data)
{
	std::u8string_view m = (char8_t*)data;
	auto parts = StringUtils::Split<std::u8string_view>(m, u8" ");

	MergeInfo merge;
	for (size_t i = 0; i < parts.size(); i++)
	{
		merge.parts.emplace_back(parts[i]);
		merge.hashes.emplace_back(Token::CalcHash(parts[i]));
	}

	this->merges.push_back(std::move(merge));
}

void TokenizerJsonLoader::ProcessArrayMerges(cJSON* json)
{
	MergeInfo merge;

	cJSON* item = nullptr;
	cJSON_ArrayForEach(item, json)
	{
		std::u8string_view m = (char8_t*)item->valuestring;

		merge.parts.emplace_back(m);
		merge.hashes.emplace_back(Token::CalcHash(m));
	}

	this->merges.push_back(std::move(merge));
}
