#ifndef TOKENIZER_JSON_LOADER_H
#define TOKENIZER_JSON_LOADER_H

struct cJSON;

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

#include "./Tokenizers.h"

class TokenizerJsonLoader 
{
public:
	struct ModelInfo
	{
		std::string tokenType;
		std::string dropout;
		std::string unk_token;
		std::string continuing_subword_prefix;
		std::string end_of_word_suffix;
		bool fuse_unk;
		bool byte_fallback;
		bool ignore_merges;
	};

	//=============================================================
	
	struct AddedToken : public Token
	{		
		using Token::Token;
	};

	
	//=============================================================

	struct ReplaceType;
	struct SplitType;
	struct ByteLevelType;
	struct MetaspaceType;
	struct TemplateProcessingType;

	struct IType
	{
		std::string type;

		IType(const char* type) : 
			type(type)
		{}
		virtual ~IType() = default;

		virtual const SplitType* GetSplitType() const { return nullptr; }
		virtual const ByteLevelType* GetByteLevelType() const { return nullptr; }
		virtual const MetaspaceType* GetMetaspaceType() const { return nullptr; }
		virtual const TemplateProcessingType* GetTemplateProcessingType() const { return nullptr; }
		virtual const ReplaceType* GetReplaceType() const { return nullptr; }
	};

	struct SplitType : public IType
	{
		enum class SplitDataType 
		{
			Regex,
			String,
			Unknown
		};

		std::string splitData;
		SplitDataType splitType;
		std::string behavior;
		bool invert;

		SplitType(const char* splitData, SplitDataType splitType,  const char* behavior, bool invert) :
			IType("Split"),
			splitData(splitData),
			splitType(splitType),
			behavior(behavior),
			invert(invert)
		{}

		const SplitType* GetSplitType() const override { return this; }
	};

	struct ByteLevelType : public IType
	{
		bool add_prefix_space;
		bool trim_offsets;
		bool use_regex;

		ByteLevelType(bool add_prefix_space, bool trim_offsets, bool use_regex) :
			IType("ByteLevel"),
			add_prefix_space(add_prefix_space),
			trim_offsets(trim_offsets),
			use_regex(use_regex)
		{}

		const ByteLevelType* GetByteLevelType() const override { return this; }
	};

	struct MetaspaceType : public IType
	{
		std::string replacement;
		std::string prepend_scheme;
		bool split;

		MetaspaceType(const char* replacement, const char* prepend_scheme, bool split) :
			IType("Metaspace"),
			replacement(replacement ? replacement : " "),
			prepend_scheme(prepend_scheme ? prepend_scheme : "always"),
			split(split)
		{}

		const MetaspaceType* GetMetaspaceType() const override { return this; }
	};

	struct TemplateProcessingType : public IType
	{
		TokenMap single;
		TokenMap pair;
		std::unordered_map<StringUtf8, std::vector<TokenId>> special;

		TemplateProcessingType() : 
			IType("TemplateProcessing")
		{}

		const TemplateProcessingType* GetTemplateProcessingType() const override { return this; }
	};

	struct ReplaceType : public IType
	{
		std::string splitData;
		std::string content;

		ReplaceType(const char* splitData, const char* content) :
			IType("ReplaceType"),
			splitData(splitData ? splitData : ""),
			content(content ? content : "")
		{
		}

		const ReplaceType* GetReplaceType() const override { return this; }
	};

	//=============================================================

	struct MergeInfo 
	{
		std::vector<StringUtf8> parts;
		std::vector<StringUtf8Hash> hashes;
	};

	//=============================================================

	TokenizerJsonLoader(const std::string& jsonPath);
	virtual ~TokenizerJsonLoader();

	const ModelInfo& GetModelInfo() const;
	std::shared_ptr<IType> GetNormalizer() const;

	const std::vector<AddedToken>& AddedTokens() const;
	const TokenHashMap& GetVocab() const;
	const ReverseTokenMap& GetVocabReversed() const;
	const std::vector<MergeInfo>& GetMerges() const;

	const std::vector<std::shared_ptr<IType>>& GetPreTokenizers() const;
	const std::vector<std::shared_ptr<IType>>& GetPostProcessors() const;

	template <typename Type>
	std::shared_ptr<Type> GetPretokenizerType() const
	{
		for (auto p : this->preTokenizers)
		{
			auto tmp = std::dynamic_pointer_cast<Type>(p);
			if (tmp)
			{
				return tmp;
			}
		}

		return nullptr;
	}

	template <typename Type>
	std::shared_ptr<Type> GetPostProcessorType() const
	{
		for (auto p : this->postProcessors)
		{
			auto tmp = std::dynamic_pointer_cast<Type>(p);
			if (tmp)
			{
				return tmp;
			}
		}

		return nullptr;
	}

	void Load();

protected:
	std::string jsonPath;
	
	ModelInfo mi;

	std::vector<AddedToken> addedTokens;
	TokenHashMap vocabData;
	ReverseTokenMap vocabDataReverse;
	std::vector<MergeInfo> merges;

	std::shared_ptr<IType> normalizer;

	std::vector<std::shared_ptr<IType>> preTokenizers;
	std::vector<std::shared_ptr<IType>> postProcessors;
	
	//------------

	std::vector<char> LoadFile();

	void LoadAddedTokens(cJSON* json);

	void LoadNormalizer(cJSON* json);
	void LoadPreTokenizer(cJSON* json);		
	void LoadPostProcessor(cJSON* json);

	std::vector<std::shared_ptr<IType>> LoadSequenceType(cJSON* json);

	std::shared_ptr<SplitType> LoadSplitType(cJSON* json);
	std::shared_ptr<ReplaceType> LoadReplaceType(cJSON* json);
	std::shared_ptr<ByteLevelType> LoadByteLevelType(cJSON* json);
	std::shared_ptr<MetaspaceType> LoadMetaspaceType(cJSON* json);
	std::shared_ptr<TemplateProcessingType> LoadTemplateProcessingType(cJSON* json);
	
	Token LoadSpecialTokenOrSequence(cJSON* json);


	void LoadModelInfo(cJSON* json);
	void LoadModelVocab(cJSON* json);
	void LoadModelMerges(cJSON* json);

	void ProcessStringMerges(const char* data);
	void ProcessArrayMerges(cJSON* json);
};

#endif
