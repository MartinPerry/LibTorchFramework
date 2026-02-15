#ifndef TOKENIZER_BPE_H
#define TOKENIZER_BPE_H

class UnicodeRegex;

#include <string>
#include <memory>

#include "./Tokenizers.h"
#include "./TokenizerJsonLoader.h"

class TokenizerBPE
{
public:
	TokenizerBPE(const std::string& jsonPath);
	~TokenizerBPE();

	void Load();

	std::vector<TokenId> Encode(const StringUtf8& str, bool addBos, bool addEos);
	StringUtf8 Decode(const std::vector<TokenId>& ids);

protected:

	std::shared_ptr<TokenizerJsonLoader> json;

	Token bos;
	Token eos;

	std::unordered_map<StringUtf8Hash, std::unordered_map<StringUtf8Hash, int>> bpeRanks;

	std::shared_ptr<UnicodeRegex> splitRx;

	std::unordered_map<char8_t, UnicodeCodePoint> bytesToUnicodeMapping;
	std::unordered_map<UnicodeCodePoint, char8_t> unicodeToBytesMapping;

	
	TokenId GetSpecialTokenId(const StringUtf8& token) const;

	void CreateBytesToUnicodeMapping();

	std::vector<StringUtf8> SplitIsolated(const StringUtf8& str);

	std::vector<TokenId> EncodePiece(const std::vector<UnicodeCodePoint>& u);
	
};

#endif
