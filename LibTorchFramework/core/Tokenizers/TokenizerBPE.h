#ifndef TOKENIZER_BPE_H
#define TOKENIZER_BPE_H

class UnicodeRegex;

#include <string>
#include <memory>

#include "./Tokenizers.h"
#include "./TokenizerJsonLoader.h"

class TokenizerBPE : public Tokenizer
{
public:
	TokenizerBPE(const std::string& jsonPath);
	virtual ~TokenizerBPE();

	const Token& GetBos() const;
	const Token& GetEos() const;

	void Load();

	std::vector<TokenId> Encode(const StringUtf8& str) override;
	std::vector<TokenId> Encode(const StringUtf8& str, bool addBos, bool addEos);
	
	StringUtf8 Decode(const std::vector<TokenId>& ids) override;

protected:

	std::shared_ptr<TokenizerJsonLoader> json;

	Token bos;
	Token eos;
	Token unk;

	std::unordered_map<StringUtf8, TokenId> specialTokenIds;

	std::unordered_map<StringUtf8Hash, std::unordered_map<StringUtf8Hash, int>> bpeRanks;

	std::shared_ptr<UnicodeRegex> splitRx;
	StringUtf8 splitStr;
	std::string splitBehavior;
	bool splitInvert;
	bool useByteLevelEncoding;
	
	StringUtf8 decodeReplaceFrom;
	StringUtf8 decodeReplaceTo;

	std::unordered_map<char8_t, UnicodeCodePoint> bytesToUnicodeMapping;
	std::unordered_map<UnicodeCodePoint, char8_t> unicodeToBytesMapping;

	
	TokenId GetSpecialTokenId(const StringUtf8& token) const;

	void CreateBytesToUnicodeMapping();

	StringUtf8 RunNormalizer(const StringUtf8& str);

	std::vector<StringUtf8> SplitIsolated(const StringUtf8& str);
	std::vector<StringUtf8> SplitIsolatedRegex(const StringUtf8& str);
	std::vector<StringUtf8> SplitMergedWithPrevious(const StringUtf8& str) const;
	TokenId FindByteFallbackId(char8_t b) const;
	void AppendFallbackIds(UnicodeCodePoint cp, std::vector<TokenId>& ids);
	
	std::vector<std::pair<bool, StringUtf8>> SplitSpecialTokens(const StringUtf8& str) const;

	std::vector<TokenId> EncodePiece(const std::vector<UnicodeCodePoint>& u);
	

	static bool TryParseByteFallbackToken(const StringUtf8& token, char8_t& outByte);

	void AppendUtf8Bytes(UnicodeCodePoint cp, std::vector<char8_t>& out);
};

#endif
