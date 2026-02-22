#include "./TokenizerBPE.h"

#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <cstring>
#include <limits>
#include <utility>

#include "./Strings/UnicodeRegex.h"

#include <Utils/Strings/StringUtils.h>
#include <Utils/Strings/StringIterators.h>


TokenizerBPE::TokenizerBPE(const std::string& jsonPath) : 
	json(std::make_shared<TokenizerJsonLoader>(jsonPath)),
	bos(u8"", -1),
	eos(u8"", -1),
	unk(u8"", -1),
	splitBehavior("Isolated"),
	splitInvert(false),
	useByteLevelEncoding(true)	
{
	
}


TokenizerBPE::~TokenizerBPE()
{	
}

const Token& TokenizerBPE::GetBos() const
{
	return this->bos;
}

const Token& TokenizerBPE::GetEos() const
{
	return this->eos;
}

//==========================================================================
// Loading and prepearing data
//==========================================================================

void TokenizerBPE::Load()
{
	this->CreateBytesToUnicodeMapping();
	
	json->Load();
	this->splitRx.reset();
	this->splitStr.clear();
	this->bpeRanks.clear();

	auto split = json->GetPretokenizerType<TokenizerJsonLoader::SplitType>();
	if (split)
	{
		this->splitBehavior = split->behavior;
		this->splitInvert = split->invert;
		

		if (split->splitType == TokenizerJsonLoader::SplitType::SplitDataType::Regex)
		{
			splitRx = std::make_shared<UnicodeRegex>(split->splitData);
		}
		else if (split->splitType == TokenizerJsonLoader::SplitType::SplitDataType::String)
		{
			splitStr = split->splitData;
		}
	}
	else
	{
		this->splitBehavior = "Isolated";
		this->splitInvert = false;
	}

	this->useByteLevelEncoding = (json->GetPretokenizerType<TokenizerJsonLoader::ByteLevelType>() != nullptr);

	this->decodeReplaceFrom.clear();
	this->decodeReplaceTo.clear();

	auto normalizer = json->GetNormalizer();
	if ((normalizer != nullptr) && (normalizer->GetReplaceType() != nullptr))
	{
		auto repl = normalizer->GetReplaceType();
		if (repl->splitData == u8" ")
		{
			this->decodeReplaceFrom = repl->content;
			this->decodeReplaceTo = repl->splitData;
		}
	}

	auto mi = json->GetModelInfo();
	if (!mi.unk_token.empty())
	{
		this->unk.id = this->GetSpecialTokenId(mi.unk_token);
	}
	
	std::vector<StringUtf8> bosTokens = { u8"<|begin_of_text|>", u8"<bos>", u8"<s>" };
	std::vector<StringUtf8> eosTokens = { u8"<|end_of_text|>", u8"<eos>", u8"</s>" };
		
	for (const auto& token : bosTokens)
	{
		bos.id = this->GetSpecialTokenId(token);
		if (bos.id != -1)
		{
			bos.content = token;
			break;
		}
	}

	for (const auto& token : eosTokens)
	{
		eos.id = this->GetSpecialTokenId(token);
		if (eos.id != -1)
		{
			eos.content = token;
			break;
		}
	}

			
	this->specialTokenIds.clear();
	
	
	const auto& added = json->AddedTokens();
	for (const auto& it : added)
	{
		//StringUtf8Hash h = Token::CalcHash(it.content);
		this->specialTokenIds.try_emplace(it.content, it.id);
	}
	
		

	const auto& merges = json->GetMerges();
		
	for (int i = 0; i < merges.size(); i++)
	{
		const auto& mi = merges[i];
			
		if (mi.hashes.size() == 2)
		{			
			auto it = this->bpeRanks.try_emplace(mi.hashes[0]);
			it.first->second.try_emplace(mi.hashes[1], i);
		}		
	}	
	
}

TokenId TokenizerBPE::GetSpecialTokenId(const StringUtf8& token) const
{	
	const auto& vocab = json->GetVocab();
	
	const auto& added = json->AddedTokens();
	auto tpl = json->GetPostProcessorType<TokenizerJsonLoader::TemplateProcessingType>();
	if (tpl != nullptr)
	{
		auto it = tpl->special.find(token);
		if (it != tpl->special.end())
		{
			return it->second[0];
		}
	}

	auto tokenHash = Token::CalcHash(token);

	auto jt = vocab.find(tokenHash);
	if (jt != vocab.end())
	{
		return jt->second;
	}
	else
	{
		for (const auto& t : added)
		{
			if (t.content == token)
			{
				return t.id;					
			}
		}
	}

	return -1;
}


void TokenizerBPE::CreateBytesToUnicodeMapping()
{
	// GPT-2 byte encoder mapping bytes -> unicode chars
	std::vector<char8_t> bs;
	std::vector<UnicodeCodePoint> cs;
	bs.reserve(256);
	cs.reserve(256);

	for (char8_t b = '!'; b <= '~'; b++)
	{
		bs.push_back(b);
		cs.push_back(b);
	}
	for (char8_t b = 0xA1; b <= 0xAC; b++)
	{
		bs.push_back(b);
		cs.push_back(b);
	}
	for (char8_t b = 0xAE; b < 0xFF; b++)
	{
		bs.push_back(b);
		cs.push_back(b);
	}
	bs.push_back(0xFF);
	cs.push_back(0xFF);
	
	int n = 0;
	for (int b = 0; b < 256; b++) 
	{
		bool found = false;
		for (int v : bs) 
		{
			if (v == b) 
			{
				found = true;
				break;
			}
		}

		if (!found) 
		{
			bs.push_back(b);
			cs.push_back(256 + n);
			n++;
		}
	}

	bytesToUnicodeMapping.clear();
	bytesToUnicodeMapping.reserve(256);
	for (size_t i = 0; i < bs.size(); ++i) 
	{
		bytesToUnicodeMapping.try_emplace(bs[i], cs[i]);

		unicodeToBytesMapping.try_emplace(cs[i], bs[i]);
	}	
}

//==========================================================================
// Run encode for input text
//==========================================================================

StringUtf8 TokenizerBPE::RunNormalizer(const StringUtf8& str)
{
	auto n = json->GetNormalizer();
	if (n == nullptr)
	{
		return str;
	}
	
	if (auto repl = n->GetReplaceType())
	{
		auto tmp = str;
		StringUtils::ReplaceAllSubStr(tmp, repl->splitData, repl->content);		
		return tmp;
	}

	return str;
}

std::vector<StringUtf8> TokenizerBPE::SplitIsolated(const StringUtf8& str)
{
	if (str.empty())
	{
		return {};
	}

	if (this->splitRx)
	{
		return this->SplitIsolatedRegex(str);
	}

	if (this->splitStr.empty())
	{
		return { str };
	}

	if (this->splitInvert)
	{
		return { str };
	}

	if (this->splitBehavior == "MergedWithPrevious")
	{
		return this->SplitMergedWithPrevious(str);
	}
	
	return StringUtils::Split(str, this->splitStr);	
}

/// <summary>
/// Split behavior=Isolated: keep matched spans as tokens, plus any gaps.
/// Your regex usually covers the whole string, but we do it correctly anyway.
/// </summary>
/// <param name="str"></param>
/// <returns></returns>
std::vector<StringUtf8> TokenizerBPE::SplitIsolatedRegex(const StringUtf8& str)
{
	std::vector<StringUtf8> res;
	size_t last = 0;

	auto spans = this->splitRx->FindSpans(str);
	
	for (const auto& span : spans)
	{		
		const size_t a = span.a;
		const size_t b = span.b;

		if (a > last)
		{
			//gap = str[last:a]			
			res.emplace_back(str.substr(last, a - last));
		}

		//tok = text[a:b]		
		res.emplace_back(str.substr(a, b - a));
		last = b;
	}

	if (last < str.length())
	{
		//tail = text[last:]			
		res.emplace_back(str.substr(last));
	}

	return res;
}

std::vector<StringUtf8> TokenizerBPE::SplitMergedWithPrevious(const StringUtf8& str) const
{
	std::vector<StringUtf8> out;
	if (str.empty())
	{
		return out;
	}
	if (this->splitStr.empty())
	{
		out.emplace_back(str);
		return out;
	}

	size_t pos = 0;
	const size_t dlen = this->splitStr.size();
	while (pos < str.size())
	{
		size_t found = str.find(this->splitStr, pos);
		if (found == StringUtf8::npos)
		{
			if (pos < str.size())
			{
				out.emplace_back(str.substr(pos));
			}
			break;
		}

		size_t next = found + dlen;
		if (found > pos)
		{
			out.emplace_back(str.substr(pos, next - pos));
		}
		else
		{
			if (!out.empty())
			{
				out.back().append(this->splitStr);
			}
			else
			{
				out.emplace_back(this->splitStr);
			}
		}
		pos = next;
	}

	return out;
}


void TokenizerBPE::AppendFallbackIds(UnicodeCodePoint cp, std::vector<TokenId>& ids)
{
	if (cp < 0x80) {                   // one octet
		ids.push_back(FindByteFallbackId(static_cast<char8_t>(cp)));		
	}
	else if (cp < 0x800) {                // two octets
		ids.push_back(FindByteFallbackId(static_cast<char8_t>((cp >> 6) | 0xc0)));
		ids.push_back(FindByteFallbackId(static_cast<char8_t>((cp & 0x3f) | 0x80)));
	}
	else if (cp < 0x10000) {              // three octets
		ids.push_back(FindByteFallbackId(static_cast<char8_t>((cp >> 12) | 0xe0)));
		ids.push_back(FindByteFallbackId(static_cast<char8_t>(((cp >> 6) & 0x3f) | 0x80)));
		ids.push_back(FindByteFallbackId(static_cast<char8_t>((cp & 0x3f) | 0x80)));
	}
	else {                                // four octets
		ids.push_back(FindByteFallbackId(static_cast<char8_t>((cp >> 18) | 0xf0)));
		ids.push_back(FindByteFallbackId(static_cast<char8_t>(((cp >> 12) & 0x3f) | 0x80)));
		ids.push_back(FindByteFallbackId(static_cast<char8_t>(((cp >> 6) & 0x3f) | 0x80)));
		ids.push_back(FindByteFallbackId(static_cast<char8_t>((cp & 0x3f) | 0x80)));
	}
}

TokenId TokenizerBPE::FindByteFallbackId(char8_t b) const
{
	constexpr char8_t hex[] = u8"0123456789ABCDEF";
	StringUtf8 tok = u8"<0x00>";
	tok[3] = hex[(b >> 4) & 0xF];
	tok[4] = hex[b & 0xF];

	const auto& vocab = json->GetVocab();

	auto h = Token::CalcHash(tok);
	auto it = vocab.find(h);
	if (it != vocab.end())
	{
		return it->second;
	}

	return this->unk.id;
}


std::vector<std::pair<bool, StringUtf8>> TokenizerBPE::SplitSpecialTokens(const StringUtf8& str) const
{
	if (str.empty())
	{
		return {};
	}
	if (this->specialTokenIds.empty())
	{
		return { { false, str } };
	}

	std::vector<std::pair<bool, StringUtf8>> out;
	size_t pos = 0;
	while (pos < str.size())
	{
		size_t bestPos = StringUtf8::npos;
		StringUtf8 bestToken;

		for (const auto& kv : this->specialTokenIds)
		{
			const auto& tok = kv.first;
			
			size_t found = str.find(tok, pos);
			if (found == StringUtf8::npos)
			{
				continue;
			}

			if ((bestPos == StringUtf8::npos) || (found < bestPos) || ((found == bestPos) && (tok.size() > bestToken.size())))
			{
				bestPos = found;
				bestToken = tok;
			}
		}

		if (bestPos == StringUtf8::npos)
		{
			out.emplace_back(false, str.substr(pos));
			break;
		}

		if (bestPos > pos)
		{
			out.emplace_back(false, str.substr(pos, bestPos - pos));
		}

		out.emplace_back(true, bestToken);
		pos = bestPos + bestToken.size();
	}

	return out;
}


std::vector<TokenId> TokenizerBPE::EncodePiece(const std::vector<UnicodeCodePoint>& unicodes)
{
	
	const auto& vocab = json->GetVocab();

	//tokenizers BPE: with ignore_merges=true, first try full-token vocab lookup.
	if (json->GetModelInfo().ignore_merges)
	{
		auto tokenHash = Token::CalcHash(unicodes);
		auto it = vocab.find(tokenHash);
		if (it != vocab.end())
		{
			return { it->second };
		}
	}

	//if not found, look in merges

	
	std::vector<std::vector<UnicodeCodePoint>> symbols;
	for (auto ch : unicodes)
	{
		symbols.push_back({ ch });
	}
	
	while (symbols.size() > 1) 
	{		
		int best_rank = std::numeric_limits<int>::max();
		std::pair<StringUtf8Hash, StringUtf8Hash> best_pair{};

		for (size_t i = 0; i < symbols.size() - 1; i++) 
		{
			auto h0 = Token::CalcHash(symbols[i]);
						
			auto it = bpeRanks.find(h0);
			if (it == bpeRanks.end())
			{
				continue;
			}

			auto h1 = Token::CalcHash(symbols[i + 1]);

			auto jt = it->second.find(h1);
			if (jt == it->second.end())
			{
				continue;
			}

			if (jt->second < best_rank) 
			{
				best_rank = jt->second;
				best_pair = { h0, h1 };				
			}
		}

		if (best_rank == std::numeric_limits<int>::max())
		{
			break;
		}

		std::vector<std::vector<UnicodeCodePoint>> merged;
		merged.reserve(symbols.size());

		size_t i = 0;
		while (i < symbols.size()) 
		{			
			if ((i + 1 < symbols.size()) &&
				(Token::CalcHash(symbols[i]) == best_pair.first) &&
				(Token::CalcHash(symbols[i + 1]) == best_pair.second))
			{				
				auto& tmp = merged.emplace_back(symbols[i]);
				tmp.insert(tmp.end(), symbols[i + 1].begin(), symbols[i + 1].end());
				i += 2;
			}
			else 
			{
				merged.push_back(symbols[i]);
				i++;
			}
		}

		symbols.swap(merged);
	}
	
	std::vector<TokenId> ids;
	for (const auto& s : symbols)
	{
		auto tokenHash = Token::CalcHash(s);
		auto it = vocab.find(tokenHash);
		if (it != vocab.end())
		{
			ids.push_back(it->second);
		}
		else 
		{
			if (json->GetModelInfo().byte_fallback)
			{								
				for (auto ch : s)
				{
					AppendFallbackIds(ch, ids);
				}
			}
			else
			{
				for (auto ch : s)
				{
					auto tokenHash = Token::CalcHash(ch);
					auto it = vocab.find(tokenHash);
					if (it != vocab.end())
					{
						ids.push_back(it->second);
					}
					else 
					{
						ids.push_back(this->unk.id);
					}
				}
			}
		}
	}
	
	return ids;
}

std::vector<TokenId> TokenizerBPE::Encode(const StringUtf8& str)
{
	return this->Encode(str, false, false);
}

/// <summary>
/// str - unicode string
/// </summary>
/// <param name="str"></param>
/// <param name="addBos"></param>
/// <param name="addEos"></param>
/// <returns></returns>
std::vector<TokenId> TokenizerBPE::Encode(const StringUtf8& str, bool addBos, bool addEos)
{	
	std::vector<TokenId> ids;

	//post_processor: TemplateProcessing prepends <|begin_of_text|>
	if ((addBos) && (bos.id != -1))
	{
		ids.push_back(bos.id);
	}

	auto normalizedStr = this->RunNormalizer(str);


	auto split = this->SplitSpecialTokens(normalizedStr);
	for (const auto& segment : split)
	{
		if (segment.second.empty())
		{
			continue;
		}

		if (segment.first)
		{
			auto it = this->specialTokenIds.find(segment.second);
			if (it != this->specialTokenIds.end())
			{
				ids.push_back(it->second);
			}
			continue;
		}

		auto pieces = this->SplitIsolated(segment.second);
		for (const auto& p : pieces)
		{
			if (p.empty())
			{
				continue;
			}

			std::vector<UnicodeCodePoint> unicodes;
			unicodes.reserve(p.size());

			if (this->useByteLevelEncoding)
			{				
				for (auto b : p)
				{
					auto it = this->bytesToUnicodeMapping.find(b);
					unicodes.push_back((it != this->bytesToUnicodeMapping.end()) ? it->second : static_cast<UnicodeCodePoint>(b));
				}
			}
			else
			{
				CustomU8Iterator it(p);
				UnicodeCodePoint ch;
				while ((ch = it.GetCurrentAndAdvance()) != it.DONE)
				{
					unicodes.push_back(ch);
				}
			}


			auto tmp = this->EncodePiece(unicodes);

			ids.insert(ids.end(), tmp.begin(), tmp.end());
		}
	}


	if ((addEos) && (eos.id != -1))
	{
		ids.push_back(eos.id);
	}
	
	return ids;
}

//==========================================================================
// Decode
//==========================================================================

bool TokenizerBPE::TryParseByteFallbackToken(const StringUtf8& token, char8_t& outByte)
{
	if (token.size() != 6)
	{
		return false;
	}
	if ((token[0] != u8'<') || (token[1] != u8'0') || (token[2] != u8'x') || (token[5] != u8'>'))
	{
		return false;
	}

	auto hexValue = [](char8_t c) -> int
		{
			if ((c >= u8'0') && (c <= u8'9'))
			{
				return c - u8'0';
			}
			if ((c >= u8'a') && (c <= u8'f'))
			{
				return c - u8'a' + 10;
			}
			if ((c >= u8'A') && (c <= u8'F'))
			{
				return c - u8'A' + 10;
			}
			return -1;
		};

	int hi = hexValue(token[3]);
	int lo = hexValue(token[4]);
	if ((hi < 0) || (lo < 0))
	{
		return false;
	}

	outByte = static_cast<char8_t>((hi << 4) | lo);
	return true;
}

void TokenizerBPE::AppendUtf8Bytes(UnicodeCodePoint cp, std::vector<char8_t>& out)
{
	uint32_t c = static_cast<uint32_t>(cp);
	if (c <= 0x7F)
	{
		out.push_back(static_cast<char8_t>(c));
	}
	else if (c <= 0x7FF)
	{
		out.push_back(static_cast<char8_t>(0xC0 | (c >> 6)));
		out.push_back(static_cast<char8_t>(0x80 | (c & 0x3F)));
	}
	else if (c <= 0xFFFF)
	{
		out.push_back(static_cast<char8_t>(0xE0 | (c >> 12)));
		out.push_back(static_cast<char8_t>(0x80 | ((c >> 6) & 0x3F)));
		out.push_back(static_cast<char8_t>(0x80 | (c & 0x3F)));
	}
	else if (c <= 0x10FFFF)
	{
		out.push_back(static_cast<char8_t>(0xF0 | (c >> 18)));
		out.push_back(static_cast<char8_t>(0x80 | ((c >> 12) & 0x3F)));
		out.push_back(static_cast<char8_t>(0x80 | ((c >> 6) & 0x3F)));
		out.push_back(static_cast<char8_t>(0x80 | (c & 0x3F)));
	}
}

StringUtf8 TokenizerBPE::Decode(const std::vector<TokenId>& ids)
{
	const auto& revVocab = json->GetVocabReversed();

	std::vector<char8_t> outBytes;

	for (auto id : ids)
	{
		auto it = revVocab.find(id);
		if (it == revVocab.end())
		{
			continue;
		}

		auto token = it->second;
		if (!this->decodeReplaceFrom.empty())
		{
			StringUtils::ReplaceAllSubStr(token, this->decodeReplaceFrom, this->decodeReplaceTo);
		}

		char8_t decodedByte = 0;
		if (json->GetModelInfo().byte_fallback && TryParseByteFallbackToken(token, decodedByte))
		{
			outBytes.push_back(decodedByte);
			continue;
		}

		if (this->useByteLevelEncoding)
		{
			CustomU8Iterator itU8(token);
			UnicodeCodePoint ch;
			while ((ch = itU8.GetCurrentAndAdvance()) != itU8.DONE)
			{
				auto jt = unicodeToBytesMapping.find(ch);
				if (jt != unicodeToBytesMapping.end())
				{
					outBytes.push_back(jt->second);
				}
				else
				{
					AppendUtf8Bytes(ch, outBytes);
				}
			}
		}
		else
		{
			outBytes.insert(outBytes.end(), token.begin(), token.end());
		}		
	}

	if (outBytes.empty())
	{
		return u8"";
	}

	return StringUtf8(outBytes.data(), outBytes.size());
}
