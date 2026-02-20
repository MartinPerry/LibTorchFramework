#include "./TokenizerBPE.h"

#include <stdexcept>
#include <iostream>

#include "./Strings/UnicodeRegex.h"

#include <Utils/Strings/StringUtils.h>
#include <Utils/Strings/StringIterators.h>

TokenizerBPE::TokenizerBPE(const std::string& jsonPath) : 
	json(std::make_shared<TokenizerJsonLoader>(jsonPath)),
	bos(u8"<|begin_of_text|>", -1),
	eos(u8"<|end_of_text|>", -1)	
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

	auto split = json->GetPretokenizerType<TokenizerJsonLoader::SplitType>();
	if (split)
	{
		//if ((split->behavior != "Isolated") || (split->invert))
		//{
		//	throw std::runtime_error("Unsupported Split config: behavior=" + split->behavior);
		//}
		

		if (split->splitType == TokenizerJsonLoader::SplitType::SplitDataType::Regex)
		{
			splitRx = std::make_shared<UnicodeRegex>(split->splitData);
		}
		else if (split->splitType == TokenizerJsonLoader::SplitType::SplitDataType::String)
		{
			splitStr = split->splitData;
		}
	}
	
	bos.id = this->GetSpecialTokenId(bos.content);
	eos.id = this->GetSpecialTokenId(eos.content);
	
	
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
	if (tpl == nullptr)
	{
		return -1;
	}

	auto it = tpl->special.find(token);
	if (it != tpl->special.end())
	{
		return it->second[0];
	}
	else
	{
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
	if (this->splitRx)
	{
		return this->SplitIsolatedRegex(str);
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
					ids.push_back(0);
				}
			}
		}
	}
	
	return ids;
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
	//auto pieces = this->SplitIsolated(str);
	
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

			std::vector<UnicodeCodePoint> unicodes;

			for (auto b : p)
			{
				unicodes.push_back(this->bytesToUnicodeMapping[b]);
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

StringUtf8 TokenizerBPE::Decode(const std::vector<TokenId>& ids)
{
	const auto& revVocab = json->GetVocabReversed();
	
	//ids->token strings->unicode byte stream->bytes->utf - 8
	
	StringUtf8 tmp = u8"";
	for (auto id : ids)
	{
		auto it = revVocab.find(id);
		if (it != revVocab.end())
		{
			tmp.append(it->second.data(), it->second.length());
		}		
	}

	CustomU8Iterator it(tmp);	
	UnicodeCodePoint ch;

	std::vector<char8_t> data;
	while ((ch = it.GetCurrentAndAdvance()) != it.DONE)
	{
		data.push_back(unicodeToBytesMapping[ch]);
	}
	data.push_back(0);

	return StringUtf8(data.data(), data.size());
}