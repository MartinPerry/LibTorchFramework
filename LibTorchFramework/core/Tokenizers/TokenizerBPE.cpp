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

void TokenizerBPE::Load()
{
	this->CreateBytesToUnicodeMapping();
	
	json->Load();

	auto split = json->GetPretokenizerType<TokenizerJsonLoader::SplitType>();
	if ((split->behavior != "Isolated") || (split->invert))
	{		
		throw std::runtime_error("Unsupported Split config: behavior=" + split->behavior);
	}
	if (split->regex.empty())
	{
		throw std::runtime_error("Split regex is empty");
	}
	
	bos.id = this->GetSpecialTokenId(bos.content);
	eos.id = this->GetSpecialTokenId(eos.content);
	
	
	splitRx = std::make_shared<UnicodeRegex>(split->regex);
	

	const auto& merges = json->GetMerges();
		
	for (int i = 0; i < merges.size(); i++)
	{
		std::u8string_view m = merges[i];
	
		auto parts = StringUtils::Split<std::u8string_view>(m, u8" ");
		if (parts.size() == 2)
		{
			auto h0 = Token::CalcHash(parts[0]);
			auto h1 = Token::CalcHash(parts[1]);

			auto it = this->bpeRanks.try_emplace(h0);
			it.first->second.try_emplace(h1, i);
		}		
	}	
	
}

TokenId TokenizerBPE::GetSpecialTokenId(const StringUtf8& token) const
{	
	const auto& vocab = json->GetVocab();
	const auto& added = json->AddedTokens();
	auto tpl = json->GetPostProcessorType<TokenizerJsonLoader::TemplateProcessingType>();

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


/// <summary>
/// Split behavior=Isolated: keep matched spans as tokens, plus any gaps.
/// Your regex usually covers the whole string, but we do it correctly anyway.
/// </summary>
/// <param name="str"></param>
/// <returns></returns>
std::vector<StringUtf8> TokenizerBPE::SplitIsolated(const StringUtf8& str)
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
	auto pieces = this->SplitIsolated(str);
	
	std::vector<TokenId> ids;

	//post_processor: TemplateProcessing prepends <|begin_of_text|>
	if ((addBos) && (bos.id != -1))
	{
		ids.push_back(bos.id);
	}

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

	if ((addEos) && (eos.id != -1))
	{
		ids.push_back(eos.id);
	}
	
	return ids;
}

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