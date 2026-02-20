#ifndef TOKENIZERS_H
#define TOKENIZERS_H

#include <cstdint>
#include <string>
#include <unordered_map>
#include <memory>
#include <vector>

#include <Utils/3rdParty/xxhash.hpp>

#include <Utils/Strings/StringIterators.h>
#include <Utils/Strings/Utf8Utils.h>

//=============================================================
//Tokens:
//Example: (the weird G) means this token begins with a space.
//UTF8 -> 0xC4 0xA0 (2 byte - 196, 160)
//UTF32 -> 0x00000120
//Unicode -> U+0120

// std::u8string are internally stored in UTF8
// std::string is stored as it is

using TokenId = int32_t; //use signed type - we use invalid token as negative
using UnicodeCodePoint = char32_t;
//using StringUtf8 = std::u8string;
using StringUtf8Hash = uint64_t;

using TokenMap = std::unordered_map<StringUtf8, TokenId>;
using ReverseTokenMap = std::unordered_map<TokenId, StringUtf8>;
using TokenHashMap = std::unordered_map<StringUtf8Hash, TokenId>;

//=============================================================

struct Token
{
	template <typename T>
	static StringUtf8Hash CalcHash(const T& s)
	{
		std::vector<UnicodeCodePoint> buf;
		buf.reserve(s.length());

		auto it = CustomU8Iterator(s);
		UnicodeCodePoint v;
		while ((v = it.GetCurrentAndAdvance()) != it.DONE)
		{
			buf.push_back(v);
		}

		return Token::CalcHash(buf);
	}

	static StringUtf8Hash CalcHash(const std::vector<UnicodeCodePoint>& s)
	{
		return Token::CalcHash(s.data(), s.size());		
	}

	static StringUtf8Hash CalcHash(const UnicodeCodePoint* s, size_t count)
	{
		return xxh::xxhash<64>(s, count * sizeof(UnicodeCodePoint));
	}

	static StringUtf8Hash CalcHash(UnicodeCodePoint c)
	{
		return Token::CalcHash(&c, 1);
	}

	StringUtf8 content;
	TokenId id;

	Token(const char8_t* content, TokenId id) :
		content(content),
		id(id)
	{
	}

	Token(const StringUtf8& content, TokenId id) :
		content(content),
		id(id)
	{
	}

	Token(Token&& other) noexcept :
		content(std::exchange(other.content, content)),
		id(other.id)
	{
	}
};

#endif
