#ifndef UNICODE_REGEX_H
#define UNICODE_REGEX_H


#include <memory>
#include <string>
#include <vector>

#include <unicode/regex.h>


class UnicodeRegex
{
public:
	struct MatchSpanUtf8 
	{
		size_t a;      // start byte offset in original UTF-8 string
		size_t b;      // end byte offset (exclusive) in original UTF-8 string
		std::string s; // matched substring (UTF-8)
	};

	UnicodeRegex(const std::u8string& pattern);
	~UnicodeRegex();

	std::vector<std::string> Run(const std::u8string& utf8);
	std::vector<MatchSpanUtf8> FindSpans(const std::u8string& utf8);

protected:

	std::unique_ptr<icu::RegexPattern> compiled;	
};

#endif