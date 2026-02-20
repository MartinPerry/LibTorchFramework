#include "./UnicodeRegex.h"

#include <stdexcept>

#include <unicode/regex.h>
#include <unicode/unistr.h>
#include <unicode/utypes.h>

#ifdef _DEBUG
#   pragma comment(lib, "icudtd.lib")
#   pragma comment(lib, "icuucd.lib")
#   pragma comment(lib, "icuind.lib")
#else
#   pragma comment(lib, "icudt.lib")
#   pragma comment(lib, "icuuc.lib")
#   pragma comment(lib, "icuin.lib")
#endif

UnicodeRegex::UnicodeRegex(const std::u8string& pattern)
{
    // Raw string literal to avoid double-escaping.
    // NOTE: The pattern below is the *logical* regex (not JSON-escaped).
    icu::UnicodeString patternUni = icu::UnicodeString::fromUTF8((const char*)(pattern.c_str()));
        
    UErrorCode status = U_ZERO_ERROR;
    icu::RegexPattern* tmp = icu::RegexPattern::compile(patternUni, 0, status);
    
    compiled = std::unique_ptr<icu::RegexPattern>(tmp);
 
    if (U_FAILURE(status) || !compiled) 
    {
        throw std::runtime_error("Regex compilation failed");
    }
}

UnicodeRegex::~UnicodeRegex()
{
}

std::vector<std::string> UnicodeRegex::Run(const std::u8string& utf8)
{
    
    icu::UnicodeString text = icu::UnicodeString::fromUTF8(
        icu::StringPiece(reinterpret_cast<const char*>(utf8.data()), utf8.size())
    );

    UErrorCode status = U_ZERO_ERROR;
    std::unique_ptr<icu::RegexMatcher> m(compiled->matcher(text, status));
    if (U_FAILURE(status) || !m) 
    {
        return {};
    }

    std::vector<std::string> out;
    while (m->find(status) && U_SUCCESS(status)) 
    {
        icu::UnicodeString tok = m->group(status);
        std::string tok8;
        tok.toUTF8String(tok8);
        out.push_back(std::move(tok8));
    }
    return out;
}

std::vector<UnicodeRegex::MatchSpanUtf8> UnicodeRegex::FindSpans(const std::u8string& utf8)
{
    UErrorCode status = U_ZERO_ERROR;

    icu::UnicodeString utext = icu::UnicodeString::fromUTF8(
        icu::StringPiece(reinterpret_cast<const char*>(utf8.data()), utf8.size())
    );
    

    std::unique_ptr<icu::RegexMatcher> m(compiled->matcher(utext, status));
    if (U_FAILURE(status) || !m) {
        throw std::runtime_error("ICU: RegexPattern::matcher failed");
    }

    std::vector<MatchSpanUtf8> out;

    while (m->find(status) && U_SUCCESS(status)) 
    {
        const int32_t ua = m->start(status); // UTF-16 code unit index
        const int32_t ub = m->end(status);   // UTF-16 code unit index (exclusive)
        if (U_FAILURE(status))
        {
            break;
        }

        // Convert UTF-16 indices -> UTF-8 byte offsets in the ORIGINAL utf8 string
        // by slicing the ICU string up to those indices and measuring UTF-8 length.
        std::string prefixA, prefixB, token;
        utext.tempSubStringBetween(0, ua).toUTF8String(prefixA);
        utext.tempSubStringBetween(0, ub).toUTF8String(prefixB);
        utext.tempSubStringBetween(ua, ub).toUTF8String(token);

        MatchSpanUtf8 ms;
        ms.a = prefixA.size();
        ms.b = prefixB.size();
        ms.s = std::move(token);
        out.push_back(std::move(ms));
    }

    if (U_FAILURE(status)) 
    {
        throw std::runtime_error("ICU: RegexMatcher::find failed");
    }

    return out;
}
