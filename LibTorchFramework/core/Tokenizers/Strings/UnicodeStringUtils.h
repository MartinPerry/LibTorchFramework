#ifndef UNICODE_STRING_UTILS_H
#define UNICODE_STRING_UTILS_H


#include <unicode/unistr.h>
#include <unicode/schriter.h>



#include "./ICUUtils.h"

//String manipulation

/*
typedef utf8_string UnicodeString;

#define BIDI(x) x
#define UTF8_TEXT(x) x
#define UTF8_UNESCAPE(x) utf8_string::build_from_escaped(x.c_str())
*/


#ifdef USE_ICU_LIBRARY

typedef icu::UnicodeString UnicodeString;
typedef icu::StringCharacterIterator UnicodeCharacterPtr;

#	define BIDI(x) BidiHelper::ConvertOneLine(x)

#   define NEED_BIDI(x) IcuUtils::RequiresBidi(x)

//may need reinterpret_cast<const char*>(x) in C++20 for u8"" strings
//https://stackoverflow.com/questions/57402464/is-c20-char8-t-the-same-as-our-old-char
#	define UTF8_TEXT(x) icu::UnicodeString::fromUTF8(x)

#	define UTF8_UNESCAPE(x) icu::UnicodeString::fromUTF8(x).unescape()

#endif


/*
Unicode -> UTF8
"è" 
Unicode: U+010D
UTF8 [0xC4, 0x8D]  ->  [196, 141]

b = s.encode("utf-8")


*/



#endif


