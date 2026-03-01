#pragma once

class TokenizerBPE;

#include <cstdint>
#include <memory>

namespace CustomScenarios
{
	namespace _tests_
	{
		void RunBpeJsonTests(const char* jsonPath, TokenizerBPE& tok);
	}
}