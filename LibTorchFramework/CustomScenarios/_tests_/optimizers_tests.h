#pragma once


#include <cstdint>
#include <memory>

namespace CustomScenarios
{
	namespace _tests_
	{
		void test_matches_adamw_when_quant_off();

		void test_loss_decreases_toy_regression_adamw8();
		void test_loss_decreases_toy_regression_fused_adamw8();
		
	}
}