#include "./setup_llama.h"

//=========================================================
// Core
//=========================================================

#include "../../core/Structures.h"
#include "../../core/Runner.h"
#include "../../core/Trainer.h"
#include "../../core/AbstractModel.h"

#include "../../core/Metrics/PredictionEvaluators.h"
#include "../../core/Metrics/MetricsDefault.h"
#include "../../core/Metrics/MetricsImage.h"

#include "../../core/Modules/LossFunctions/DiceLoss.h"
#include "../../core/Modules/LossFunctions/MultiBceLoss.h"

#include "../../core/Snapshot/PretrainedManager.h"
#include "../../core/Snapshot/SnapshotSaver.h"
#include "../../core/Snapshot/SnapshotLoader.h"

#include "../../core/Tokenizers/Tokenizers.h"
#include "../../core/Tokenizers/TokenizerBPE.h"

//=========================================================
// Inputs
//=========================================================

#include "../../InputProcessing/DefaultDataset.h"
#include "../../InputProcessing/InputLoadersWrapper.h"
#include "../../InputProcessing/InputLoader.h"
#include "../../InputProcessing/DataLoaderData.h"

#include "../../InputProcessing/InputLoaders/SegmentationInputLoader.h"

//=========================================================
// ModelZoo
//=========================================================

#include "../../ModelZoo/LLMs/llama.h"
#include "../../ModelZoo/LLMs/LLamaSafeTensorLoader.h"

//=========================================================
// Utils
//=========================================================

#include "../../Utils/TorchUtils.h"
#include "../../Utils/TorchImageUtils.h"
#include "../../Utils/TrainingHelper.h"

//=========================================================

using namespace ModelZoo::llama;


namespace CustomScenarios::LLMs::Llama
{
	void setup()
	{
		std::string modelDir = "e:\\Programming\\Python\\test\\PythonApplication1\\py_cpt\\Llama-3.2-3B-Instruct\\";

		LlamaConfig cfg = LlamaConfig::FromJsonFile(modelDir + "config.json");

		LlamaForCausalLM llama(cfg);

		LLamaSafeTensorLoader tl;
		tl.LoadFromHfSafetensors(llama, modelDir);

		auto bpe = TokenizerBPE("d://tokenizer.json");
		bpe.Load();

	}
}