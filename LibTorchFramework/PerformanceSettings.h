#ifndef PERFORMANCE_SETTINGS_H
#define PERFORMANCE_SETTINGS_H


struct PerformanceSettings
{
	enum class MatMulPrecision
	{
		HIGHEST, //float32 matrix multiplications use the float32(default)
		HIGH,    //float32 matrix multiplications use the TensorFloat32 or bfloat16_3x
		MEDIUM   //float32 matrix multiplications use the bfloat16
	};

	//https://discuss.pytorch.org/t/deploy-mixed-precision-model-in-libtorch/89046/13
	//https://discuss.pytorch.org/t/equivalent-of-gradscaler-in-the-c-api/190234/2
	bool enableAutoCast = false;

	//https://discuss.pytorch.org/t/should-we-set-non-blocking-to-true/38234/4
	//whether to use non_blocking in.to() calls on tensors
	bool useNonBlockingTransfers = true;

	//use pin memory for InputLoaders
	bool usePinMemory = true;

	//todo
	//std::shared_ptr<AbstractProfiler> profiler = nullptr;

	PerformanceSettings();

	void EnableCudnn(bool val);
	void EnableCudnnFloat32(bool val);
	void SetMatMulPrec(MatMulPrecision m);

	MatMulPrecision GetMatMulPrec() const;

protected:
	bool useCudnn;
};



#endif