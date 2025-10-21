#ifndef SIMVP_V2_MODULES_H
#define SIMVP_V2_MODULES_H

#include <cmath>
#include <vector>
#include <string>

#include <torch/torch.h>

namespace ModelZoo {
	namespace SimVPv2 {
		
		// ---------------------------
		// DropPath (Stochastic Depth)
		// ---------------------------
		struct DropPathImpl : public torch::nn::Module
		{
			double drop_prob;
			bool scale_by_keep;

			DropPathImpl(double drop_prob_ = 0.0, bool scale_by_keep_ = true);

			torch::Tensor forward(const torch::Tensor& x);
		};
		TORCH_MODULE(DropPath);

		// ---------------------------
		// DWConv (Depthwise conv 3x3)
		// ---------------------------
		struct DWConvImpl : public torch::nn::Module
		{
			torch::nn::Conv2d dwconv{ nullptr };
			DWConvImpl(int64_t dim = 768);

			torch::Tensor forward(const torch::Tensor& x);
		};
		TORCH_MODULE(DWConv);

		// ---------------------------
		// Mlp (1x1 conv -> dwconv -> act -> 1x1 conv; dropout)
		// ---------------------------
		struct MlpImpl : public torch::nn::Module
		{
			torch::nn::Conv2d fc1{ nullptr };
			DWConv dwconv{ nullptr };
			// Using GELU module if available, otherwise use functional gelu in forward.
			torch::nn::GELU act{ nullptr };
			torch::nn::Conv2d fc2{ nullptr };
			torch::nn::Dropout drop{ nullptr };

			MlpImpl(int64_t in_features, int64_t hidden_features = -1, int64_t out_features = -1,
				double drop_prob = 0.0);

			torch::Tensor forward(const torch::Tensor& x);

		private:
			void _init_weights(torch::nn::Module& m);
		};
		TORCH_MODULE(Mlp);

		// ---------------------------
		// AttentionModule (Large Kernel Attention)
		// ---------------------------
		struct AttentionModuleImpl : public torch::nn::Module
		{
			// depthwise convs
			torch::nn::Conv2d conv0{ nullptr };
			torch::nn::Conv2d conv_spatial{ nullptr };
			torch::nn::Conv2d conv1{ nullptr };

			// reduction FCs
			int64_t reduction;
			torch::nn::AdaptiveAvgPool2d avg_pool{ nullptr };
			torch::nn::Linear fc1{ nullptr };
			torch::nn::ReLU relu{ nullptr };
			torch::nn::Linear fc2{ nullptr };
			torch::nn::Sigmoid sigmoid{ nullptr };

			// gate convs
			torch::nn::Conv2d conv2_0{ nullptr };
			torch::nn::Conv2d conv2_spatial{ nullptr };
			torch::nn::Conv2d conv2_1{ nullptr };

			AttentionModuleImpl(int64_t dim, int64_t kernel_size, int64_t dilation = 3);

			torch::Tensor forward(const torch::Tensor& x);
		
		};
		TORCH_MODULE(AttentionModule);

		// ---------------------------
		// SpatialAttention (wraps AttentionModule with 1x1 convs and GELU)
		// ---------------------------
		struct SpatialAttentionImpl : public torch::nn::Module {
			torch::nn::Conv2d proj_1{ nullptr };
			torch::nn::GELU activation{ nullptr };
			AttentionModule spatial_gating_unit{ nullptr };
			torch::nn::Conv2d proj_2{ nullptr };

			SpatialAttentionImpl(int64_t d_model, int64_t kernel_size = 21);

			torch::Tensor forward(const torch::Tensor& x);
		};
		TORCH_MODULE(SpatialAttention);

		// ---------------------------
		// GASubBlock (Gated Attention Sub-block)
		// ---------------------------
		struct GASubBlockImpl : public torch::nn::Module {
			torch::nn::BatchNorm2d norm1{ nullptr };
			SpatialAttention attn{ nullptr };
			DropPath drop_path{ nullptr };

			torch::nn::BatchNorm2d norm2{ nullptr };
			Mlp mlp{ nullptr };

			torch::Tensor layer_scale_1;
			torch::Tensor layer_scale_2;

			GASubBlockImpl(int64_t dim, int64_t kernel_size = 21, double mlp_ratio = 4.0,
				double drop = 0.0, double drop_path_prob = 0.1);

			torch::Tensor forward(const torch::Tensor& x);

		private:
			void _init_weights(torch::nn::Module& m);
		};
		TORCH_MODULE(GASubBlock);

	}
}

#endif // SIMVP_V2_MODULES_H
