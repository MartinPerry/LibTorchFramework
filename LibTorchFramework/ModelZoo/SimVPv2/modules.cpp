#include "modules.h"

#include "../../core/Modules/WeightsInit/TruncatedNormalInit.h"

#include "../../Utils/TorchUtils.h"

using namespace ModelZoo::SimVPv2;

// ======================================================================================
// DropPathImpl
// ======================================================================================

DropPathImpl::DropPathImpl(float drop_prob_, bool scale_by_keep_) :
	drop_prob(drop_prob_),
	scale_by_keep(scale_by_keep_)
{
	// nothing to register
}

torch::Tensor DropPathImpl::forward(const torch::Tensor& x)
{
	if (drop_prob == 0.0 || !this->is_training())
	{
		return x;
	}

	float keep_prob = 1.0f - drop_prob;
	// shape = (batch, 1, 1, 1, ...) matching x.ndimension()
	std::vector<int64_t> shape;
	shape.push_back(x.size(0));
	for (int i = 1; i < x.dim(); ++i)
	{
		shape.push_back(1);
	}

	auto random_tensor = torch::empty(shape, x.options()).bernoulli_(keep_prob);
	if (keep_prob > 0.0 && scale_by_keep)
	{
		random_tensor = random_tensor.div(keep_prob);
	}
	return x * random_tensor;
}

// ======================================================================================
// DWConvImpl
// ======================================================================================

DWConvImpl::DWConvImpl(int64_t dim)
{
	torch::nn::Conv2dOptions opts = torch::nn::Conv2dOptions(dim, dim, /*kernel_size=*/3)
		.stride(1).padding(1).bias(true).groups(dim);

	AUTO_REGISTER_NEW_MODULE(dwconv, torch::nn::Conv2d(opts));
}

torch::Tensor DWConvImpl::forward(const torch::Tensor& x)
{
	return dwconv->forward(x);
}

// ======================================================================================
// MlpImpl
// ======================================================================================

MlpImpl::MlpImpl(int64_t in_features, int64_t hidden_features, int64_t out_features, float drop_prob)
{
	if (out_features == -1)
	{
		out_features = in_features;
	}
	if (hidden_features == -1)
	{
		hidden_features = in_features;
	}

	// 1x1 conv: emulate Linear with conv kernel=1
	AUTO_REGISTER_NEW_MODULE(fc1, torch::nn::Conv2d(torch::nn::Conv2dOptions(in_features, hidden_features, 1).stride(1).padding(0)));
	AUTO_REGISTER_NEW_MODULE(dwconv, DWConv(hidden_features));
	AUTO_REGISTER_NEW_MODULE(act, torch::nn::GELU());
	AUTO_REGISTER_NEW_MODULE(fc2, torch::nn::Conv2d(torch::nn::Conv2dOptions(hidden_features, out_features, 1).stride(1).padding(0)));
	AUTO_REGISTER_NEW_MODULE(drop, torch::nn::Dropout(torch::nn::DropoutOptions(drop_prob)));
	
	
	this->apply([&](torch::nn::Module& m){
		this->_init_weights(m);
	});	
	
}


void MlpImpl::_init_weights(torch::nn::Module& m)
{
	if (auto* linear = dynamic_cast<torch::nn::LinearImpl*>(&m))
	{
		TruncatedNormalInit::trunc_normal(linear->weight, 0.0, 0.02, -2.0, 2.0); // std = 0.02

		if (linear->bias.defined())
		{
			torch::nn::init::constant_(linear->bias, 0.0f);
		}
	}
	else if (auto* layerNorm = dynamic_cast<torch::nn::LayerNormImpl*>(&m))
	{				
		if (layerNorm->bias.defined())
		{
			torch::nn::init::constant_(layerNorm->bias, 0.0f);
		}
		if (layerNorm->weight.defined())
		{
			torch::nn::init::constant_(layerNorm->weight, 1.0f);
		}
	}
	else if (auto* conv2d = dynamic_cast<torch::nn::Conv2dImpl*>(&m))
	{				
		auto ks = conv2d->options.kernel_size();
		int64_t fan_out = ks->at(0) * ks->at(1) * conv2d->options.out_channels();
		fan_out /= conv2d->options.groups();

		conv2d->weight.data().normal_(0.0f, std::sqrt(2.0f / static_cast<float>(fan_out)));

		if (conv2d->bias.defined())
		{
			conv2d->bias.data().zero_();
		}		
	}
	else
	{
		// TODO: handle other layer types if needed
	}
}


torch::Tensor MlpImpl::forward(const torch::Tensor& x)
{
	auto y = fc1->forward(x);
	y = dwconv->forward(y);
	y = act->forward(y);
	y = drop->forward(y);
	y = fc2->forward(y);
	y = drop->forward(y);
	return y;
}

// ======================================================================================
// AttentionModuleImpl
// ======================================================================================

AttentionModuleImpl::AttentionModuleImpl(int64_t dim, int64_t kernel_size, int64_t dilation)	
{

	//avg_pool = register_module("avg_pool", torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions(1)));
	//relu = register_module("relu", torch::nn::ReLU(true));
	//sigmoid = register_module("sigmoid", torch::nn::Sigmoid());

	int64_t d_k = 2 * dilation - 1;
	int64_t d_p = (d_k - 1) / 2;
	// Compute dd_k as in Python: kernel_size // dilation + ((kernel_size // dilation) % 2 - 1)
	int64_t dd_k = kernel_size / dilation + ((kernel_size / dilation) % 2 - 1);
	if (dd_k < 1)
	{
		dd_k = 1;
	}
	int64_t dd_p = (dilation * (dd_k - 1) / 2);

	// depthwise convs
	AUTO_REGISTER_NEW_MODULE(conv0, torch::nn::Conv2d(torch::nn::Conv2dOptions(dim, dim, d_k).padding(d_p).groups(dim)));
	AUTO_REGISTER_NEW_MODULE(conv_spatial, torch::nn::Conv2d(torch::nn::Conv2dOptions(dim, dim, dd_k).stride(1).padding(dd_p).groups(dim).dilation(dilation)));
	AUTO_REGISTER_NEW_MODULE(conv1, torch::nn::Conv2d(torch::nn::Conv2dOptions(dim, dim * 2, 1)));

	/*
	reduction = std::max<int64_t>(dim / 16, 4);
	fc1 = register_module("fc1", torch::nn::Linear(dim, dim / reduction, false));
	fc2 = register_module("fc2", torch::nn::Linear(dim / reduction, dim, false));

	// gate convs (they mirror conv0/conv_spatial/conv1 with different out-channels)
	conv2_0 = register_module("conv2_0", torch::nn::Conv2d(torch::nn::Conv2dOptions(dim, dim, d_k).padding(d_p).groups(dim)));
	conv2_spatial = register_module("conv2_spatial", torch::nn::Conv2d(torch::nn::Conv2dOptions(dim, dim, dd_k).stride(1).padding(dd_p).groups(dim).dilation(dilation)));
	conv2_1 = register_module("conv2_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(dim, dim, 1)));
	*/
}

torch::Tensor AttentionModuleImpl::forward(const torch::Tensor& x)
{
	// u = x.clone()  // Python used clone but not used later; keep op cheap
	auto attn = conv0->forward(x);
	attn = conv_spatial->forward(attn);

	auto f_g = conv1->forward(attn);
	// split channels
	int64_t split_dim = f_g.size(1) / 2;
	auto slices = f_g.split_with_sizes({ split_dim, split_dim }, 1);
	auto f_x = slices[0];
	auto g_x = slices[1];

	// gate: sigmoid(g_x) * f_x
	return torch::sigmoid(g_x) * f_x;
}

// ======================================================================================
// SpatialAttentionImpl
// ======================================================================================

SpatialAttentionImpl::SpatialAttentionImpl(int64_t d_model, int64_t kernel_size)	
{	
	AUTO_REGISTER_NEW_MODULE(proj_1, torch::nn::Conv2d(torch::nn::Conv2dOptions(d_model, d_model, 1)));
	AUTO_REGISTER_NEW_MODULE(spatial_gating_unit, AttentionModule(d_model, kernel_size));
	AUTO_REGISTER_NEW_MODULE(proj_2, torch::nn::Conv2d(torch::nn::Conv2dOptions(d_model, d_model, 1)));

	AUTO_REGISTER_NEW_MODULE(activation, torch::nn::GELU());
	
}

torch::Tensor SpatialAttentionImpl::forward(const torch::Tensor& x)
{
	auto shortcut = x;
	auto y = proj_1->forward(x);
	y = activation->forward(y);
	y = spatial_gating_unit->forward(y);
	y = proj_2->forward(y);
	y = y + shortcut;
	return y;
}

// ======================================================================================
// GASubBlockImpl
// ======================================================================================

GASubBlockImpl::GASubBlockImpl(int64_t dim, int64_t kernel_size, float mlp_ratio, float drop, float drop_path_prob)
{
	AUTO_REGISTER_NEW_MODULE(norm1, torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(dim)));
	AUTO_REGISTER_NEW_MODULE(attn, SpatialAttention(dim, kernel_size));
	if (drop_path_prob > 0.0f)
	{
		 AUTO_REGISTER_NEW_MODULE(drop_path, DropPath(drop_path_prob));
	}
	else
	{
		AUTO_REGISTER_NEW_MODULE(drop_path, DropPath(0.0f));
	}

	AUTO_REGISTER_NEW_MODULE(norm2, torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(dim)));
	int64_t mlp_hidden_dim = static_cast<int64_t>(dim * mlp_ratio);
	AUTO_REGISTER_NEW_MODULE(mlp, Mlp(dim, mlp_hidden_dim, -1, drop));

	// layer scale parameters (initialized to small constant)
	float init_val = static_cast<float>(1e-2);
	layer_scale_1 = register_parameter("layer_scale_1", torch::full({ dim }, init_val));
	layer_scale_2 = register_parameter("layer_scale_2", torch::full({ dim }, init_val));

	this->apply([&](torch::nn::Module& m) {
		this->_init_weights(m);
	});
}

void GASubBlockImpl::_init_weights(torch::nn::Module& m)
{
	if (auto* linear = dynamic_cast<torch::nn::LinearImpl*>(&m))
	{
		// Truncated normal initialization
		TruncatedNormalInit::trunc_normal(linear->weight, 0.0, 0.02, -2.0, 2.0);

		if (linear->bias.defined())
		{
			torch::nn::init::constant_(linear->bias, 0);
		}
	}
	else if (auto* layerNorm = dynamic_cast<torch::nn::LayerNormImpl*>(&m))
	{
		if (layerNorm->bias.defined())
		{
			torch::nn::init::constant_(layerNorm->bias, 0);
		}
		if (layerNorm->weight.defined())
		{
			torch::nn::init::constant_(layerNorm->weight, 1.0);
		}
	}
	else if (auto* conv2d = dynamic_cast<torch::nn::Conv2dImpl*>(&m))
	{
		auto ks = conv2d->options.kernel_size();
		int64_t fan_out = ks->at(0) * ks->at(1) * conv2d->options.out_channels();		
		fan_out /= conv2d->options.groups();

		conv2d->weight.data().normal_(0, std::sqrt(2.0f / static_cast<float>(fan_out)));

		if (conv2d->bias.defined())
		{
			conv2d->bias.data().zero_();
		}
	}
	else
	{
		// other module types can be added here if needed
	}
}

torch::Tensor GASubBlockImpl::forward(const torch::Tensor& x)
{
	// norm1 expects (N,C,H,W). The Python used BatchNorm2d same
	auto a = norm1->forward(x);
	auto attn_out = attn->forward(a);
	// layer_scale_1: shape (C) -> (C,1,1) then broadcast
	auto scale1 = layer_scale_1.view({ 1, -1, 1, 1 });
	auto after1 = x + drop_path->forward(scale1 * attn_out);

	auto b = norm2->forward(after1);
	auto mlp_out = mlp->forward(b);
	auto scale2 = layer_scale_2.view({ 1, -1, 1, 1 });
	auto out = after1 + drop_path->forward(scale2 * mlp_out);
	return out;
}

