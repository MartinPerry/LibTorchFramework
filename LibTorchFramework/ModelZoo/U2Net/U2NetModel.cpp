#include "./U2NetModel.h"

#include <Utils/Logger.h>

#include "../../InputProcessing/DataLoaderData.h"

using namespace torch::nn;
using namespace ModelZoo::u2net;

// ======================================================================================
// Helper function: upsample one tensor to match another
// ======================================================================================

static torch::Tensor upsample_like(const torch::Tensor& src, const torch::Tensor& tar) 
{
    // compute target spatial size
    auto sizes = tar.sizes();
    int64_t h = sizes[2];
    int64_t w = sizes[3];
    return torch::nn::functional::interpolate(src,
        torch::nn::functional::InterpolateFuncOptions()
        .size(std::vector<int64_t>({ h, w }))
        .mode(torch::kBilinear)
        .align_corners(false));
}

// ======================================================================================
// REBNCONV
// ======================================================================================

REBNCONVImpl::REBNCONVImpl(int in_ch, int out_ch, int dirate, ConvType convType) 
{
    // Note: Python code used bias=False for classic conv because followed by BatchNorm.
    bool use_bias = false;
    
    // For DEFORMABLE / COORD, in this translation we fall back to normal conv;
    // replace with actual implementations if available.
    if (convType == ConvType::DEFORMABLE || convType == ConvType::COORD) 
    {
        use_bias = true; // match original python where custom conv used bias=True
        // TODO: replace with DeformConv2d or CoordConv2d implementations if you have them
        MY_LOG_ERROR("Not supported");
    }

    conv = register_module("conv", Conv2d(Conv2dOptions(in_ch, out_ch, 3).padding(dirate).dilation(dirate).bias(use_bias)));
    bn = register_module("bn", BatchNorm2d(out_ch));
    relu = register_module("relu", ReLU(ReLUOptions().inplace(true)));
}

torch::Tensor REBNCONVImpl::forward(const torch::Tensor& x) 
{
    auto hx = x;
    auto y = conv->forward(hx);
    y = bn->forward(y);
    y = relu->forward(y);
    return y;
}


// ======================================================================================
// RSU7
// ======================================================================================

RSU7Impl::RSU7Impl(int in_ch, int mid_ch, int out_ch, ConvType convType) 
{
    rebnconvin = register_module("rebnconvin", REBNCONV(in_ch, out_ch, 1, convType));

    rebnconv1 = register_module("rebnconv1", REBNCONV(out_ch, mid_ch, 1, convType));
    pool1 = register_module("pool1", MaxPool2d(MaxPool2dOptions(2).stride(2).ceil_mode(true)));

    rebnconv2 = register_module("rebnconv2", REBNCONV(mid_ch, mid_ch, 1, convType));
    pool2 = register_module("pool2", MaxPool2d(MaxPool2dOptions(2).stride(2).ceil_mode(true)));

    rebnconv3 = register_module("rebnconv3", REBNCONV(mid_ch, mid_ch, 1, convType));
    pool3 = register_module("pool3", MaxPool2d(MaxPool2dOptions(2).stride(2).ceil_mode(true)));

    rebnconv4 = register_module("rebnconv4", REBNCONV(mid_ch, mid_ch, 1, convType));
    pool4 = register_module("pool4", MaxPool2d(MaxPool2dOptions(2).stride(2).ceil_mode(true)));

    rebnconv5 = register_module("rebnconv5", REBNCONV(mid_ch, mid_ch, 1, convType));
    pool5 = register_module("pool5", MaxPool2d(MaxPool2dOptions(2).stride(2).ceil_mode(true)));

    rebnconv6 = register_module("rebnconv6", REBNCONV(mid_ch, mid_ch, 1, convType));
    rebnconv7 = register_module("rebnconv7", REBNCONV(mid_ch, mid_ch, 2, convType));

    rebnconv6d = register_module("rebnconv6d", REBNCONV(mid_ch * 2, mid_ch, 1, convType));
    rebnconv5d = register_module("rebnconv5d", REBNCONV(mid_ch * 2, mid_ch, 1, convType));
    rebnconv4d = register_module("rebnconv4d", REBNCONV(mid_ch * 2, mid_ch, 1, convType));
    rebnconv3d = register_module("rebnconv3d", REBNCONV(mid_ch * 2, mid_ch, 1, convType));
    rebnconv2d = register_module("rebnconv2d", REBNCONV(mid_ch * 2, mid_ch, 1, convType));
    rebnconv1d = register_module("rebnconv1d", REBNCONV(mid_ch * 2, out_ch, 1, convType));
}

torch::Tensor RSU7Impl::forward(const torch::Tensor& x) 
{
    auto hx = x;
    auto hxin = rebnconvin->forward(hx);

    auto hx1 = rebnconv1->forward(hxin);
    hx = pool1->forward(hx1);

    auto hx2 = rebnconv2->forward(hx);
    hx = pool2->forward(hx2);

    auto hx3 = rebnconv3->forward(hx);
    hx = pool3->forward(hx3);

    auto hx4 = rebnconv4->forward(hx);
    hx = pool4->forward(hx4);

    auto hx5 = rebnconv5->forward(hx);
    hx = pool5->forward(hx5);

    auto hx6 = rebnconv6->forward(hx);

    auto hx7 = rebnconv7->forward(hx6);

    auto hx6d = rebnconv6d->forward(torch::cat({ hx7, hx6 }, 1));
    auto hx6dup = upsample_like(hx6d, hx5);

    auto hx5d = rebnconv5d->forward(torch::cat({ hx6dup, hx5 }, 1));
    auto hx5dup = upsample_like(hx5d, hx4);

    auto hx4d = rebnconv4d->forward(torch::cat({ hx5dup, hx4 }, 1));
    auto hx4dup = upsample_like(hx4d, hx3);

    auto hx3d = rebnconv3d->forward(torch::cat({ hx4dup, hx3 }, 1));
    auto hx3dup = upsample_like(hx3d, hx2);

    auto hx2d = rebnconv2d->forward(torch::cat({ hx3dup, hx2 }, 1));
    auto hx2dup = upsample_like(hx2d, hx1);

    auto hx1d = rebnconv1d->forward(torch::cat({ hx2dup, hx1 }, 1));

    return hx1d + hxin;
}

// ======================================================================================
// RSU6
// ======================================================================================

RSU6Impl::RSU6Impl(int in_ch, int mid_ch, int out_ch, ConvType convType) 
{
    rebnconvin = register_module("rebnconvin", REBNCONV(in_ch, out_ch, 1, convType));

    rebnconv1 = register_module("rebnconv1", REBNCONV(out_ch, mid_ch, 1, convType));
    pool1 = register_module("pool1", MaxPool2d(MaxPool2dOptions(2).stride(2).ceil_mode(true)));

    rebnconv2 = register_module("rebnconv2", REBNCONV(mid_ch, mid_ch, 1, convType));
    pool2 = register_module("pool2", MaxPool2d(MaxPool2dOptions(2).stride(2).ceil_mode(true)));

    rebnconv3 = register_module("rebnconv3", REBNCONV(mid_ch, mid_ch, 1, convType));
    pool3 = register_module("pool3", MaxPool2d(MaxPool2dOptions(2).stride(2).ceil_mode(true)));

    rebnconv4 = register_module("rebnconv4", REBNCONV(mid_ch, mid_ch, 1, convType));
    pool4 = register_module("pool4", MaxPool2d(MaxPool2dOptions(2).stride(2).ceil_mode(true)));

    rebnconv5 = register_module("rebnconv5", REBNCONV(mid_ch, mid_ch, 1, convType));
    rebnconv6 = register_module("rebnconv6", REBNCONV(mid_ch, mid_ch, 2, convType));

    rebnconv5d = register_module("rebnconv5d", REBNCONV(mid_ch * 2, mid_ch, 1, convType));
    rebnconv4d = register_module("rebnconv4d", REBNCONV(mid_ch * 2, mid_ch, 1, convType));
    rebnconv3d = register_module("rebnconv3d", REBNCONV(mid_ch * 2, mid_ch, 1, convType));
    rebnconv2d = register_module("rebnconv2d", REBNCONV(mid_ch * 2, mid_ch, 1, convType));
    rebnconv1d = register_module("rebnconv1d", REBNCONV(mid_ch * 2, out_ch, 1, convType));
}

torch::Tensor RSU6Impl::forward(const torch::Tensor& x) 
{
    auto hx = x;
    auto hxin = rebnconvin->forward(hx);

    auto hx1 = rebnconv1->forward(hxin);
    hx = pool1->forward(hx1);

    auto hx2 = rebnconv2->forward(hx);
    hx = pool2->forward(hx2);

    auto hx3 = rebnconv3->forward(hx);
    hx = pool3->forward(hx3);

    auto hx4 = rebnconv4->forward(hx);
    hx = pool4->forward(hx4);

    auto hx5 = rebnconv5->forward(hx);

    auto hx6 = rebnconv6->forward(hx5);

    auto hx5d = rebnconv5d->forward(torch::cat({ hx6, hx5 }, 1));
    auto hx5dup = upsample_like(hx5d, hx4);

    auto hx4d = rebnconv4d->forward(torch::cat({ hx5dup, hx4 }, 1));
    auto hx4dup = upsample_like(hx4d, hx3);

    auto hx3d = rebnconv3d->forward(torch::cat({ hx4dup, hx3 }, 1));
    auto hx3dup = upsample_like(hx3d, hx2);

    auto hx2d = rebnconv2d->forward(torch::cat({ hx3dup, hx2 }, 1));
    auto hx2dup = upsample_like(hx2d, hx1);

    auto hx1d = rebnconv1d->forward(torch::cat({ hx2dup, hx1 }, 1));

    return hx1d + hxin;
}

// ======================================================================================
// RSU5
// ======================================================================================

RSU5Impl::RSU5Impl(int in_ch, int mid_ch, int out_ch, ConvType convType) 
{
    rebnconvin = register_module("rebnconvin", REBNCONV(in_ch, out_ch, 1, convType));

    rebnconv1 = register_module("rebnconv1", REBNCONV(out_ch, mid_ch, 1, convType));
    pool1 = register_module("pool1", MaxPool2d(MaxPool2dOptions(2).stride(2).ceil_mode(true)));

    rebnconv2 = register_module("rebnconv2", REBNCONV(mid_ch, mid_ch, 1, convType));
    pool2 = register_module("pool2", MaxPool2d(MaxPool2dOptions(2).stride(2).ceil_mode(true)));

    rebnconv3 = register_module("rebnconv3", REBNCONV(mid_ch, mid_ch, 1, convType));
    pool3 = register_module("pool3", MaxPool2d(MaxPool2dOptions(2).stride(2).ceil_mode(true)));

    rebnconv4 = register_module("rebnconv4", REBNCONV(mid_ch, mid_ch, 1, convType));
    rebnconv5 = register_module("rebnconv5", REBNCONV(mid_ch, mid_ch, 2, convType));

    rebnconv4d = register_module("rebnconv4d", REBNCONV(mid_ch * 2, mid_ch, 1, convType));
    rebnconv3d = register_module("rebnconv3d", REBNCONV(mid_ch * 2, mid_ch, 1, convType));
    rebnconv2d = register_module("rebnconv2d", REBNCONV(mid_ch * 2, mid_ch, 1, convType));
    rebnconv1d = register_module("rebnconv1d", REBNCONV(mid_ch * 2, out_ch, 1, convType));
}

torch::Tensor RSU5Impl::forward(const torch::Tensor& x) 
{
    auto hxin = rebnconvin->forward(x);

    auto hx1 = rebnconv1->forward(hxin);
    auto hx = pool1->forward(hx1);

    auto hx2 = rebnconv2->forward(hx);
    hx = pool2->forward(hx2);

    auto hx3 = rebnconv3->forward(hx);
    hx = pool3->forward(hx3);

    auto hx4 = rebnconv4->forward(hx);
    auto hx5 = rebnconv5->forward(hx4);

    auto hx4d = rebnconv4d->forward(torch::cat({ hx5, hx4 }, 1));
    auto hx4dup = upsample_like(hx4d, hx3);

    auto hx3d = rebnconv3d->forward(torch::cat({ hx4dup, hx3 }, 1));
    auto hx3dup = upsample_like(hx3d, hx2);

    auto hx2d = rebnconv2d->forward(torch::cat({ hx3dup, hx2 }, 1));
    auto hx2dup = upsample_like(hx2d, hx1);

    auto hx1d = rebnconv1d->forward(torch::cat({ hx2dup, hx1 }, 1));

    return hx1d + hxin;
}

// ======================================================================================
// RSU4
// ======================================================================================

RSU4Impl::RSU4Impl(int in_ch, int mid_ch, int out_ch, ConvType convType) 
{
    rebnconvin = register_module("rebnconvin", REBNCONV(in_ch, out_ch, 1, convType));

    rebnconv1 = register_module("rebnconv1", REBNCONV(out_ch, mid_ch, 1, convType));
    pool1 = register_module("pool1", MaxPool2d(MaxPool2dOptions(2).stride(2).ceil_mode(true)));

    rebnconv2 = register_module("rebnconv2", REBNCONV(mid_ch, mid_ch, 1, convType));
    pool2 = register_module("pool2", MaxPool2d(MaxPool2dOptions(2).stride(2).ceil_mode(true)));

    rebnconv3 = register_module("rebnconv3", REBNCONV(mid_ch, mid_ch, 1, convType));
    rebnconv4 = register_module("rebnconv4", REBNCONV(mid_ch, mid_ch, 2, convType));

    rebnconv3d = register_module("rebnconv3d", REBNCONV(mid_ch * 2, mid_ch, 1, convType));
    rebnconv2d = register_module("rebnconv2d", REBNCONV(mid_ch * 2, mid_ch, 1, convType));
    rebnconv1d = register_module("rebnconv1d", REBNCONV(mid_ch * 2, out_ch, 1, convType));
}

torch::Tensor RSU4Impl::forward(const torch::Tensor& x) 
{
    auto hxin = rebnconvin->forward(x);

    auto hx1 = rebnconv1->forward(hxin);
    auto hx = pool1->forward(hx1);

    auto hx2 = rebnconv2->forward(hx);
    hx = pool2->forward(hx2);

    auto hx3 = rebnconv3->forward(hx);
    auto hx4 = rebnconv4->forward(hx3);

    auto hx3d = rebnconv3d->forward(torch::cat({ hx4, hx3 }, 1));
    auto hx3dup = upsample_like(hx3d, hx2);

    auto hx2d = rebnconv2d->forward(torch::cat({ hx3dup, hx2 }, 1));
    auto hx2dup = upsample_like(hx2d, hx1);

    auto hx1d = rebnconv1d->forward(torch::cat({ hx2dup, hx1 }, 1));

    return hx1d + hxin;
}

// ======================================================================================
// RSU4F (Full Conv - no pooling)
// ======================================================================================

RSU4FImpl::RSU4FImpl(int in_ch, int mid_ch, int out_ch, ConvType convType) 
{
    rebnconvin = register_module("rebnconvin", REBNCONV(in_ch, out_ch, 1, convType));

    rebnconv1 = register_module("rebnconv1", REBNCONV(out_ch, mid_ch, 1, convType));
    rebnconv2 = register_module("rebnconv2", REBNCONV(mid_ch, mid_ch, 2, convType));
    rebnconv3 = register_module("rebnconv3", REBNCONV(mid_ch, mid_ch, 4, convType));
    rebnconv4 = register_module("rebnconv4", REBNCONV(mid_ch, mid_ch, 8, convType));

    rebnconv3d = register_module("rebnconv3d", REBNCONV(mid_ch * 2, mid_ch, 4, convType));
    rebnconv2d = register_module("rebnconv2d", REBNCONV(mid_ch * 2, mid_ch, 2, convType));
    rebnconv1d = register_module("rebnconv1d", REBNCONV(mid_ch * 2, out_ch, 1, convType));
}

torch::Tensor RSU4FImpl::forward(const torch::Tensor& x) 
{
    auto hxin = rebnconvin->forward(x);

    auto hx1 = rebnconv1->forward(hxin);
    auto hx2 = rebnconv2->forward(hx1);
    auto hx3 = rebnconv3->forward(hx2);
    auto hx4 = rebnconv4->forward(hx3);

    auto hx3d = rebnconv3d->forward(torch::cat({ hx4, hx3 }, 1));
    auto hx2d = rebnconv2d->forward(torch::cat({ hx3d, hx2 }, 1));
    auto hx1d = rebnconv1d->forward(torch::cat({ hx2d, hx1 }, 1));

    return hx1d + hxin;
}

// ======================================================================================
// U2Net 
// ======================================================================================


U2NetModel::U2NetModel(int in_ch, int out_ch, bool small, ConvType convType)
{
    
    if (!small) 
    {
        // ---------- Full Model ----------
        stage1 = register_module("stage1", RSU7(in_ch, 32, 64, convType));
        pool12 = register_module("pool12", MaxPool2d(MaxPool2dOptions(2).stride(2).ceil_mode(true)));

        stage2 = register_module("stage2", RSU6(64, 32, 128, convType));
        pool23 = register_module("pool23", MaxPool2d(MaxPool2dOptions(2).stride(2).ceil_mode(true)));

        stage3 = register_module("stage3", RSU5(128, 64, 256, convType));
        pool34 = register_module("pool34", MaxPool2d(MaxPool2dOptions(2).stride(2).ceil_mode(true)));

        stage4 = register_module("stage4", RSU4(256, 128, 512, convType));
        pool45 = register_module("pool45", MaxPool2d(MaxPool2dOptions(2).stride(2).ceil_mode(true)));

        stage5 = register_module("stage5", RSU4F(512, 256, 512, convType));
        pool56 = register_module("pool56", MaxPool2d(MaxPool2dOptions(2).stride(2).ceil_mode(true)));

        stage6 = register_module("stage6", RSU4F(512, 256, 512, convType));

        // Decoder
        stage5d = register_module("stage5d", RSU4F(1024, 256, 512, convType));
        stage4d = register_module("stage4d", RSU4(1024, 128, 256, convType));
        stage3d = register_module("stage3d", RSU5(512, 64, 128, convType));
        stage2d = register_module("stage2d", RSU6(256, 32, 64, convType));
        stage1d = register_module("stage1d", RSU7(128, 16, 64, convType));

        // Side outputs
        side1 = register_module("side1", Conv2d(Conv2dOptions(64, out_ch, 3).padding(1)));
        side2 = register_module("side2", Conv2d(Conv2dOptions(64, out_ch, 3).padding(1)));
        side3 = register_module("side3", Conv2d(Conv2dOptions(128, out_ch, 3).padding(1)));
        side4 = register_module("side4", Conv2d(Conv2dOptions(256, out_ch, 3).padding(1)));
        side5 = register_module("side5", Conv2d(Conv2dOptions(512, out_ch, 3).padding(1)));
        side6 = register_module("side6", Conv2d(Conv2dOptions(512, out_ch, 3).padding(1)));

        outconv = register_module("outconv", Conv2d(Conv2dOptions(6 * out_ch, out_ch, 1)));
    }
    else 
    {
        // ---------- Small Model ----------
        stage1 = register_module("stage1", RSU7(in_ch, 16, 64, convType));
        pool12 = register_module("pool12", MaxPool2d(MaxPool2dOptions(2).stride(2).ceil_mode(true)));

        stage2 = register_module("stage2", RSU6(64, 16, 64, convType));
        pool23 = register_module("pool23", MaxPool2d(MaxPool2dOptions(2).stride(2).ceil_mode(true)));

        stage3 = register_module("stage3", RSU5(64, 16, 64, convType));
        pool34 = register_module("pool34", MaxPool2d(MaxPool2dOptions(2).stride(2).ceil_mode(true)));

        stage4 = register_module("stage4", RSU4(64, 16, 64, convType));
        pool45 = register_module("pool45", MaxPool2d(MaxPool2dOptions(2).stride(2).ceil_mode(true)));

        stage5 = register_module("stage5", RSU4F(64, 16, 64, convType));
        pool56 = register_module("pool56", MaxPool2d(MaxPool2dOptions(2).stride(2).ceil_mode(true)));

        stage6 = register_module("stage6", RSU4F(64, 16, 64, convType));

        stage5d = register_module("stage5d", RSU4F(128, 16, 64, convType));
        stage4d = register_module("stage4d", RSU4(128, 16, 64, convType));
        stage3d = register_module("stage3d", RSU5(128, 16, 64, convType));
        stage2d = register_module("stage2d", RSU6(128, 16, 64, convType));
        stage1d = register_module("stage1d", RSU7(128, 16, 64, convType));

        side1 = register_module("side1", Conv2d(Conv2dOptions(64, out_ch, 3).padding(1)));
        side2 = register_module("side2", Conv2d(Conv2dOptions(64, out_ch, 3).padding(1)));
        side3 = register_module("side3", Conv2d(Conv2dOptions(64, out_ch, 3).padding(1)));
        side4 = register_module("side4", Conv2d(Conv2dOptions(64, out_ch, 3).padding(1)));
        side5 = register_module("side5", Conv2d(Conv2dOptions(64, out_ch, 3).padding(1)));
        side6 = register_module("side6", Conv2d(Conv2dOptions(64, out_ch, 3).padding(1)));

        outconv = register_module("outconv", Conv2d(Conv2dOptions(6 * out_ch, out_ch, 1)));
    }    
}


const char* U2NetModel::GetName() const
{
    return "U2Net";
}

std::vector<torch::Tensor> U2NetModel::forward(const torch::Tensor& x)
{
    auto hx = x;

    // Encoder
    auto hx1 = stage1->forward(hx);
    hx = pool12->forward(hx1);

    auto hx2 = stage2->forward(hx);
    hx = pool23->forward(hx2);

    auto hx3 = stage3->forward(hx);
    hx = pool34->forward(hx3);

    auto hx4 = stage4->forward(hx);
    hx = pool45->forward(hx4);

    auto hx5 = stage5->forward(hx);
    hx = pool56->forward(hx5);

    auto hx6 = stage6->forward(hx);
    auto hx6up = upsample_like(hx6, hx5);

    // Decoder
    auto hx5d = stage5d->forward(torch::cat({ hx6up, hx5 }, 1));
    auto hx5dup = upsample_like(hx5d, hx4);

    auto hx4d = stage4d->forward(torch::cat({ hx5dup, hx4 }, 1));
    auto hx4dup = upsample_like(hx4d, hx3);

    auto hx3d = stage3d->forward(torch::cat({ hx4dup, hx3 }, 1));
    auto hx3dup = upsample_like(hx3d, hx2);

    auto hx2d = stage2d->forward(torch::cat({ hx3dup, hx2 }, 1));
    auto hx2dup = upsample_like(hx2d, hx1);

    auto hx1d = stage1d->forward(torch::cat({ hx2dup, hx1 }, 1));

    // Side outputs
    auto d1 = side1->forward(hx1d);

    auto d2 = side2->forward(hx2d);
    d2 = upsample_like(d2, d1);

    auto d3 = side3->forward(hx3d);
    d3 = upsample_like(d3, d1);

    auto d4 = side4->forward(hx4d);
    d4 = upsample_like(d4, d1);

    auto d5 = side5->forward(hx5d);
    d5 = upsample_like(d5, d1);

    auto d6 = side6->forward(hx6);
    d6 = upsample_like(d6, d1);

    auto d0 = outconv->forward(torch::cat({ d1, d2, d3, d4, d5, d6 }, 1));

    return { d0, d1, d2, d3, d4, d5, d6 };            
}

std::vector<torch::Tensor> U2NetModel::RunForward(DataLoaderData& batch)
{
    //input size must be w >= 256 and h >= 256

    auto x = this->forward(batch.input);
    
    x.push_back(batch.target);

    return x;// { x, batch.target };
}