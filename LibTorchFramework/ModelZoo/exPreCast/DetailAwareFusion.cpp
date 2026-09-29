#include "DetailAwareFusion.h"


DetailAwareFusionImpl::DetailAwareFusionImpl(
    int64_t channels,
    int64_t hiddenChannels,
    float maxCorrection)
    :
    channels(channels),
    hiddenChannels(hiddenChannels),
    maxCorrection(maxCorrection)
{
    /*
        Inputs:

            unet         [B,S,C,H,W]
            flow         [B,S,C,H,W]

        We deliberately expose both absolute and differential information.

        u
        f
        f - u
        |f - u|
        HP(f)
        HP(f) - HP(u)
        LP(f) - LP(u)

        The signed differences are important:
            abs(f-u) loses the direction of the correction.
    */

    // 7 * C input channels:
    //
    //   u
    //   f
    //   f-u
    //   |f-u|
    //   HP(f)
    //   HP(f)-HP(u)
    //   LP(f)-LP(u)
    //
    encoder = register_module("encoder",
        torch::nn::Sequential(
            torch::nn::Conv2d(
                torch::nn::Conv2dOptions(
                    channels * 7,
                    hiddenChannels,
                    3)
                .padding(1)),

            torch::nn::GELU(),

            torch::nn::Conv2d(
                torch::nn::Conv2dOptions(
                    hiddenChannels,
                    hiddenChannels,
                    3)
                .padding(1)),

            torch::nn::GELU()
        ));


    /*
        One gate per output channel/pixel.

        This does NOT directly blend UNet and flow.

        It controls how much residual correction is allowed.
    */
    gate = register_module("gate",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(hiddenChannels, channels, 1)));


    /*
        The correction itself.

        It is deliberately initialized to zero.

        Therefore:

            correction == 0

        at initialization and the entire module behaves as:

            output == UNet
    */
    correction = register_module("correction", 
        torch::nn::Conv2d(torch::nn::Conv2dOptions(hiddenChannels, channels, 3).padding(1)));


    /*
        Identity initialization.

        We do NOT want:

            output = 0.5 * UNet + 0.5 * Flow

        at initialization.

        We want:

            output = UNet
    */
    torch::NoGradGuard noGrad;

    correction->weight.zero_();
    correction->bias.zero_();


    /*
        sigmoid(-3) ~= 0.047

        So even after correction weights become non-zero,
        the initial gate is strongly conservative.

        More importantly, correction itself starts at zero,
        so the actual initial output is exactly UNet.
    */
    gate->weight.zero_();
    gate->bias.fill_(-3.0);
}


torch::Tensor DetailAwareFusionImpl::forward(const torch::Tensor& unet, const torch::Tensor& flow)
{
    /*
        Expected:

            unet = [B,S,C,H,W]
            flow = [B,S,C,H,W]
    */

    TORCH_CHECK(
        unet.dim() == 5,
        "DetailAwareFusion: unet must be [B,S,C,H,W], got ",
        unet.sizes());

    TORCH_CHECK(
        flow.dim() == 5,
        "DetailAwareFusion: flow must be [B,S,C,H,W], got ",
        flow.sizes());

    TORCH_CHECK(
        unet.sizes() == flow.sizes(),
        "DetailAwareFusion: unet and flow must have identical shapes. "
        "unet=", unet.sizes(),
        ", flow=", flow.sizes());

    const int64_t B = unet.size(0);
    const int64_t S = unet.size(1);
    const int64_t C = unet.size(2);
    const int64_t H = unet.size(3);
    const int64_t W = unet.size(4);

    TORCH_CHECK(
        C == channels,
        "DetailAwareFusion: expected ",
        channels,
        " channels, got ",
        C);


    /*
        Conv2d operates on:

            [N,C,H,W]

        so collapse B and sequence.
    */
    auto u = unet.reshape({B * S, C, H, W});

    auto f = flow.reshape({B * S, C, H, W});


    /*
        Signed flow difference.

        This is more useful than only abs(u-f), because:

            f-u > 0

        and

            f-u < 0

        represent fundamentally different corrections.
    */
    auto difference = f - u;

    auto absoluteDifference = torch::abs(difference);


    /*
        High-frequency information.

        HP(x) = x - LP(x)
    */
    auto flowDetail = HighPass(f);

    auto unetDetail = HighPass(u);

    auto detailDifference = flowDetail - unetDetail;


    /*
        Low-frequency difference.

        This represents the part of flow which is most relevant
        to large-scale advection / displacement.

        We expose this explicitly rather than asking the network
        to discover the decomposition entirely by itself.
    */
    auto lowDifference = LowPass(f) - LowPass(u);


    /*
        7*C channels:

            0: u
            1: f
            2: f-u
            3: |f-u|
            4: HP(f)
            5: HP(f)-HP(u)
            6: LP(f)-LP(u)
    */
    auto x = torch::cat(
        {
            u,
            f,
            difference,
            absoluteDifference,
            flowDetail,
            detailDifference,
            lowDifference
        },
        1);


    auto features = encoder->forward(x);


    /*
        Gate controls where correction is useful.

        Unlike the old formulation:

            alpha*u + (1-alpha)*f

        this cannot directly select flow as the output.

        It only controls the residual added to UNet.
    */
    auto gateValue = torch::sigmoid(gate->forward(features));


    /*
        Bounded residual.

        tanh gives:

            -1 <= delta <= +1

        and maxCorrection limits its absolute magnitude.

        With maxCorrection=0.1:

            -0.1 <= correction <= +0.1
    */
    auto delta = torch::tanh(correction->forward(features));

    auto residual = maxCorrection * gateValue * delta;


    /*
        THE IMPORTANT PART:

            output = UNet + residual

        Therefore UNet remains the hard baseline.

        Flow can:
            - move/correct precipitation
            - sharpen details
            - suppress details
            - compensate for flow errors

        but it cannot simply replace UNet.
    */
    auto out = u + residual;


    /*
        Optional physical-range protection.

        I would NOT necessarily enable this if your training
        expects unconstrained logits.

        If UNet/flow are actual normalized precipitation values
        in [0,1], this is useful.

        If they are logits, REMOVE this clamp.

        Kept commented intentionally.
    */

    // out = torch::clamp(out, 0.0, 1.0);


    return out.reshape({B, S, C, H, W});
}


torch::Tensor DetailAwareFusionImpl::HighPass(const torch::Tensor& x)
{
    /*
        Average pooling with same spatial dimensions.

        HP(x) = x - LP(x)

        Padding=1 keeps H,W unchanged.
    */
    auto low = torch::avg_pool2d(
        x,
        { 3, 3 },
        { 1, 1 },
        { 1, 1 });

    return x - low;
}


torch::Tensor DetailAwareFusionImpl::LowPass(const torch::Tensor& x)
{
    /*
        Slightly larger receptive field than HighPass.

        This is intentionally relatively cheap.

        The low-frequency component gives the fusion network
        explicit information about large-scale displacement.
    */
    return torch::avg_pool2d(
        x,
        { 5, 5 },
        { 1, 1 },
        { 2, 2 });
}

