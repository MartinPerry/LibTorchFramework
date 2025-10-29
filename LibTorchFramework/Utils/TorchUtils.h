#ifndef TORCH_UTILS_H
#define TORCH_UTILS_H

#include <utility>

#include <ATen/ATen.h>
#include <torch/torch.h>

//===============================================================================

#define AUTO_REGISTER_NEW_MODULE(var, ...) \
    var = register_module(#var __VA_OPT__(,) __VA_ARGS__)

#define AUTO_REGISTER_EXISTING_MODULE(var) \
    var = register_module(#var, var)

//===============================================================================

// traits classes for PyTorch tensor type constants
template <typename T>
struct tensor_type_traits {};

template <>
struct tensor_type_traits<int32_t> {
    static constexpr auto typenum = at::kInt;
};
template <>
struct tensor_type_traits<int64_t> {
    static constexpr auto typenum = at::kLong;
};
template <>
struct tensor_type_traits<float> {
    static constexpr auto typenum = at::kFloat;
};

//===============================================================================


class TorchUtils
{
public:
	template <typename T, typename A>
	static at::Tensor make_tensor(std::vector<T, A>&& vec, 
        torch::TensorOptions options = {});

    template <typename T, typename A>
    static at::Tensor make_tensor(std::vector<T, A>&& vec, at::IntArrayRef size, 
        torch::TensorOptions options = {});

};

//===============================================================================

template <typename T, typename A>
at::Tensor TorchUtils::make_tensor(std::vector<T, A>&& vec, torch::TensorOptions options)
{
    return TorchUtils::make_tensor(
        std::forward<typename std::vector<T, A>>(vec),
        { static_cast<int64_t>(vec.size()) }, 
        options
    );
}

/// <summary>
/// https://discuss.pytorch.org/t/how-to-convert-vector-int-into-c-torch-tensor/66539/5
/// </summary>
/// <typeparam name="T"></typeparam>
/// <typeparam name="A"></typeparam>
/// <param name="vec"></param>
/// <param name="options"></param>
/// <returns></returns>
template <typename T, typename A>
at::Tensor TorchUtils::make_tensor(std::vector<T, A>&& vec, at::IntArrayRef size, torch::TensorOptions options)
{
    using V = std::vector<T, A>;

    // allocate storage for placement new (on exception also prevents leaks)
    auto buf = std::make_unique<uint8_t[]>(sizeof(V));
    
    // placement new + get pointer to moved vector
    auto vptr = new(buf.get()) V{ std::move(vec) };
    
    // create Torch tensor
    auto ten = torch::from_blob(
        vptr->data(),
        size,
        // note: argument is unused since we are deleting through vptr
        [vptr](void*)
        {
            // take ownership of the buffer for later deletion on scope exit
            std::unique_ptr<uint8_t[]> vbuf{ (uint8_t*)vptr };
            vptr->~V();
        },
        // data type determined via traits class specializations
        options.dtype(tensor_type_traits<T>::typenum)
    );

    // we only release the buffer now in case from_blob throws
    buf.release();
    return ten;
}

#endif