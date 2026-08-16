cmake_minimum_required(VERSION 3.20)
project(LibTorchFramework LANGUAGES CXX)

#cmake --fresh -S . -B build_cmake ^
#   -G "Visual Studio 17 2022" ^
#   -A x64 ^
#   -DLIBTORCH_FRAMEWORK_SOURCE_DIR=D:\path\to\LibTorchFramework ^
#   -DLIBTORCH_CONFIGURATION=Release ^
#   -DLIBTORCH_ROOT=D:\path\to\libtorch\Release\libtorch_2.10.0 ^
#   -DPLAYGROUND_ROOT=D:\Martin\Programming\test\Playground ^
#   -DPLAYGROUND_LIBRARY_RELEASE=D:\path\to\Playground.lib

#cmake --build build_cmake --config Release --parallel 1

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(LIBTORCH_FRAMEWORK_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}" CACHE PATH
    "Directory containing LibTorchFramework sources and main.cpp")

set(LIBTORCH_VERSION "2.10.0" CACHE STRING "Version suffix of the local LibTorch directory")
set(LIBTORCH_CONFIGURATION "Release" CACHE STRING "Local LibTorch directory: Debug or Release")
set_property(CACHE LIBTORCH_CONFIGURATION PROPERTY STRINGS Debug Release)
set(LIBTORCH_ROOT "${LIBTORCH_FRAMEWORK_SOURCE_DIR}/../libtorch/${LIBTORCH_CONFIGURATION}/libtorch_${LIBTORCH_VERSION}"
    CACHE PATH "Root of the LibTorch distribution containing share/cmake/Torch")

option(LIBTORCH_FRAMEWORK_USE_CUDA "Enable CUDA-backed LibTorch code" OFF)

if(LIBTORCH_FRAMEWORK_USE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)

    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "CUDA requested but no CUDA compiler (nvcc) was found")
    endif()

    enable_language(CUDA)

    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    find_package(CUDAToolkit REQUIRED)
endif()

set(PLAYGROUND_ROOT "${LIBTORCH_FRAMEWORK_SOURCE_DIR}/../Playground" CACHE PATH
    "Root containing Playground source, include, and library directories")
set(PLAYGROUND_SOURCE_DIR "${PLAYGROUND_ROOT}/Playground" CACHE PATH "Playground source/header directory")
set(PLAYGROUND_INCLUDE_DIR "${PLAYGROUND_ROOT}/include" CACHE PATH "Playground external include directory")
set(PLAYGROUND_CMAKE_BUILD_DIR "${PLAYGROUND_ROOT}/build_cmake" CACHE PATH
    "Build directory produced by the Playground CMake project")
if(WIN32)
    set(_playground_release_library
        "${PLAYGROUND_CMAKE_BUILD_DIR}/Release/Playground.lib")
    set(_playground_debug_library
        "${PLAYGROUND_CMAKE_BUILD_DIR}/Debug/Playground.lib")
else()
    set(_playground_release_library
        "${PLAYGROUND_CMAKE_BUILD_DIR}/libPlayground.a")
    set(_playground_debug_library
        "${PLAYGROUND_CMAKE_BUILD_DIR}/libPlayground.a")
endif()
set(PLAYGROUND_LIBRARY_RELEASE "${_playground_release_library}" CACHE FILEPATH
    "Release Playground library built by CMake")
set(PLAYGROUND_LIBRARY_DEBUG "${_playground_debug_library}" CACHE FILEPATH
    "Debug Playground library built by CMake")

if(NOT WIN32)
    find_package(ICU REQUIRED COMPONENTS uc i18n)
    find_package(JPEG REQUIRED)
    find_package(PNG REQUIRED)

    find_library(TURBOJPEG_LIBRARY NAMES turbojpeg REQUIRED)
    find_library(WEBP_LIBRARY NAMES webp REQUIRED)
endif()

add_library(Playground::Playground STATIC IMPORTED GLOBAL)

set_target_properties(Playground::Playground PROPERTIES
    IMPORTED_CONFIGURATIONS "Debug;Release;RelWithDebInfo;MinSizeRel"
    IMPORTED_LOCATION_DEBUG "${PLAYGROUND_LIBRARY_DEBUG}"
    IMPORTED_LOCATION_RELEASE "${PLAYGROUND_LIBRARY_RELEASE}"
    IMPORTED_LOCATION_RELWITHDEBINFO "${PLAYGROUND_LIBRARY_RELEASE}"
    IMPORTED_LOCATION_MINSIZEREL "${PLAYGROUND_LIBRARY_RELEASE}"
    INTERFACE_INCLUDE_DIRECTORIES "${PLAYGROUND_SOURCE_DIR};${PLAYGROUND_INCLUDE_DIR}"
)

if(NOT WIN32)
    set_target_properties(Playground::Playground PROPERTIES
        INTERFACE_LINK_LIBRARIES "JPEG::JPEG;PNG::PNG;${TURBOJPEG_LIBRARY};${WEBP_LIBRARY}"
    )
endif()

# Source files are intentionally listed explicitly.
set(LIBTORCH_FRAMEWORK_SOURCES
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/AbstractModel.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/CudaGraphHelper.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Metrics/MetricsDefault.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Metrics/MetricsImage.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Metrics/MetricsVideo.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Modules/Convolutions/ConvGRU.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Modules/Convolutions/CoordConv.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Modules/Convolutions/DeformConv.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Modules/Convolutions/DeformConvImpl/torchvision/vision_deform_conv2d.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Modules/Convolutions/DeformConvImpl/torchvision/vision_deform_conv2d_kernel.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Modules/Convolutions/DeformConvImpl/torchvision/vision_deform_conv2d_kernel_autocast.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Modules/Convolutions/DeformConvImpl/torchvision/vision_deform_conv2d_kernel_autograd.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Modules/Convolutions/DeformConvImpl/tvdcn/ops/autocast/deform_autocast_conv1d_kernel.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Modules/Convolutions/DeformConvImpl/tvdcn/ops/autocast/deform_autocast_conv2d_kernel.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Modules/Convolutions/DeformConvImpl/tvdcn/ops/autocast/deform_autocast_conv3d_kernel.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Modules/Convolutions/DeformConvImpl/tvdcn/ops/autocast/deform_autocast_conv_transpose1d_kernel.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Modules/Convolutions/DeformConvImpl/tvdcn/ops/autocast/deform_autocast_conv_transpose2d_kernel.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Modules/Convolutions/DeformConvImpl/tvdcn/ops/autocast/deform_autocast_conv_transpose3d_kernel.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Modules/Convolutions/DeformConvImpl/tvdcn/ops/autograd/deform_autograd_conv1d_kernel.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Modules/Convolutions/DeformConvImpl/tvdcn/ops/autograd/deform_autograd_conv2d_kernel.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Modules/Convolutions/DeformConvImpl/tvdcn/ops/autograd/deform_autograd_conv3d_kernel.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Modules/Convolutions/DeformConvImpl/tvdcn/ops/autograd/deform_autograd_conv_transpose1d_kernel.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Modules/Convolutions/DeformConvImpl/tvdcn/ops/autograd/deform_autograd_conv_transpose2d_kernel.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Modules/Convolutions/DeformConvImpl/tvdcn/ops/autograd/deform_autograd_conv_transpose3d_kernel.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Modules/Convolutions/DeformConvImpl/tvdcn/ops/cpu/deform_conv1d_common_kernels.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Modules/Convolutions/DeformConvImpl/tvdcn/ops/cpu/deform_conv1d_kernel.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Modules/Convolutions/DeformConvImpl/tvdcn/ops/cpu/deform_conv2d_common_kernels.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Modules/Convolutions/DeformConvImpl/tvdcn/ops/cpu/deform_conv2d_kernel.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Modules/Convolutions/DeformConvImpl/tvdcn/ops/cpu/deform_conv3d_common_kernels.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Modules/Convolutions/DeformConvImpl/tvdcn/ops/cpu/deform_conv3d_kernel.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Modules/Convolutions/DeformConvImpl/tvdcn/ops/cpu/deform_conv_transpose1d_kernel.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Modules/Convolutions/DeformConvImpl/tvdcn/ops/cpu/deform_conv_transpose2d_kernel.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Modules/Convolutions/DeformConvImpl/tvdcn/ops/cpu/deform_conv_transpose3d_kernel.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Modules/Convolutions/DeformConvImpl/tvdcn/ops/cuda/cuda_deform_conv1d_kernel.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Modules/Convolutions/DeformConvImpl/tvdcn/ops/cuda/cuda_deform_conv2d_kernel.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Modules/Convolutions/DeformConvImpl/tvdcn/ops/cuda/cuda_deform_conv3d_kernel.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Modules/Convolutions/DeformConvImpl/tvdcn/ops/cuda/cuda_deform_conv_transpose1d_kernel.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Modules/Convolutions/DeformConvImpl/tvdcn/ops/cuda/cuda_deform_conv_transpose2d_kernel.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Modules/Convolutions/DeformConvImpl/tvdcn/ops/cuda/cuda_deform_conv_transpose3d_kernel.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Modules/Convolutions/DeformConvImpl/tvdcn/ops/deform_conv1d.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Modules/Convolutions/DeformConvImpl/tvdcn/ops/deform_conv2d.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Modules/Convolutions/DeformConvImpl/tvdcn/ops/deform_conv3d.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Modules/Convolutions/DeformConvImpl/tvdcn/ops/deform_conv_transpose1d.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Modules/Convolutions/DeformConvImpl/tvdcn/ops/deform_conv_transpose2d.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Modules/Convolutions/DeformConvImpl/tvdcn/ops/deform_conv_transpose3d.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Modules/Convolutions/DeformConvImpl/tvdcn/ops/utils/tensor_utils.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Modules/Convolutions/DeformConvImpl/tvdcn/tvdcn.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Modules/DropPath.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Modules/Embedding.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Modules/Linear.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Modules/LoRALinear.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Modules/LossFunctions/FACL.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Modules/LossFunctions/FocalFrequencyLoss.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Modules/LossFunctions/SSIMLoss.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Modules/MLP.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Modules/WeightsInit/TruncatedNormalInit.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Optimizers/AdamW8bit.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Optimizers/FusedAdamW8bit.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Optimizers/LAMB.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Runner.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Snapshot/PretrainedManager.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Snapshot/SafeTensorLoader.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Snapshot/safetensors.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Snapshot/SnapshotLoader.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Snapshot/SnapshotSaver.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Tokenizers/Strings/UnicodeRegex.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Tokenizers/TokenizerBPE.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Tokenizers/TokenizerJsonLoader.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/Trainer.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/CustomScenarios/exPreCastTraining/MeteonetInputLoader.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/CustomScenarios/exPreCastTraining/setup_exprecast.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/CustomScenarios/LLMs/setup_llama.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/CustomScenarios/MrmsDatasetTraining/MrmsInputLoader.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/CustomScenarios/MrmsDatasetTraining/MrmsLoss.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/CustomScenarios/MrmsDatasetTraining/setup_mrms.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/CustomScenarios/SDVAE/setup_sdvae.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/CustomScenarios/U2NetTraining/setup_u2net.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/CustomScenarios/UNetTraining/setup_unet.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/CustomScenarios/_tests_/llm_smoke_tests.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/CustomScenarios/_tests_/optimizers_tests.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/CustomScenarios/_tests_/tokenizer_tests.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/InputProcessing/DefaultDataset.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/InputProcessing/InputLoader.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/InputProcessing/InputLoadersWrapper.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/InputProcessing/InputLoaders/EncoderDecoderInputLoader.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/InputProcessing/InputLoaders/SegmentationInputLoader.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/InputProcessing/InputLoaders/TextFilesInputLoader.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/InputProcessing/InputLoaders/VideoSequenceInputLoader.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/main.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/ModelZoo/exPreCast/BasicLayerSkip.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/ModelZoo/exPreCast/CubicDualUpsample.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/ModelZoo/exPreCast/exPreCastModel.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/ModelZoo/exPreCast/PatchEmbed3D.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/ModelZoo/exPreCast/PatchExpanding3D.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/ModelZoo/exPreCast/PatchMerging.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/ModelZoo/exPreCast/PixelShuffle3D.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/ModelZoo/exPreCast/SwinTransformerBlock3D.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/ModelZoo/exPreCast/WindowAttention3D.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/ModelZoo/exPreCast/WindowUtils.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/ModelZoo/LLMs/llama.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/ModelZoo/LLMs/LLamaSafeTensorLoader.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/ModelZoo/ResNet/ResNetModel.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/ModelZoo/SDVAE/attention.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/ModelZoo/SDVAE/decoder.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/ModelZoo/SDVAE/encoder.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/ModelZoo/SDVAE/SDVAEModel.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/ModelZoo/SimVPv2/modules.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/ModelZoo/SimVPv2/SimVPv2Model.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/ModelZoo/U2Net/U2NetModel.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/ModelZoo/UNet/UNetModel.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/PerformanceSettings.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/SettingsLoader.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/Utils/ModelInfo.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/Utils/ProgressBar.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/Utils/TorchImageUtils.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/Utils/TorchUtils.cpp
    ${LIBTORCH_FRAMEWORK_SOURCE_DIR}/Utils/TrainingHelper.cpp
)

if(LIBTORCH_FRAMEWORK_USE_CUDA)
    file(GLOB_RECURSE LIBTORCH_FRAMEWORK_CUDA_SOURCES CONFIGURE_DEPENDS
        "${LIBTORCH_FRAMEWORK_SOURCE_DIR}/core/*.cu"
        "${LIBTORCH_FRAMEWORK_SOURCE_DIR}/CustomScenarios/*.cu"
        "${LIBTORCH_FRAMEWORK_SOURCE_DIR}/InputProcessing/*.cu"
        "${LIBTORCH_FRAMEWORK_SOURCE_DIR}/ModelZoo/*.cu"
        "${LIBTORCH_FRAMEWORK_SOURCE_DIR}/Utils/*.cu"
    )

    list(APPEND LIBTORCH_FRAMEWORK_SOURCES ${LIBTORCH_FRAMEWORK_CUDA_SOURCES})

    message(STATUS "CUDA sources:")
    foreach(_cuda_source IN LISTS LIBTORCH_FRAMEWORK_CUDA_SOURCES)
        message(STATUS "    ${_cuda_source}")
    endforeach()
endif()

add_executable(LibTorchFramework ${LIBTORCH_FRAMEWORK_SOURCES})

target_include_directories(LibTorchFramework PRIVATE
    "${LIBTORCH_FRAMEWORK_SOURCE_DIR}"
    "${PLAYGROUND_SOURCE_DIR}"
    "${PLAYGROUND_INCLUDE_DIR}")
target_compile_definitions(LibTorchFramework PRIVATE
    NOMINMAX
    _CONSOLE
    $<$<CONFIG:Debug>:_DEBUG>
    $<$<NOT:$<CONFIG:Debug>>:NDEBUG>
    $<$<NOT:$<CONFIG:Debug>>:USE_RELEASE_LOG_INFO>
    $<$<BOOL:${LIBTORCH_FRAMEWORK_USE_CUDA}>:USE_CUDA>)

if(MSVC)
    target_compile_options(LibTorchFramework PRIVATE /W3 /Zc:preprocessor)
    set_property(TARGET LibTorchFramework PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    target_link_options(LibTorchFramework PRIVATE
        "/INCLUDE:?warp_size@cuda@at@@YAHXZ")
else()
    target_compile_options(LibTorchFramework PRIVATE -Wall -Wextra)
endif()

if(NOT EXISTS "${LIBTORCH_ROOT}")
    message(FATAL_ERROR "LibTorch root does not exist: ${LIBTORCH_ROOT}. Set LIBTORCH_ROOT explicitly.")
endif()
list(PREPEND CMAKE_PREFIX_PATH "${LIBTORCH_ROOT}")
find_package(Torch REQUIRED)

if(LIBTORCH_FRAMEWORK_USE_CUDA)
    set_target_properties(LibTorchFramework PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION ON
    )

    target_compile_options(LibTorchFramework PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    )
endif()

# Torch::Torch supplies the normal transitive dependency graph. These explicit
# entries also mirror the MSVC auto-link directives used by the source and make
# the same dependencies visible to non-MSVC linkers.
target_link_directories(LibTorchFramework PRIVATE "${LIBTORCH_ROOT}/lib")
if(TARGET Torch::Torch)
    target_link_libraries(LibTorchFramework PRIVATE Torch::Torch)
else()
    target_link_libraries(LibTorchFramework PRIVATE ${TORCH_LIBRARIES})
endif()
if(TORCH_CXX_FLAGS)
    separate_arguments(_torch_cxx_flags NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
    target_compile_options(LibTorchFramework PRIVATE ${_torch_cxx_flags})
endif()


find_package(ICU REQUIRED COMPONENTS uc i18n)
find_package(JPEG REQUIRED)
find_package(PNG REQUIRED)

find_library(TURBOJPEG_LIBRARY NAMES turbojpeg REQUIRED)
find_library(WEBP_LIBRARY NAMES webp REQUIRED)


target_link_libraries(LibTorchFramework PRIVATE
    Playground::Playground
    c10
    cpuinfo
    dnnl
    kineto
    pthreadpool
    torch
    torch_cpu
    XNNPACK	   
)

if(LIBTORCH_FRAMEWORK_USE_CUDA)
    target_link_libraries(LibTorchFramework PRIVATE
        c10_cuda
        torch_cuda
        CUDA::cudart
    )

    if(WIN32)
        target_link_libraries(LibTorchFramework PRIVATE caffe2_nvrtc)
    endif()
endif()

if(WIN32)
    target_link_libraries(LibTorchFramework PRIVATE		
        $<$<CONFIG:Debug>:libprotobufd>
        $<$<CONFIG:Debug>:libprotocd>
        $<$<NOT:$<CONFIG:Debug>>:libprotobuf>
        $<$<NOT:$<CONFIG:Debug>>:libprotoc>
    )
else()
    target_link_libraries(LibTorchFramework PRIVATE
		ICU::uc
		ICU::i18n		
        protobuf
        protoc
    )
endif()