cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(LibTorch)



# ========================= Config =============================================
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
option(TORCH_CONFIGURATION "Project Configuration Debug/Release, used only on Windows" "Release")
if(UNIX)
    # override configs to use only Linux version of the torchlib
    set(TORCH_CONFIGURATION "Linux")
endif()

# prevent from using debug configuration when using release version of libtorch
# and vice versa
if(NOT TORCH_CONFIGURATION STREQUAL "Linux" )
    message("Setting configuration type ${TORCH_CONFIGURATION}")
    set(CMAKE_CONFIGURATION_TYPES "${TORCH_CONFIGURATION}" CACHE STRING "" FORCE)
endif()

# ========================= TORCH =============================================

# Download if necessary
set(DOWNLOAD_TORCH_CONFIGS ${TORCH_CONFIGURATION})
include(DownloadLibTorch.cmake)









