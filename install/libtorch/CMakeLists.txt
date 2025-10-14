cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(NNCProj)



# ========================= Config =============================================
option(NN_EXAMPLES "Build NN Examples" ON)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
option(TORCH_CONFIGURATION "Project Configuration Debug/Release, used only on Windows" "Release")
if(UNIX)
    # override configs to use only Linux version of the torchlib
    set(TORCH_CONFIGURATION "Linux")
endif()

# prevent from using debug configuration when using release version of libtorch
# and vice versa
if(DEFINED OneConfigOnly AND NOT TORCH_CONFIGURATION STREQUAL "Linux" )
    message("Setting only one configuration type ${TORCH_CONFIGURATION}")
    set(CMAKE_CONFIGURATION_TYPES "${TORCH_CONFIGURATION}" CACHE STRING "" FORCE)
endif()
# ========================= Project ============================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
if(MSVC)
  add_definitions(/MP)
endif()


# ========================= TORCH =============================================

# Download if necessary
set(DOWNLOAD_TORCH_CONFIGS ${TORCH_CONFIGURATION})
include(DownloadLibTorch.cmake)


#set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH} ${PROJECT_SOURCE_DIR}/vendor/libtorch/${TORCH_CONFIGURATION}/libtorch")
set(CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR}/vendor/libtorch/${TORCH_CONFIGURATION}/libtorch")
message(STATUS "CMake thinks that torch should be installed in: ${CMAKE_PREFIX_PATH}")
find_package(Torch REQUIRED)

target_link_libraries(${TARGET_NAME} "${TORCH_LIBRARIES}")
target_include_directories(${TARGET_NAME} PRIVATE "${TORCH_INCLUDE_DIRS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET ${TARGET_NAME}
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:${TARGET_NAME}>)
endif (MSVC)

# ========================= OPEN_CV ============================================

#find_package(OpenCV REQUIRED)
#include_directories(${OpenCV_INCLUDE_DIRS})
#target_link_libraries( ${TARGET_NAME} ${OpenCV_LIBS} )

if (MSVC)
# Find and copy OpenCV DLLs to avoid runtime errors
#  file(GLOB OPENCV_DLLS "${OpenCV_DIR}/x64/vc15/bin/*.dll")
#  add_custom_command(TARGET ${TARGET_NAME}
#                     POST_BUILD
#                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
#                     ${OPENCV_DLLS}
#                     $<TARGET_FILE_DIR:${TARGET_NAME}>)
endif()

# ========================= SPDLOG ============================================

target_include_directories(${TARGET_NAME} PUBLIC "vendor/spdlog") 

# ========================= TensorBoard ============================================

# NOTE tensorboard logger has issues with Linux
include(FetchContent)
FetchContent_Declare(
    tensorboard_logger
    GIT_REPOSITORY      https://github.com/Cooble/tensorboard_logger.git
    GIT_TAG             master
)
FetchContent_MakeAvailable(tensorboard_logger)

find_package(Protobuf)
if (Protobuf_FOUND)
    message(STATUS "Protobuf library found")
    message(STATUS "Protobuf_INCLUDE_DIRS: ${Protobuf_INCLUDE_DIRS}")
    message(STATUS "Protobuf_LIBRARIES: ${Protobuf_LIBRARIES}")
    message(STATUS "Protobuf_PROTOC_EXECUTABLE: ${Protobuf_PROTOC_EXECUTABLE}")
    message(STATUS "Protobuf_PROTOC_LIBRARY: ${Protobuf_PROTOC_LIBRARY}")
    message(STATUS "Protobuf_LITE_LIBRARIES: ${Protobuf_LITE_LIBRARIES}")
else()
    message(FATAL_ERROR "Protobuf library not found")
endif()


target_link_libraries(${TARGET_NAME} tensorboard_logger )


source_group(TREE ${PROJECT_SOURCE_DIR}/src PREFIX "src" FILES ${SOURCES})







