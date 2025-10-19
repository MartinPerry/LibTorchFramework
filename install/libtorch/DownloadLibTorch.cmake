# Downloads and Extracts LiBTorch for various configs
#
# !Note that LibTorch on Windows has two separate versions 
# based on configuration (Debug, Release)
#


# === Set cache variable for configurations to download
set(DOWNLOAD_TORCH_CONFIGS "Release;Debug" CACHE STRING "List of configurations to download (semicolon-separated)")

# === Set the urls to download
# (Feel free to modify)
set(LIBTORCH_VERSION "2.9.0")
set(CUDA_VERSION "128")
set(LIBTORCH_LINK_Release "https://download.pytorch.org/libtorch/cu${CUDA_VERSION}/libtorch-win-shared-with-deps-${LIBTORCH_VERSION}%2Bcu${CUDA_VERSION}.zip")
set(LIBTORCH_LINK_Debug "https://download.pytorch.org/libtorch/cu${CUDA_VERSION}/libtorch-win-shared-with-deps-debug-${LIBTORCH_VERSION}%2Bcu${CUDA_VERSION}.zip")
set(LIBTORCH_LINK_Linux "https://download.pytorch.org/libtorch/cu${CUDA_VERSION}/libtorch-shared-with-deps-${LIBTORCH_VERSION}%2Bcu${CUDA_VERSION}.zip")



# ================ Implementation ================
# (no need to change it)

# Macro to download and extract LibTorch
macro(download_and_extract_libtorch CONFIG_VERSION LINK)
    set(LIBTORCH_DIR "${PROJECT_SOURCE_DIR}/vendor/libtorch_${LIBTORCH_VERSION}")
    set(LIBTORCH_${CONFIG_VERSION}_DIR "${LIBTORCH_DIR}/${CONFIG_VERSION}_${LIBTORCH_VERSION}")
    set(LIBTORCH_${CONFIG_VERSION}_ZIP "${LIBTORCH_DIR}/${CONFIG_VERSION}_${LIBTORCH_VERSION}.zip")

    if(NOT EXISTS "${LIBTORCH_${CONFIG_VERSION}_DIR}")
        message("LibTorch ${CONFIG_VERSION} not found: ${LIBTORCH_${CONFIG_VERSION}_DIR}")

        if(NOT EXISTS "${LIBTORCH_${CONFIG_VERSION}_ZIP}")
            message("Downloading ${CONFIG_VERSION}: ${LINK}")
            file(MAKE_DIRECTORY "${LIBTORCH_${CONFIG_VERSION}_DIR}")
            file(DOWNLOAD "${LINK}" "${LIBTORCH_${CONFIG_VERSION}_ZIP}" SHOW_PROGRESS)
        else()
            message("File found: ${LIBTORCH_${CONFIG_VERSION}_ZIP}")
        endif()

        message("Extracting ${LIBTORCH_${CONFIG_VERSION}_ZIP}")
        file(ARCHIVE_EXTRACT INPUT "${LIBTORCH_${CONFIG_VERSION}_ZIP}" DESTINATION "${LIBTORCH_${CONFIG_VERSION}_DIR}")
        message("LibTorch ${CONFIG_VERSION} ready.")
    endif()
endmacro()


# Main folder
set(LIBTORCH_DIR "${PROJECT_SOURCE_DIR}/vendor/libtorch")
file(MAKE_DIRECTORY "${LIBTORCH_DIR}")

# Download and extract configurations in DOWNLOAD_TORCH_CONFIGS
foreach(CONFIG IN LISTS DOWNLOAD_TORCH_CONFIGS)
    download_and_extract_libtorch(${CONFIG} "${LIBTORCH_LINK_${CONFIG}}")
endforeach()
