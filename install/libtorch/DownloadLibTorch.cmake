# Downloads and Extracts LiBTorch for various configs
#
# !Note that LibTorch on Windows has two separate versions 
# based on configuration (Debug, Release)
#


# === Set cache variable for configurations to download
set(DOWNLOAD_TORCH_CONFIGS "Release;Debug" CACHE STRING "List of configurations to download (semicolon-separated)")

# === Set the urls to download
# (Feel free to modify)
set(LIBTORCH_LINK_Release "https://download.pytorch.org/libtorch/cu128/libtorch-win-shared-with-deps-2.8.0%2Bcu128.zip")
set(LIBTORCH_LINK_Debug "https://download.pytorch.org/libtorch/cu128/libtorch-win-shared-with-deps-debug-2.8.0%2Bcu128.zip")
set(LIBTORCH_LINK_Linux "https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcu118.zip")



# ================ Implementation ================
# (no need to change it)

# Macro to download and extract LibTorch
macro(download_and_extract_libtorch VERSION LINK)
    set(LIBTORCH_DIR "${PROJECT_SOURCE_DIR}/vendor/libtorch")
    set(LIBTORCH_${VERSION}_DIR "${LIBTORCH_DIR}/${VERSION}")
    set(LIBTORCH_${VERSION}_ZIP "${LIBTORCH_DIR}/${VERSION}.zip")

    if(NOT EXISTS "${LIBTORCH_${VERSION}_DIR}")
        message("LibTorch ${VERSION} not found: ${LIBTORCH_${VERSION}_DIR}")

        if(NOT EXISTS "${LIBTORCH_${VERSION}_ZIP}")
            message("Downloading ${VERSION}: ${LINK}")
            file(MAKE_DIRECTORY "${LIBTORCH_${VERSION}_DIR}")
            file(DOWNLOAD "${LINK}" "${LIBTORCH_${VERSION}_ZIP}" SHOW_PROGRESS)
        else()
            message("File found: ${LIBTORCH_${VERSION}_ZIP}")
        endif()

        message("Extracting ${LIBTORCH_${VERSION}_ZIP}")
        file(ARCHIVE_EXTRACT INPUT "${LIBTORCH_${VERSION}_ZIP}" DESTINATION "${LIBTORCH_${VERSION}_DIR}")
        message("LibTorch ${VERSION} ready.")
    endif()
endmacro()


# Main folder
set(LIBTORCH_DIR "${PROJECT_SOURCE_DIR}/vendor/libtorch")
file(MAKE_DIRECTORY "${LIBTORCH_DIR}")

# Download and extract configurations in DOWNLOAD_TORCH_CONFIGS
foreach(CONFIG IN LISTS DOWNLOAD_TORCH_CONFIGS)
    download_and_extract_libtorch(${CONFIG} "${LIBTORCH_LINK_${CONFIG}}")
endforeach()
