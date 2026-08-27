#!/usr/bin/env bash

set -euo pipefail

sudo apt-get install -y \
    build-essential \
    cmake \
    unzip \
    curl \
    ninja-build \
    pkg-config \
	libicu-dev \
	libprotobuf-dev \
	protobuf-compiler

LIBTORCH_FRAMEWORK_DIR="/mnt/e/Programming/Cpp/LibTorchFramework"
PLAYGROUND_DIR="/mnt/d/Martin/Programming/test/Playground"

# Override these on the command line when another release or CUDA build is
# needed, for example:
#   LIBTORCH_VERSION=2.11.0 LIBTORCH_VARIANT=cu130 ./build_debian.sh
LIBTORCH_VERSION="${LIBTORCH_VERSION:-2.13.0}"
LIBTORCH_VARIANT="${LIBTORCH_VARIANT:-cu130}"
USE_NCCL="${USE_NCCL:-OFF}"
#LIBTORCH_VARIANT="cpu"

BIN_DIR="${LIBTORCH_FRAMEWORK_DIR}/bin/${LIBTORCH_VERSION}_${LIBTORCH_VARIANT}"

LIBTORCH_BASE="${LIBTORCH_FRAMEWORK_DIR}/libtorch_debian/Release_${LIBTORCH_VERSION}_${LIBTORCH_VARIANT}"
LIBTORCH_ROOT="${LIBTORCH_BASE}/libtorch_${LIBTORCH_VERSION}"
#BUILD_DIR="${BUILD_DIR:-/tmp/libtorchframework-build-release}"
BUILD_DIR="${LIBTORCH_FRAMEWORK_DIR}/LibTorchFramework/build_cmake_debian_${LIBTORCH_VERSION}_${LIBTORCH_VARIANT}"
LIBTORCH_VARIANT_FILE="${LIBTORCH_ROOT}/.libtorch-variant"

# Current Linux CUDA distributions use this archive name. It can be overridden
# for a PyTorch release that uses a different ABI/archive naming convention.
LIBTORCH_ARCHIVE_NAME="${LIBTORCH_ARCHIVE_NAME:-libtorch-shared-with-deps-${LIBTORCH_VERSION}+${LIBTORCH_VARIANT}.zip}"
LIBTORCH_URL="https://download.pytorch.org/libtorch/${LIBTORCH_VARIANT}/${LIBTORCH_ARCHIVE_NAME//+/%2B}"

mkdir -p "${LIBTORCH_BASE}"

installed_variant=""
if [[ -f "${LIBTORCH_VARIANT_FILE}" ]]; then
    installed_variant="$(<"${LIBTORCH_VARIANT_FILE}")"
fi

if [[ ! -f "${LIBTORCH_ROOT}/share/cmake/Torch/TorchConfig.cmake" ||
      "${installed_variant}" != "${LIBTORCH_VARIANT}" ]]; then
    echo "Downloading LibTorch ${LIBTORCH_VERSION} (${LIBTORCH_VARIANT})"
    mkdir -p "${LIBTORCH_BASE}"

    # A previous interrupted cross-filesystem move can leave a partial target.
    # Only remove the versioned installation directory managed by this script.
    if [[ -e "${LIBTORCH_ROOT}" ]]; then
        case "${LIBTORCH_ROOT}" in
            "${LIBTORCH_BASE}"/libtorch_*)
                echo "Removing incomplete LibTorch installation: ${LIBTORCH_ROOT}"
                rm -rf -- "${LIBTORCH_ROOT}"
                ;;
            *)
                echo "Refusing to remove unexpected path: ${LIBTORCH_ROOT}" >&2
                exit 1
                ;;
        esac
    fi

    # Keep staging on /mnt/e. Moving from /tmp to a Windows-mounted filesystem
    # makes mv copy metadata that DrvFS may reject with Operation not permitted.
    download_dir="$(mktemp -d "${LIBTORCH_BASE}/.libtorch-download.XXXXXX")"
    trap 'rm -rf "${download_dir}"' EXIT

    curl --fail --location --retry 3 \
        --output "${download_dir}/libtorch.zip" \
        "${LIBTORCH_URL}"
    # -DD avoids restoring archive timestamps on the Windows-mounted volume.
    unzip -q -DD "${download_dir}/libtorch.zip" -d "${download_dir}/unpacked"

    if [[ ! -d "${download_dir}/unpacked/libtorch" ]]; then
        echo "Unexpected LibTorch archive layout: missing libtorch directory" >&2
        exit 1
    fi

    mv "${download_dir}/unpacked/libtorch" "${LIBTORCH_ROOT}"
    printf '%s\n' "${LIBTORCH_VARIANT}" >"${LIBTORCH_VARIANT_FILE}"
    rm -rf "${download_dir}"
    trap - EXIT
else
    echo "Using existing LibTorch installation: ${LIBTORCH_ROOT}"
fi

USE_CUDA=ON
CMAKE_CUDA_ARGS=()

if [[ "${LIBTORCH_VARIANT}" == "cpu" ]]; then
    USE_CUDA=OFF
    echo "Configuring a CPU-only LibTorch build"
else
    # A CUDA-enabled LibTorch archive contains CUDA libraries but not the nvcc
    # compiler/toolkit needed by TorchConfig.cmake.
    CUDA_ROOT="${CUDA_ROOT:-}"
    if [[ -z "${CUDA_ROOT}" ]]; then
        if command -v nvcc >/dev/null 2>&1; then
            CUDA_ROOT="$(dirname "$(dirname "$(command -v nvcc)")")"
        elif [[ -x /usr/local/cuda-13.0/bin/nvcc ]]; then
            CUDA_ROOT=/usr/local/cuda-13.0
        elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
            CUDA_ROOT=/usr/local/cuda
        fi
    fi

    if [[ ! -x "${CUDA_ROOT}/bin/nvcc" ]]; then
        cat >&2 <<'EOF'
CUDA 13 toolkit was not found. The cu130 LibTorch package requires nvcc and
the CUDA runtime development libraries. Install the CUDA toolkit (not a Linux
NVIDIA driver inside WSL), then rerun this script. You may also specify it as:

  CUDA_ROOT=/usr/local/cuda-13.0 ./build_debian.sh

Verify the installation with:

  /usr/local/cuda-13.0/bin/nvcc --version
EOF
        exit 1
    fi

    export PATH="${CUDA_ROOT}/bin:${PATH}"

    # LibTorch's Linux CUDA archive does not contain all NVIDIA runtimes.
    # Keep these versions aligned with PyTorch 2.13's CUDA 13.0 build.
    CUDNN_VERSION="${CUDNN_VERSION:-9.20.0.48}"
    CUSPARSELT_VERSION="${CUSPARSELT_VERSION:-0.8.1.1}"
    NCCL_DEBIAN_VERSION="${NCCL_DEBIAN_VERSION:-2.29.7-1+cuda13.2}"
    NVSHMEM_VERSION="${NVSHMEM_VERSION:-3.4.5}"
    CUDA_DEPS_DIR="${LIBTORCH_BASE}/cuda_deps_cudnn-${CUDNN_VERSION}_cusparselt-${CUSPARSELT_VERSION}_nvshmem-${NVSHMEM_VERSION}"

    CUDNN_DIR="${CUDA_DEPS_DIR}/cudnn"
    CUSPARSELT_DIR="${CUDA_DEPS_DIR}/cusparselt"
    NCCL_DIR="${CUDA_DEPS_DIR}/nccl"
    NVSHMEM_DIR="${CUDA_DEPS_DIR}/nvshmem"
    CUDNN_LIB_DIR="${CUDNN_DIR}/lib"
    CUSPARSELT_LIB_DIR="${CUSPARSELT_DIR}/lib"
    NCCL_LIB_DIR="${NCCL_DIR}/usr/lib/x86_64-linux-gnu"
    NVSHMEM_LIB_DIR="${NVSHMEM_DIR}/lib"

    install_nvidia_redist() {
        local archive_name="$1"
        local archive_url="$2"
        local destination="$3"
        local expected_library="$4"

        if [[ -e "${expected_library}" ]]; then
            return
        fi

        mkdir -p "${destination}"
        local redist_download_dir
        redist_download_dir="$(mktemp -d "${LIBTORCH_BASE}/.cuda-redist-download.XXXXXX")"
        trap 'rm -rf "${redist_download_dir}"' EXIT

        echo "Downloading ${archive_name}"
        curl --fail --location --retry 3 \
            --output "${redist_download_dir}/${archive_name}.tar.xz" \
            "${archive_url}/${archive_name}.tar.xz"
        tar -xJf "${redist_download_dir}/${archive_name}.tar.xz" \
            --strip-components=1 \
            -C "${destination}"

        if [[ ! -e "${expected_library}" ]]; then
            echo "Archive did not provide expected library: ${expected_library}" >&2
            exit 1
        fi

        rm -rf "${redist_download_dir}"
        trap - EXIT
    }

    install_nccl_debian_package() {
        local package_name="$1"
        local expected_file="$2"

        if [[ -e "${expected_file}" ]]; then
            return
        fi

        mkdir -p "${NCCL_DIR}"
        local nccl_download_dir
        local nccl_package
        nccl_download_dir="$(mktemp -d "${LIBTORCH_BASE}/.nccl-download.XXXXXX")"
        trap 'rm -rf "${nccl_download_dir}"' EXIT

        echo "Downloading ${package_name} ${NCCL_DEBIAN_VERSION}"
        (
            cd "${nccl_download_dir}"
            apt-get download "${package_name}=${NCCL_DEBIAN_VERSION}"
        )
        nccl_package="$(find "${nccl_download_dir}" -maxdepth 1 -type f \
            -name "${package_name}_*.deb" -print -quit)"
        if [[ -z "${nccl_package}" ]]; then
            echo "The downloaded ${package_name} package was not found" >&2
            exit 1
        fi

        dpkg-deb -x "${nccl_package}" "${NCCL_DIR}"
        if [[ ! -e "${expected_file}" ]]; then
            echo "${package_name} did not provide ${expected_file}" >&2
            exit 1
        fi

        rm -rf "${nccl_download_dir}"
        trap - EXIT
    }

    CUDNN_ARCHIVE="cudnn-linux-x86_64-${CUDNN_VERSION}_cuda13-archive"
    install_nvidia_redist \
        "${CUDNN_ARCHIVE}" \
        "https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64" \
        "${CUDNN_DIR}" \
        "${CUDNN_LIB_DIR}/libcudnn.so.9"

    CUSPARSELT_ARCHIVE="libcusparse_lt-linux-x86_64-${CUSPARSELT_VERSION}_cuda13-archive"
    install_nvidia_redist \
        "${CUSPARSELT_ARCHIVE}" \
        "https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-x86_64" \
        "${CUSPARSELT_DIR}" \
        "${CUSPARSELT_LIB_DIR}/libcusparseLt.so.0"

    NVSHMEM_ARCHIVE="libnvshmem-linux-x86_64-${NVSHMEM_VERSION}_cuda13-archive"
    install_nvidia_redist \
        "${NVSHMEM_ARCHIVE}" \
        "https://developer.download.nvidia.com/compute/nvshmem/redist/libnvshmem/linux-x86_64" \
        "${NVSHMEM_DIR}" \
        "${NVSHMEM_LIB_DIR}/libnvshmem_host.so.3"

    # The CUDA LibTorch binary itself has a DT_NEEDED entry for libnccl.so.2,
    # even when this application does not use the NCCL API directly.
    install_nccl_debian_package \
        libnccl2 \
        "${NCCL_LIB_DIR}/libnccl.so.2"

    if [[ "${USE_NCCL}" == "ON" ]]; then
        install_nccl_debian_package \
            libnccl-dev \
            "${NCCL_DIR}/usr/include/nccl.h"
    fi

    CUDA_DEPENDENCY_DIRS="${CUDNN_LIB_DIR};${CUSPARSELT_LIB_DIR};${NCCL_LIB_DIR};${NVSHMEM_LIB_DIR}"
    CUDA_DEPENDENCY_LIBRARY_PATH="${CUDNN_LIB_DIR}:${CUSPARSELT_LIB_DIR}:${NCCL_LIB_DIR}:${NVSHMEM_LIB_DIR}"
    export LD_LIBRARY_PATH="${CUDA_DEPENDENCY_LIBRARY_PATH}:${CUDA_ROOT}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    CMAKE_CUDA_ARGS=(
        "-DCMAKE_CUDA_COMPILER=${CUDA_ROOT}/bin/nvcc"
        "-DCUDAToolkit_ROOT=${CUDA_ROOT}"
        "-DCUDA_TOOLKIT_ROOT_DIR=${CUDA_ROOT}"
        "-DLIBTORCH_CUDA_DEPENDENCY_DIRS=${CUDA_DEPENDENCY_DIRS}"
        "-DLIBTORCH_FRAMEWORK_USE_NCCL=${USE_NCCL}"
        "-DLIBTORCH_NCCL_ROOT=${NCCL_DIR}/usr"
    )
fi

cmake -S . -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLIBTORCH_FRAMEWORK_SOURCE_DIR="${LIBTORCH_FRAMEWORK_DIR}/LibTorchFramework" \
    -DLIBTORCH_CONFIGURATION=Release \
    -DLIBTORCH_ROOT="${LIBTORCH_ROOT}" \
    -DLIBTORCH_FRAMEWORK_USE_CUDA="${USE_CUDA}" \
    "${CMAKE_CUDA_ARGS[@]}" \
    -DPLAYGROUND_ROOT=${PLAYGROUND_DIR} \
	-DPLAYGROUND_INCLUDE_DIR=${PLAYGROUND_DIR}/include_debian \
    -DPLAYGROUND_LIBRARY_RELEASE=${PLAYGROUND_DIR}/Playground/build_cmake_debian/libPlayground.a

#cmake --build "${BUILD_DIR}" --parallel 1
cmake --build "${BUILD_DIR}" --parallel 8

mkdir -p "${BIN_DIR}"
cp -f "${BUILD_DIR}/LibTorchFramework" "${BIN_DIR}/LibTorchFramework"

echo "Build output: ${BUILD_DIR}/LibTorchFramework"
echo "Binary copied to: ${BIN_DIR}/LibTorchFramework"
