sudo apt-get install -y \
    build-essential \
    cmake \
    unzip \
    curl \
    ninja-build \
    pkg-config \
	libicu-dev \
	libprotobuf-dev \
	protobuf-compiler \
	cudnn9-cuda-13

LIBTORCH_FRAMEWORK_DIR="/mnt/e/Programming/Cpp/LibTorchFramework"
PLAYGROUND_DIR="/mnt/d/Martin/Programming/test/Playground"

# Override these on the command line when another release or CUDA build is
# needed, for example:
#   LIBTORCH_VERSION=2.11.0 LIBTORCH_VARIANT=cu130 ./build_debian.sh
LIBTORCH_VERSION="${LIBTORCH_VERSION:-2.10.0}"
LIBTORCH_VARIANT="${LIBTORCH_VARIANT:-cu130}"
#LIBTORCH_VARIANT="cpu"

BIN_DIR="${LIBTORCH_FRAMEWORK_DIR}/bin/${LIBTORCH_VARIANT}"

LIBTORCH_BASE="${LIBTORCH_FRAMEWORK_DIR}/libtorch_debian/Release_${LIBTORCH_VARIANT}"
LIBTORCH_ROOT="${LIBTORCH_BASE}/libtorch_${LIBTORCH_VERSION}"
#BUILD_DIR="${BUILD_DIR:-/tmp/libtorchframework-build-release}"
BUILD_DIR="${LIBTORCH_FRAMEWORK_DIR}/LibTorchFramework/build_cmake_debian_${LIBTORCH_VARIANT}"
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
    export LD_LIBRARY_PATH="${CUDA_ROOT}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    CMAKE_CUDA_ARGS=(
        "-DCMAKE_CUDA_COMPILER=${CUDA_ROOT}/bin/nvcc"
        "-DCUDAToolkit_ROOT=${CUDA_ROOT}"
        "-DCUDA_TOOLKIT_ROOT_DIR=${CUDA_ROOT}"
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