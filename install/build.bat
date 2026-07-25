@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem ============================================================================
rem Build LibTorch on Windows using MSVC 2022 + CUDA
rem ============================================================================

rem ---- Configuration ---------------------------------------------------------

set BUILD_TYPE=Debug

set PYTORCH_VERSION=main
set ROOT_DIR=%~dp0
set SOURCE_DIR=%ROOT_DIR%pytorch
set BUILD_DIR=%ROOT_DIR%build-libtorch
set INSTALL_DIR=%ROOT_DIR%libtorch_%BUILD_TYPE%

rem Set to 1 to fetch, checkout and update submodules for an existing clone.
rem Set to 0 to use the existing source tree without any Git updates.
set GIT_UPDATE=0

set PYTHON_INSTALL=0

rem Delete build directory before compiling.
rem 1 = full clean rebuild
rem 0 = incremental build (reuse existing object files)
set CLEAN_BUILD=0

rem Change this to your installed CUDA Toolkit.
set CUDA_PATH=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0

rem CUDA architectures to compile.
rem Examples:
rem   RTX 20xx: 7.5
rem   RTX 30xx: 8.6
rem   RTX 40xx: 8.9
rem   RTX 50xx: 12.0
set TORCH_CUDA_ARCH_LIST=8.6;8.9

rem Limit parallel compilation if RAM usage is too high.
set MAX_JOBS=4

rem ON creates DLLs and import libraries, which is recommended on Windows.
set BUILD_SHARED_LIBS=ON

rem ---- Validate MSVC environment ---------------------------------------------

where cl.exe >nul 2>&1
if errorlevel 1 (
    echo ERROR: MSVC environment is not initialized.
    echo Run this script from:
    echo x64 Native Tools Command Prompt for VS 2022
    exit /b 1
)

where link.exe >nul 2>&1
if errorlevel 1 (
    echo ERROR: MSVC linker was not found.
    exit /b 1
)

echo MSVC environment detected:
cl.exe 2>&1 | findstr /C:"Version"

rem ---- Validate dependencies --------------------------------------------------

where cl.exe >nul 2>&1
if errorlevel 1 (
    echo ERROR: cl.exe was not found.
    exit /b 1
)

where git.exe >nul 2>&1
if errorlevel 1 (
    echo ERROR: Git was not found.
    exit /b 1
)

where python.exe >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python was not found.
    exit /b 1
)

if not exist "%CUDA_PATH%/bin/nvcc.exe" (
    echo ERROR: CUDA nvcc was not found:
    echo "%CUDA_PATH%/bin/nvcc.exe"
    exit /b 1
)

set PATH=%CUDA_PATH%/bin;%CUDA_PATH%/libnvvp;%PATH%
set CUDACXX=%CUDA_PATH%/bin/nvcc.exe
set CUDA_HOME=%CUDA_PATH%
set CUDA_BIN_PATH=%CUDA_PATH%

echo.
echo MSVC:
cl.exe 2>&1 | findstr /C:"Version"
echo.
echo CUDA:
nvcc.exe --version
echo.

rem ---- Clone PyTorch ----------------------------------------------------------

if not exist "%SOURCE_DIR%\.git" (
    git clone --recursive https://github.com/pytorch/pytorch.git "%SOURCE_DIR%"

    if errorlevel 1 (
        echo ERROR: Failed to clone PyTorch.
        exit /b 1
    )
) else (
    if "%GIT_UPDATE%"=="1" (
        echo Updating existing PyTorch repository...

        pushd "%SOURCE_DIR%"

        git fetch origin

        if errorlevel 1 (
            popd
            echo ERROR: Failed to fetch PyTorch repository.
            exit /b 1
        )

        git checkout "%PYTORCH_VERSION%"

        if errorlevel 1 (
            popd
            echo ERROR: Failed to checkout %PYTORCH_VERSION%.
            exit /b 1
        )

        git pull --ff-only origin "%PYTORCH_VERSION%"

        if errorlevel 1 (
            popd
            echo ERROR: Failed to update %PYTORCH_VERSION%.
            exit /b 1
        )

        git submodule sync
        git submodule update --init --recursive

        if errorlevel 1 (
            popd
            echo ERROR: Failed to update PyTorch submodules.
            exit /b 1
        )

        popd
    ) else (
        echo Using existing PyTorch source tree without Git update:
        echo   %SOURCE_DIR%
    )
)

rem ---- Create Python environment ---------------------------------------------

if not exist "%ROOT_DIR%venv/Scripts/python.exe" (
    python -m venv "%ROOT_DIR%venv"

    if errorlevel 1 (
        echo ERROR: Failed to create Python virtual environment.
        exit /b 1
    )
)

call "%ROOT_DIR%venv/Scripts/activate.bat"

if "%PYTHON_INSTALL%"=="1" (

	python -m pip install --upgrade pip setuptools wheel

	if errorlevel 1 (
		echo ERROR: Failed to update Python build tools.
		exit /b 1
	)

	pushd "%SOURCE_DIR%"

	python -m pip install --group dev
	python -m pip install ninja cmake mkl-static mkl-include

	if errorlevel 1 (
		popd
		echo ERROR: Failed to install PyTorch build dependencies.
		exit /b 1
	)
)


rem ---- Normalize every CMake path --------------------------------------------

set "CMAKE_ROOT_DIR=%ROOT_DIR:\=/%"
set "CMAKE_SOURCE_DIR=%SOURCE_DIR:\=/%"
set "CMAKE_BUILD_DIR=%BUILD_DIR:\=/%"
set "CMAKE_VENV_DIR=%ROOT_DIR:\=/%venv"
set "CMAKE_INSTALL_DIR=%INSTALL_DIR:\=/%"

if "%CMAKE_ROOT_DIR:~-1%"=="/" set "CMAKE_ROOT_DIR=%CMAKE_ROOT_DIR:~0,-1%"
if "%CMAKE_SOURCE_DIR:~-1%"=="/" set "CMAKE_SOURCE_DIR=%CMAKE_SOURCE_DIR:~0,-1%"
if "%CMAKE_BUILD_DIR:~-1%"=="/" set "CMAKE_BUILD_DIR=%CMAKE_BUILD_DIR:~0,-1%"
if "%CMAKE_INSTALL_DIR:~-1%"=="/" set "CMAKE_INSTALL_DIR=%CMAKE_INSTALL_DIR:~0,-1%"

rem ---- Build configuration ----------------------------------------------------

rem Conda environments can inject BLAS=APL.
set BLAS=MKL
set BLAS_HOME=
set APL_ROOT=
set APL_DIR=
set APL_INCLUDE_DIR=
set APL_LIBRARIES=

set USE_CUDA=1
set USE_CUDNN=1
set USE_ROCM=0
set USE_XPU=0
set USE_DISTRIBUTED=0
set USE_MPI=0
set USE_GLOO=0
set USE_NCCL=0
set USE_MKLDNN=1
set BUILD_TEST=0
set BUILD_PYTHON=0
set BUILD_CAFFE2=0
set BUILD_CAFFE2_OPS=0

set CMAKE_BUILD_TYPE=%BUILD_TYPE%
set CMAKE_GENERATOR=Ninja

rem These must use forward slashes. build_libtorch.py passes them to CMake.
set "CMAKE_INSTALL_PREFIX=%CMAKE_INSTALL_DIR%"
set "CMAKE_PREFIX_PATH=%CMAKE_VENV_DIR%;%CMAKE_VENV_DIR%/Lib/site-packages"

rem Force CMake to store the install prefix as a PATH cache entry.
set "CMAKE_ARGS=-DBLAS=MKL -DCMAKE_INSTALL_PREFIX:PATH=%CMAKE_INSTALL_DIR%"

set DISTUTILS_USE_SDK=1

rem Nsight Compute/NVTX must be installed as part of the CUDA Toolkit.
if not exist "%CUDA_PATH%/include/nvtx3/nvToolsExt.h" (
    echo WARNING: NVTX headers were not found.
    echo Re-run the CUDA installer and install Nsight Compute.
)

rem ---- Clean previous build ---------------------------------------------------

if "%CLEAN_BUILD%"=="1" (
    echo Performing clean build...

    if exist "%BUILD_DIR%" (
        rmdir /s /q "%BUILD_DIR%"
    )

    if exist "%INSTALL_DIR%" (
        rmdir /s /q "%INSTALL_DIR%"
    )
) else (
    echo Performing incremental build...
)

if not exist "%BUILD_DIR%" (
    mkdir "%BUILD_DIR%"
)


rem ---- Compile LibTorch -------------------------------------------------------

pushd "%BUILD_DIR%"

if "%CLEAN_BUILD%"=="1" (
    python "%SOURCE_DIR%\tools\build_libtorch.py" --rerun-cmake
) else (
    python "%SOURCE_DIR%\tools\build_libtorch.py"
)

if errorlevel 1 (
    echo.
    echo Re-running failed install target with verbose output...
    echo.

    cmake --build "%BUILD_DIR%\build" ^
        --target install ^
        --config "%BUILD_TYPE%" ^
        --verbose

    popd
    echo.
    echo ERROR: LibTorch compilation or installation failed.
    exit /b 1
)

popd

cmake --install "%BUILD_DIR%\build" ^
    --config "%BUILD_TYPE%" ^
    --prefix "%INSTALL_DIR%"

if errorlevel 1 (
    popd
    echo.
    echo ERROR: LibTorch installation failed.
    exit /b 1
)

popd

rem ---- Validate output --------------------------------------------------------

if not exist "%INSTALL_DIR%/share/cmake/Torch/TorchConfig.cmake" (
    echo ERROR: TorchConfig.cmake was not produced.
    exit /b 1
)

if not exist "%INSTALL_DIR%/lib/torch.lib" (
    echo WARNING: torch.lib was not found at the expected location.
    echo Check "%INSTALL_DIR%/lib".
)

echo.
echo ============================================================================
echo LibTorch CUDA build completed successfully.
echo.
echo Output:
echo   %INSTALL_DIR%
echo.
echo Use from CMake with:
echo   -DCMAKE_PREFIX_PATH="%INSTALL_DIR%"
echo ============================================================================
echo.

endlocal
exit /b 0