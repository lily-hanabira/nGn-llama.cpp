@echo off
rem nGn MoE offload PoC: fast local CUDA build of this checkout (Blade, laptop RTX 4070 = sm_89).
rem Not a release runtime: one CUDA architecture, native CPU, static backends. Release builds stay
rem with nyaa-bastion\deploy\ngn\build-runtime-windows.cmd.
rem
rem   build-poc.cmd [cuda architectures]      (default 89-real)
rem
rem Needs VS 2019 Build Tools (MSVC 14.29; nvcc 12.8 crashes on MSVC 14.5x), CUDA 12.8, CMake + Ninja.
setlocal
set "HERE=%~dp0"
for %%i in ("%HERE%..\..") do set "SRC=%%~fi"
set "LOG=%HERE%logs\build.log"
set "ARCH=%~1"
if "%ARCH%"=="" set "ARCH=89-real"
rem Find VS 2019 (any edition) for MSVC 14.29, and CMake/Ninja from PATH, standalone CMake or any VS 2022/2026.
if not defined VS2019 for %%e in (BuildTools Community Professional Enterprise) do if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\%%e\VC\Auxiliary\Build\vcvars64.bat" if not defined VS2019 set "VS2019=C:\Program Files (x86)\Microsoft Visual Studio\2019\%%e"
if not defined CUDA_PATH_V12_8 set "CUDA_PATH_V12_8=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
if not defined NGN_POC_BUILD set "NGN_POC_BUILD=C:\ngn\poc-build"
set "TOOLS="
for %%v in (18 2022) do for %%e in (Community Professional Enterprise BuildTools) do if exist "C:\Program Files\Microsoft Visual Studio\%%v\%%e\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe" if not defined TOOLS set "TOOLS=C:\Program Files\Microsoft Visual Studio\%%v\%%e\Common7\IDE\CommonExtensions\Microsoft\CMake"

> "%LOG%" echo build-poc %DATE% %TIME% src=%SRC% arch=%ARCH% build=%NGN_POC_BUILD% vs2019=%VS2019% tools=%TOOLS%
if not defined VS2019 ( >> "%LOG%" echo MISSING VS 2019 with C++ tools & echo MISSING VS 2019 with C++ tools ^(MSVC 14.29^) & exit /b 1 )
if not exist "%CUDA_PATH_V12_8%\bin\nvcc.exe" ( >> "%LOG%" echo MISSING CUDA 12.8 at %CUDA_PATH_V12_8% & echo MISSING CUDA 12.8 & exit /b 1 )
call "%VS2019%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
set "CUDA_PATH=%CUDA_PATH_V12_8%"
set "PATH=%CUDA_PATH%\bin;%PATH%;C:\Program Files\CMake\bin"
if defined TOOLS set "PATH=%PATH%;%TOOLS%\CMake\bin;%TOOLS%\Ninja"
where cmake >> "%LOG%" 2>&1 || ( echo MISSING cmake & exit /b 1 )
where ninja >> "%LOG%" 2>&1 || ( echo MISSING ninja & exit /b 1 )
cl 2>&1 | findstr /i "Version" >> "%LOG%"
nvcc --version | findstr /i "release" >> "%LOG%"
for /f %%c in ('git -C "%SRC%" rev-parse HEAD') do >> "%LOG%" echo commit %%c
git -C "%SRC%" status --short >> "%LOG%"

cmake -S "%SRC%" -B "%NGN_POC_BUILD%" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_CUDA=ON ^
  -DCMAKE_CUDA_ARCHITECTURES=%ARCH% -DCMAKE_CUDA_COMPILER="%CUDA_PATH%\bin\nvcc.exe" -DLLAMA_CURL=OFF ^
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON >> "%LOG%" 2>&1 || ( echo configure failed, see %LOG% & exit /b 2 )
cmake --build "%NGN_POC_BUILD%" --config Release --target llama-bench llama-server llama-perplexity -j %NUMBER_OF_PROCESSORS% >> "%LOG%" 2>&1 || ( echo build failed, see %LOG% & exit /b 3 )
>> "%LOG%" echo BUILD OK %DATE% %TIME%
echo BUILD OK: %NGN_POC_BUILD%\bin
endlocal
