# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(conway)

set(Python3_FIND_VIRTUALENV FIRST)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH})

# Help Torch find nvtx if it's in the standard CUDA targets location
if(NOT nvtx3_dir AND EXISTS "/usr/local/cuda/targets/x86_64-linux/include/nvtx3")
    set(nvtx3_dir "/usr/local/cuda/targets/x86_64-linux/include")
endif()

find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")

find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)


add_library(conway SHARED conway.cpp conway.cu)
set_target_properties(conway PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                                        EXPORT_COMPILE_COMMANDS ON
                                        CMAKE_CXX_STANDARD 17
                                        PREFIX ""
                                        SUFFIX ".${Python3_SOABI}.so")
target_include_directories(conway PRIVATE ${Python3_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})
target_link_libraries(conway ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY} ${CUDA_LIBRARIES} ${Python3_LIBRARIES} OpenMP::OpenMP_CXX)
target_compile_options(conway PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
