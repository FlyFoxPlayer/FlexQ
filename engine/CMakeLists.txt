cmake_minimum_required(VERSION 3.10)

project(FlexQ LANGUAGES CXX)

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin/${PLATFORM_ABI})

set(CXX_STD "14" CACHE STRING "C++ standard")
set(CMAKE_CXX_STANDARD "${CXX_STD}")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# setting comman compiler flags
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()
message("CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++14")
set(CMAKE_C_FLAGS_DEBUG    "${CMAKE_C_FLAGS_DEBUG}  -g -O0")
set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
if (MSVC)
    set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}    -w")
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -w")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -w")
    message("MSVC OPEN!")
endif()

## set common include directories 
set(COMMON_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}
)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CUDA.cmake)

set(CUTLASS_HEADER_DIR ${PROJECT_SOURCE_DIR}/../3rdparty/cutlass/include)
list(APPEND CUTLASS_HEADER_DIR ${PROJECT_SOURCE_DIR}/../3rdparty/cutlass/tools/util/include)


# setting nvcc compiler flags
set(SM_SETS 75 80 86 89 90)
set(USING_WMMA False)
set(FIND_SM False)
set(FIND_SM_80 False)
set(FIND_SM_75 False)
foreach(SM_NUM IN LISTS SM_SETS)
    string(FIND "${SM}" "${SM_NUM}" SM_POS)
    if(SM_POS GREATER -1)
        set(FIND_SM True)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_${SM_NUM},code=\\\"sm_${SM_NUM},compute_${SM_NUM}\\\"")
        if (SM_NUM STREQUAL 75 OR SM_NUM STREQUAL 80 OR SM_NUM STREQUAL 86 OR SM_NUM STREQUAL 89 OR SM_NUM STREQUAL 90)
            set(USING_WMMA True)        
        endif()
        if (SM_NUM STREQUAL 75)
            set(FIND_SM_75 True)        
        endif()
        if (SM_NUM STREQUAL 80 OR SM_NUM STREQUAL 86 OR SM_NUM STREQUAL 89 OR SM_NUM STREQUAL 90)
            set(FIND_SM_80 True)        
        endif()
        list(APPEND CMAKE_CUDA_ARCHITECTURES ${SM_NUM})
        message("-- Assign GPU architecture (sm=${SM_NUM})")
    endif()
endforeach()
if(USING_WMMA STREQUAL True)
    set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}    -DWMMA")
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -DWMMA")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DWMMA")
    message("-- Use WMMA")
endif()

if(FIND_SM_80 STREQUAL True)
    add_definitions(-DGPU_ARCH=80)
else()
    if(FIND_SM_75 STREQUAL True)
        add_definitions(-DGPU_ARCH=75)
    endif()
endif()

if(NOT (FIND_SM STREQUAL True))
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  \
                        -gencode=arch=compute_75,code=\\\"sm_75,compute_75\\\" \
                        -gencode=arch=compute_80,code=\\\"sm_80,compute_80\\\" \
                        -gencode=arch=compute_86,code=\\\"sm_86,compute_86\\\" \
                        -gencode=arch=compute_89,code=\\\"sm_86,compute_89\\\" \
                        ")
    set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}    -DWMMA")
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -DWMMA")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DWMMA")
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)
    message("-- Assign GPU architecture (sm=75,80,86,89)")
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++${CXX_STD}") # -DCUDA_PTX_FP8_F2FP_ENABLED
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -Xcompiler -O2")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -Xcompiler -O0 -G")
if(BUILD_FAST_MATH)
    set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} --use_fast_math")
    message("-- Open use_fast_math")
endif()
# print compile flgs
message("CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
message("CMAKE_CUDA_FLAGS_DEBUG: ${CMAKE_CUDA_FLAGS_DEBUG}")
message("CMAKE_CUDA_FLAGS_RELEASE: ${CMAKE_CUDA_FLAGS_RELEASE}")

# cuda backend header dirs and libs append
list(APPEND COMMON_HEADER_DIRS ${CUDA_PATH}/include)
list(APPEND COMMON_HEADER_DIRS ${CUTLASS_HEADER_DIR})
list(APPEND COMMON_LIB_DIRS ${CUDA_PATH}/lib/x64)

# Add header files and library dependencies
include_directories( SYSTEM
  ${COMMON_HEADER_DIRS}
)
link_directories(
  ${COMMON_LIB_DIRS}
)

# Add kernel common header file
include_directories(&{PROJECT_SOURCE_DIR}/common)

# Add kernel impl
include_directories(&{PROJECT_SOURCE_DIR}/src/pack)

set(BGEMM_FILES "" CACHE INTERNAL "bgemm files")
file(GLOB BGEMM_FILES ${BGEMM_FILES} ./src/bgemm/*.cu)
file(GLOB BGEMM_FILES ${BGEMM_FILES} ./src/pack/*.cu)

add_library(bgemm_op STATIC ${BGEMM_FILES})
set_property(TARGET bgemm_op PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET bgemm_op PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(bgemm_op PUBLIC -lcudart)

file(GLOB TEST_KERNEL_FILES ${TEST_KERNEL_FILES} ./test/*.cu)
add_library(test_kernel STATIC ${TEST_KERNEL_FILES})
set_property(TARGET test_kernel PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET test_kernel PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(test_kernel PUBLIC -lcudart bgemm_op)

add_executable(test_bgemm_kernel test_bgemm_kernel.cu)
target_link_libraries(test_bgemm_kernel bgemm_op test_kernel)

add_executable(test_packing_kernel test_packing_kernel.cu)
target_link_libraries(test_packing_kernel bgemm_op test_kernel)

add_executable(test_cublas_kernel test_cublas_kernel.cu)
target_link_libraries(test_cublas_kernel PRIVATE cublas)