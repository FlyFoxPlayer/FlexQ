
cmake_minimum_required(VERSION 3.8)
project(flexqgemm)

if(FIND_SM_80 STREQUAL True)
    add_definitions(-DGPU_ARCH=80)
else()
    if(FIND_SM_75 STREQUAL True)
        add_definitions(-DGPU_ARCH=75)
    endif()
endif()

file(GLOB FlexQ_FILES ${FlexQ_FILES} ./common/*.cu)
file(GLOB FlexQ_FILES ${FlexQ_FILES} ./*.cu)
file(GLOB FlexQ_FILES ${FlexQ_FILES} ./src/bgemm/*.cu)
file(GLOB FlexQ_FILES ${FlexQ_FILES} ./src/pack/*.cu)

add_library(flexqgemm STATIC flexq_gemm_wrapper.cu ${FlexQ_FILES})
set_property(TARGET flexqgemm PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET flexqgemm PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

