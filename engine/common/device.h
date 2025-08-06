#pragma once

#include<cuda_runtime.h>

// Get GPU multiProcessorCount
static const int DEVICEPROP_SMCOUNT = []()->int {
  // onece call
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  return deviceProp.multiProcessorCount;
}();