//Pedro Velasco Santana
//File: cuda_inference.cuh

//Purpose: parallel inference in GPU

#ifndef CUDA_INFERENCE_CUH
#define CUDA_INFERENCE_CUH

#include "neural_network.h"

void cuda_inference(const neural_network& network, const float* entries, float* output, const int num_samples, float* kernel_time_ms);

#endif;