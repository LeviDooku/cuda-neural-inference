//Pedro Velasco Santana
//File: cpu_inference.h

//Purpose: sequential inference in CPU

#ifndef CPU_INFERENCE_H
#define CPU_INFERENCE_H

#include "neural_network.h"

void cpu_inference(const neural_network& network, const float* entries, float* output, const int num_samples);

#endif