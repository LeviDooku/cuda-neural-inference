//Pedro Velasco Santana
//File: cpu_inference.cpp

//Purpose: sequential inference in CPU

#include "cpu_inference.h"

void cpu_inference(const neural_network& network, const float* entries, float* output, const int num_samples){
    for(int i = 0; i < num_samples; ++i){
        int base = i * NUM_INPUTS;
        float sample_out = forward_sample(network, &entries[base]);

        output[i] = sample_out >= 0.5 ? 1 : 0;
    }
}