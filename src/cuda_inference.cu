//Pedro Velasco Santana
//File: cuda_inference.cu

//Purpose: parallel inference in GPU

#include <iostream>
#include "cuda_inference.cuh"
#include "neural_network.h"

__constant__ neural_network constant_network;

__global__ void inference_kernel(const float* entries, float* output, int num_samples){
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < num_samples){
        int base = id * NUM_INPUTS;

        float hidden[NUM_HIDDEN];

        for(int h = 0; h < NUM_HIDDEN; ++h){
            float sum = constant_network.hidden_biases[h];

            for(int k = 0; k < NUM_INPUTS; ++k)
                sum += entries[base + k] * constant_network.input_hidden_weights[h * NUM_INPUTS + k];

            hidden[h] = relu(sum);
        }

        float output_sum = constant_network.output_bias;

        for(int h = 0; h < NUM_HIDDEN; ++h){
            output_sum += hidden[h] * constant_network.output_hidden_weights[h];
        }

        output[id] = sigmoid(output_sum);
    }
}

void cuda_inference(const neural_network& network, const float* entries, float* output, const int num_samples, float* kernel_time_ms){
    float* constant_entries = nullptr;
    float* constant_output = nullptr;

    int entries_size = num_samples * NUM_INPUTS * sizeof(float);
    int output_size = num_samples * sizeof(float);

    cudaMalloc((void**)&constant_entries, entries_size);
    cudaMalloc((void**)&constant_output, output_size);

    cudaMemcpy(constant_entries, entries, entries_size, cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(constant_network, &network, sizeof(neural_network));

    int threads_per_block = 256;
    int blocks_per_grid = (num_samples + threads_per_block - 1) / threads_per_block;

    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    inference_kernel<<<blocks_per_grid, threads_per_block>>>(constant_entries,constant_output, num_samples);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(kernel_time_ms, start, stop);

    cudaMemcpy(output, constant_output, output_size, cudaMemcpyDeviceToHost);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(constant_entries);
    cudaFree(constant_output);
}