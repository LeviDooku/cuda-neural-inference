//Pedro Velasco Santana
//File: cuda_inference.cu

//Purpose: parallel inference in GPU

#include <iostream>
#include "cuda_inference.cuh"

__constant__ neural_network constant_network;

__global__ void inference_kernel(const float* entries, float* output, int num_samples){
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < num_samples){
        int base = id * NUM_INPUTS;
        output[id] = forward_sample(constant_network, &entries[base]);
    }
}

void cuda_inference(const neural_network& network, const float* entries, float* output, const int num_samples, float* kernel_time_ms){
    float* device_entries = nullptr;
    float* device_output = nullptr;

    int entries_size = num_samples * NUM_INPUTS * sizeof(float);
    int output_size = num_samples * sizeof(float);

    cudaMalloc((void**)&device_entries, entries_size);
    cudaMalloc((void**)&device_output, output_size);

    cudaMemcpy(device_entries, entries, entries_size, cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(constant_network, &network, sizeof(neural_network));

    int threads_per_block = 256;
    int blocks_per_grid = (num_samples + threads_per_block - 1) / threads_per_block;

    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    inference_kernel<<<blocks_per_grid, threads_per_block>>>(device_entries,device_output, num_samples);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(kernel_time_ms, start, stop);

    cudaMemcpy(output, device_output, output_size, cudaMemcpyDeviceToHost);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(device_entries);
    cudaFree(device_output);
}