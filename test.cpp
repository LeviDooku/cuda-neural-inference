// Pedro Velasco Santana
// File: prueba.cpp

// Purpose: basic testing for CPU and GPU inference

#include<iostream>
#include <cstdlib>

#include "dataset.h"
#include "neural_network.h"
#include "timer.h"
#include "cpu_inference.h"
#include "cuda_inference.cuh"

int main(int argc, char *argv[]){
    if(argc < 5){
        std::cerr << "ERROR: usage: " << argv[0] << "<samples_training> <epoch> <learning_rate> <samples_inference>" << std::endl;
        return -1;
    }

    const int NUM_SAMPLES_TRAINING = atoi(argv[1]);
    
    if(NUM_SAMPLES_TRAINING <= 0){
        std::cerr << "ERROR: samples_training must be > 0" << std::endl;
        return -1;
    }

    const int EPOCH = atoi(argv[2]);

    if(EPOCH <= 0){
        std::cerr << "ERROR: epoch must be > 0" << std::endl;
        return -1;
    }

    const float LEARNING_RATE = atof(argv[3]);

    if(LEARNING_RATE <= 0.0f){
        std::cerr << "ERROR: learning_rate must be > 0" << std::endl;
        return -1;
    }

    const int NUM_SAMPLES_INFERENCE = atoi(argv[4]);

    if(NUM_SAMPLES_INFERENCE <= 0){
        std::cerr << "ERROR: samples_inference must be > 0" << std::endl;
        return -1;
    }

    dataset training_data, inference_data;
    neural_network network;

    //Create and normalize the training and inference data

    create_dataset(training_data, NUM_SAMPLES_TRAINING);
    create_dataset(inference_data, NUM_SAMPLES_INFERENCE);

    normalize_dataset(training_data);
    normalize_dataset(inference_data);

    //Initialize and train neural network

    initialize_network(network);
    train_network_cpu(network, training_data.entries, training_data.tags, NUM_SAMPLES_TRAINING, EPOCH, LEARNING_RATE);

    //Declare output arrays

    float* cpu_outputs = new float[NUM_SAMPLES_INFERENCE];
    float* gpu_outputs = new float[NUM_SAMPLES_INFERENCE];

    double cpu_accuracy = 0.0;
    double gpu_accuracy = 0.0;

    int cpu_success = 0;
    int gpu_success = 0;

    //Declare timers

    Timer t_inference_cpu, t_total_gpu;

    //CPU inference

    t_inference_cpu.start();
    cpu_inference(network, inference_data.entries, cpu_outputs, NUM_SAMPLES_INFERENCE);
    double cpu_time = t_inference_cpu.stop_milliseconds();

    //GPU inference

    float kernel_time_ms = 0.0f;

    t_total_gpu.start();
    cuda_inference(network, inference_data.entries, gpu_outputs, NUM_SAMPLES_INFERENCE, &kernel_time_ms);
    double gpu_total_time = t_total_gpu.stop_milliseconds();

    //Calculate accuracy

    for(int i = 0; i < NUM_SAMPLES_INFERENCE; ++i){
        int cpu_prediction = cpu_outputs[i] >= 0.5f ? 1 : 0;
        int gpu_prediction = gpu_outputs[i] >= 0.5f ? 1 : 0;
        int real_tag = inference_data.tags[i] >= 0.5f ? 1 : 0;

        if(cpu_prediction == real_tag)
            cpu_success++;

        if(gpu_prediction == real_tag)
            gpu_success++;
    }

    cpu_accuracy = 100.0 * cpu_success / NUM_SAMPLES_INFERENCE;
    gpu_accuracy = 100.0 * gpu_success / NUM_SAMPLES_INFERENCE;

    //Calculate memory

    size_t entries_memory_bytes = NUM_SAMPLES_INFERENCE * NUM_INPUTS * sizeof(float);
    size_t output_memory_bytes = NUM_SAMPLES_INFERENCE * sizeof(float);
    size_t network_memory_bytes = sizeof(neural_network);
    size_t total_gpu_memory_bytes = entries_memory_bytes + output_memory_bytes + network_memory_bytes;

    double total_gpu_memory_mb = total_gpu_memory_bytes / (1024.0 * 1024.0);

    std::cout << NUM_SAMPLES_INFERENCE << " " << cpu_time << " " << gpu_total_time << " " << kernel_time_ms << " " << cpu_accuracy << " " << gpu_accuracy << " " << total_gpu_memory_mb << std::endl;

    delete[] cpu_outputs;
    delete[] gpu_outputs;

    release_dataset(training_data);
    release_dataset(inference_data);
}