// Pedro Velasco Santana
// File: prueba.cpp

// Purpose: basic test for dataset, timer and neural network

#include <iostream>

#include "dataset.h"
#include "neural_network.h"
#include "timer.h"

int main() {
    const int NUM_SAMPLES = 10;

    dataset data;
    neural_network network;
    Timer timer;

    std::cout << "Creating dataset..." << std::endl;

    timer.start();
    create_dataset(data, NUM_SAMPLES);
    double dataset_time = timer.stop_milliseconds();

    std::cout << "Dataset created in " << dataset_time << " ms" << std::endl;

    std::cout << std::endl;
    std::cout << "Samples before normalization:" << std::endl;

    show_sample(data, 0);
    std::cout << std::endl << std::endl;

    show_sample(data, 1);
    std::cout << std::endl << std::endl;

    std::cout << "Normalizing dataset..." << std::endl;
    normalize_dataset(data);

    std::cout << std::endl;
    std::cout << "Samples after normalization:" << std::endl;

    show_sample(data, 0);
    std::cout << std::endl << std::endl;

    show_sample(data, 1);
    std::cout << std::endl << std::endl;

    std::cout << "Initializing neural network..." << std::endl;
    initialize_network(network);

    std::cout << "Running forward pass for sample 0..." << std::endl;

    float* sample = &data.entries[0];
    float output = forward_sample(network, sample);

    std::cout << "Network output: " << output << std::endl;

    if (output >= 0.0f && output <= 1.0f) {
        std::cout << "Forward pass OK: output is between 0 and 1" << std::endl;
    } else {
        std::cout << "Forward pass ERROR: output is outside [0, 1]" << std::endl;
    }

    release_dataset(data);

    return 0;
}