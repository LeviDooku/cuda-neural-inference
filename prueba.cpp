// Pedro Velasco Santana
// File: prueba.cpp

// Purpose: basic test for dataset, timer and neural network

#include <iostream>

#include "dataset.h"
#include "neural_network.h"
#include "timer.h"

int main() {
    const int NUM_SAMPLES = 10000;
    const int DEBUG_SAMPLES = 10;

    dataset data;
    neural_network network;
    Timer timer;

    std::cout << "Creating dataset..." << std::endl;

    create_dataset(data, NUM_SAMPLES);

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

    std::cout << std::endl;
    std::cout << "Some weights before training:" << std::endl;
    std::cout << "input_hidden_weights[0]: " << network.input_hidden_weights[0] << std::endl;
    std::cout << "input_hidden_weights[1]: " << network.input_hidden_weights[1] << std::endl;
    std::cout << "output_hidden_weights[0]: " << network.output_hidden_weights[0] << std::endl;
    std::cout << "output_hidden_weights[1]: " << network.output_hidden_weights[1] << std::endl;

    std::cout << std::endl;
    std::cout << "Training neural network..." << std::endl;

    train_network_cpu(network, data.entries, data.tags, NUM_SAMPLES, 1000, 0.01);

    std::cout << "Training finished." << std::endl;

    std::cout << std::endl;
    std::cout << "Some weights after training:" << std::endl;
    std::cout << "input_hidden_weights[0]: " << network.input_hidden_weights[0] << std::endl;
    std::cout << "input_hidden_weights[1]: " << network.input_hidden_weights[1] << std::endl;
    std::cout << "output_hidden_weights[0]: " << network.output_hidden_weights[0] << std::endl;
    std::cout << "output_hidden_weights[1]: " << network.output_hidden_weights[1] << std::endl;

    dataset real_data;

    create_dataset(real_data, 2000);
    normalize_dataset(real_data);

    int aciertos = 0;

    std::cout << std::endl;
    std::cout << "First predictions:" << std::endl;

    for(int i = 0; i < real_data.num_samples; ++i){
        int base = i * NUM_INPUTS;
        float output = forward_sample(network, &real_data.entries[base]);

        int prediction = output >= 0.5 ? 1 : 0;

        if(prediction == real_data.tags[i])
            aciertos++;

        if(i < DEBUG_SAMPLES){
            std::cout << "Student " << i << std::endl;
            std::cout << "  Assistance: " << real_data.entries[base] << std::endl;
            std::cout << "  Work hours: " << real_data.entries[base + 1] << std::endl;
            std::cout << "  Real tag: " << real_data.tags[i] << std::endl;
            std::cout << "  Output: " << output << std::endl;
            std::cout << "  Prediction: " << prediction << std::endl;
            std::cout << std::endl;
        }
    }

    double accuracy = 100.0 * aciertos / real_data.num_samples;

    std::cout << "Accuracy: " << accuracy << "%" << std::endl;

    release_dataset(data);
    release_dataset(real_data);

    return 0;
}