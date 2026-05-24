//Pedro Velasco Santana
//File: neural_network.cpp

//Purpose: define the neural network

#include "neural_network.h"
#include <random>

const float MIN_WEIGHT = -0.5f;
const float MAX_WEIGHT = MIN_WEIGHT * -1;

void initialize_network(neural_network& network){
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<float> weights(MIN_WEIGHT, MAX_WEIGHT);

    for(int i = 0; i < NUM_INPUTS * NUM_HIDDEN; ++i)
        network.input_hidden_weights[i] = weights(gen);

    for(int i = 0; i < NUM_HIDDEN; ++i)
        network.hidden_biases[i] = 0.0f;

    for(int i = 0; i < NUM_HIDDEN * NUM_OUTPUTS; i++)
        network.output_hidden_weights[i] = weights(gen);
    
    network.output_bias = 0.0f;
}

void train_network_cpu(neural_network& network, const float* entries, const float* tags, int num_samples, int epoch, float learning_rate){
    for(int i = 0; i < epoch; ++i){
        for(int j = 0; j < num_samples; ++j){
            float hidden[NUM_HIDDEN];
            int base = j * NUM_INPUTS;

            for(int h = 0; h < NUM_HIDDEN; ++h){
                float sum = network.hidden_biases[h];
        
                for(int k = 0; k < NUM_INPUTS; ++k)
                    sum += entries[base + k] * network.input_hidden_weights[h * NUM_INPUTS + k];
        
                hidden[h] = relu(sum);
            }

            float output_sum = network.output_bias;

            for(int h = 0; h < NUM_HIDDEN; ++h)
                output_sum += hidden[h] * network.output_hidden_weights[h];

            float prediction = sigmoid(output_sum);
            float error = prediction - tags[j];
            float delta_output = error * (prediction * (1 - prediction));

            float delta_hidden[NUM_HIDDEN];

            for(int h = 0; h < NUM_HIDDEN; ++h){
                float relu_derivate = hidden[h] > 0 ? 1.0f : 0.0f;
                delta_hidden[h] = delta_output * network.output_hidden_weights[h] * relu_derivate;
            }

            for(int h = 0; h < NUM_HIDDEN; ++h)
                network.output_hidden_weights[h] -= learning_rate * delta_output * hidden[h];

            network.output_bias -= learning_rate * delta_output;

            for(int h = 0; h < NUM_HIDDEN; ++h){
                for(int k = 0; k < NUM_INPUTS; ++k)
                    network.input_hidden_weights[h * NUM_INPUTS + k] -= learning_rate * delta_hidden[h] * entries[base + k];
                network.hidden_biases[h] -= learning_rate * delta_hidden[h];
            }
        }
    }
}
