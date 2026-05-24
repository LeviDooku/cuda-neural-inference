//Pedro Velasco Santana
//File: neural_network.cpp

//Purpose: define the neural network

#include "neural_network.h"
#include <cmath>
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

float relu(float value){
    return value <= 0.0f ? 0.0f : value;
}

//Sigmoid function is: 1 / (1 + e^(-x))
//But if x is very negative, e^(-x) turns very big.
//That's why this implementation uses -20 and 20 for determine the output:
//sigmoid(-20) ~= 0.000000002
//sigmoid(20) ~= 0.999999998

float sigmoid(float value){
    float output = 0.0f;

    int negative_extreme = -20;
    int positive_extreme = negative_extreme * -1;

    if(value < negative_extreme)
        output = 0.0f;
    else if (value > positive_extreme)
        output = 1.0f;
    else
        output = 1.0f / (1.0f + std::exp(value * -1));

    return output;
}

float forward_sample(const neural_network& network, const float* sample){
    float hidden[NUM_HIDDEN];

    for(int i = 0; i < NUM_HIDDEN; ++i){
        float sum = network.hidden_biases[i];

        for(int j = 0; j < NUM_INPUTS; ++j)
            sum += sample[j] * network.input_hidden_weights[i * NUM_INPUTS + j];

        hidden[i] = relu(sum);
    }

    float output_sum = network.output_bias;
    
    for(int i = 0; i < NUM_HIDDEN; ++i)
        output_sum += hidden[i] * network.output_hidden_weights[i];

    return sigmoid(output_sum);
}