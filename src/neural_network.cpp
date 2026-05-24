//Pedro Velasco Santana
//File: neural_network.cpp

//Purpose: define the neural network

#include "neural_network.h"
#include <cmath>

void initialize_network(neural_network& network){
    
}

float relu(float value){
    return value <= 0.0f ? 0.0f : 1.0f;
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

    if(value < -20)
        output = 0.0f;
    else if (value > 20)
        output = 1.0f;
    else
        output = 1.0f / (1.0f + std::exp(value * -1));
}
