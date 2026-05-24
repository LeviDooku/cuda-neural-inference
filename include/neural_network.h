//Pedro Velasco Santana
//File: neural_network.h

//Purpose: define the neural network

//This is a fixed simple neural network

//2 inputs (assistance & hours worked) ; 8 hidden neurons ; 1 output

//Total: 33 adjustable parametres. Training the network means finding good values for those 33 parametres

#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <cmath> 

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

const int NUM_INPUTS = 2;
const int NUM_HIDDEN = 8;
const int NUM_OUTPUTS = 1;

struct neural_network{
    float input_hidden_weights[NUM_INPUTS * NUM_HIDDEN]; //Each neuron receives 2 inputs
    float hidden_biases[NUM_HIDDEN]; //Biases for each neuron

    float output_hidden_weights[NUM_HIDDEN * NUM_OUTPUTS]; //8*1 but useful if we want to change
    float output_bias; //Bias for output
};

//Assign the initial values of the parametres.
//Weights are selected randomly between -0.5 and 0.5
//Biases are initialized to 0

void initialize_network(neural_network& network);

//Activation functions

//If value > 0 -> value ; if value <= 0 -> 0
//It causes some neurons to "switch off" when their value is not relevant.
//Useful in the hidden neurons

HD inline float relu(float value){
    return value <= 0.0f ? 0.0f : value;
}

//Converts any number to a value between 0 and 1
//Useful in the output

//Sigmoid function is: 1 / (1 + e^(-x))
//But if x is very negative, e^(-x) turns very big.
//That's why this implementation uses -20 and 20 for determine the output:
//sigmoid(-20) ~= 0.000000002
//sigmoid(20) ~= 0.999999998

HD inline float sigmoid(float value){
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

//Represents forward pass: doing the prediction for one student (sample)

float forward_sample(const neural_network& network, const float* sample);

void train_network_cpu(neural_network& network, const float* entries, const float* tags, int num_samples, int epoch, float learning_rate);

#endif