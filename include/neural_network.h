//Pedro Velasco Santana
//File: neural_network.h

//Purpose: define the neural network

//This is a fixed simple neural network

//2 inputs (assistance & hours worked) ; 8 hidden neurons ; 1 output

//Total: 33 adjustable parametres. Training the network means finding good values for those 33 parametres

#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

const int NUM_INPUTS = 2;
const int NUM_HIDDEN = 8;
const int NUM_OUTPUTS = 1;

struct neural_network{
    float input_hidden_weights[NUM_INPUTS * NUM_HIDDEN]; //Each neuron receives 2 inputs
    float hidden_biases[NUM_HIDDEN]; //Biases for each neuron

    float output_hidden_weights[NUM_HIDDEN * NUM_OUTPUTS]; //8*1 but useful if we want to change
    float output_bias; //Bias for output
};

void initialize_network(neural_network& network);

//Activation functions

//If value > 0 -> value ; if value <= 0 -> 0
//It causes some neurons to "switch off" when their value is not relevant.
//Useful in the hidden neurons

float relu(float value);

//Converts any number to a value between 0 and 1
//Useful in the output

float sigmoid(float value);

//Represents forward pass

float forward_sample(const neural_network& network, const float* sample);

void train_network_cpu(neural_network& network, const float* entries, const float* tags, int num_samples, int epoch, float learning_rate);

#endif