# CUDA Neural Network Inference

Compares CPU and GPU inference times for a simple neural network using CUDA. The goal is to study how GPU parallelism can improve the inference phase when processing large datasets.

## Overview

This project implements a simple neural network classifier and compares its inference performance on CPU and GPU.

The problem consists of predicting whether a student will pass a subject based on two input values: attendance hours and hours dedicated to a final project. The goal is not to build a realistic predictive model, but to use a small neural network as a case study for GPU parallelization.

The neural network follows a `2-8-1` architecture:

* 2 input neurons.
* 8 hidden neurons.
* 1 output neuron.

The two input values are normalized before being processed by the network. The hidden layer uses the ReLU activation function, while the output layer uses the sigmoid function to produce a value between `0` and `1`.

The training phase is performed on the CPU using a basic backpropagation implementation. Once the network has been trained, the inference phase is executed both sequentially on the CPU and in parallel on the GPU.

The inference stage is a good candidate for CUDA parallelization because each sample can be processed independently. The prediction for one student does not depend on the prediction for any other student, so each sample can be assigned to a different GPU thread.

This project also measures and compares:

* CPU inference time.
* Total GPU inference time, including memory transfers.
* GPU kernel execution time.
* CPU and GPU accuracy.
* GPU memory usage.

The results are used to analyze when GPU execution becomes worthwhile and how the cost of memory transfers affects the total execution time.

## Technologies used

* C++
* CUDA
* NVIDIA GPU programming
* Make
* Gnuplot
* Bash scripting




Fuentes / ayuda:

[https://cplusplus.com/reference/random/](https://cplusplus.com/reference/random/)

[https://learn.microsoft.com/es-es/cpp/standard-library/chrono-functions?view=msvc-170](https://learn.microsoft.com/es-es/cpp/standard-library/chrono-functions?view=msvc-170)

[https://en.cppreference.com/cpp/chrono](https://en.cppreference.com/cpp/chrono)

[https://www.geeksforgeeks.org/machine-learning/ml-neural-network-implementation-in-c-from-scratch/](https://www.geeksforgeeks.org/machine-learning/ml-neural-network-implementation-in-c-from-scratch/)

[https://en.cppreference.com/cpp/numeric/math/exp](https://en.cppreference.com/cpp/numeric/math/exp)

[https://forums.developer.nvidia.com/t/constant-use/4117/2](https://forums.developer.nvidia.com/t/constant-use/4117/2)

[https://stackoverflow.com/questions/7961792/device-constant-const](https://stackoverflow.com/questions/7961792/device-constant-const)