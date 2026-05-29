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

## How it works

The project is divided into several stages: dataset generation, data normalization, neural network initialization, CPU training, CPU inference, GPU inference and performance measurement.

### 1. Dataset generation

The program generates synthetic data representing students. Each student is described using two input values:

* Attendance hours.
* Hours dedicated to the final project.

Each sample also has a binary label:

* `1.0` if the student passes.
* `0.0` if the student fails.

The dataset is stored using two dynamic arrays:

* `entries`: stores the input values of all samples.
* `tags`: stores the expected output label for each sample.

For each student, the program generates random attendance hours and work hours. Then, it calculates a weighted score using both values. A small random noise value is added to make the dataset less deterministic and more realistic.

If the final score is greater than or equal to the defined threshold, the student is labeled as passed. Otherwise, the student is labeled as failed.

### 2. Data normalization

Before training or inference, the input values are normalized.

Attendance hours are divided by the maximum possible attendance value, and work hours are divided by the maximum possible number of work hours.

This transforms both input values into a similar range, making the neural network easier to train and improving numerical stability.

### 3. Neural network structure

The project uses a fixed simple neural network with a `2-8-1` architecture:

* 2 input values.
* 8 hidden neurons.
* 1 output value.

The two inputs represent the normalized attendance and work hours.

The hidden layer uses the ReLU activation function. The output layer uses the sigmoid activation function, which produces a value between `0` and `1`.

The final output is interpreted as the probability or tendency of the student passing the subject.

### 4. Network initialization

Before training, the network parameters are initialized.

The weights are initialized with small random values between `-0.5` and `0.5`. This helps avoid all neurons learning the same behavior.

The biases are initialized to `0`.

The network stores four groups of parameters:

* Weights from the input layer to the hidden layer.
* Biases of the hidden neurons.
* Weights from the hidden layer to the output neuron.
* Bias of the output neuron.

### 5. Forward pass

The forward pass is the process used to calculate the prediction for one sample.

First, each hidden neuron receives the two input values, multiplies them by their corresponding weights, adds its bias and applies the ReLU activation function.

Then, the output neuron combines the values produced by the hidden layer, applies its own weights and bias, and finally applies the sigmoid activation function.

The result is a value between `0` and `1`.

A value greater than or equal to `0.5` can be interpreted as a positive prediction, meaning that the student is predicted to pass.

### 6. CPU training

The network is trained on the CPU using a basic backpropagation implementation.

For each epoch, the program iterates over all training samples. For each sample, it performs a forward pass, compares the prediction with the expected label, calculates the error and updates the network weights and biases.

The learning rate controls how much the parameters are updated during each training step.

A higher learning rate can make training faster but less stable. A lower learning rate can make training more stable but slower.

### 7. CPU inference

Once the network has been trained, the CPU inference implementation is used as the sequential baseline.

The CPU version iterates over all inference samples. For each sample, it calculates the position of its input values in the `entries` array and calls the same `forward_sample` function used by the network.

The prediction is stored in the output array.

This version processes samples one by one, so its execution time grows linearly with the number of inference samples.

### 8. GPU inference with CUDA

The GPU version parallelizes the inference phase using CUDA.

The key idea is that each sample can be processed independently. The prediction for one student does not depend on the prediction for any other student.

Because of this, each CUDA thread is assigned to process one sample.

The neural network is copied to CUDA constant memory before launching the kernel. This is useful because the network is small, read-only during inference and shared by all threads.

The CUDA kernel calculates the global thread identifier and uses it to select the sample that the thread must process. If the thread identifier is within the valid range, the thread performs the forward pass and stores the result in the output array.

### 9. GPU memory management

The CUDA inference function performs the following steps:

1. Reserve memory on the GPU for the input samples.
2. Reserve memory on the GPU for the output predictions.
3. Copy the input data from CPU memory to GPU memory.
4. Copy the trained neural network to CUDA constant memory.
5. Configure the number of CUDA blocks and threads.
6. Launch the inference kernel.
7. Measure the kernel execution time using CUDA events.
8. Copy the output predictions from GPU memory back to CPU memory.
9. Free the allocated GPU memory.

The implementation uses `256` threads per block. The number of blocks is calculated so that all samples are covered, even when the number of samples is not exactly divisible by the block size.

### 10. Time measurement and testing

The project measures different execution times in order to compare CPU and GPU inference.

CPU time is measured using a custom timer based on `std::chrono`.

GPU kernel time is measured using CUDA events. This measures only the time spent executing the kernel on the GPU.

The total GPU time includes additional overhead, such as memory allocation, memory transfers and kernel launch.

The benchmark system compares:

* CPU inference time.
* Total GPU inference time.
* GPU kernel execution time.
* CPU accuracy.
* GPU accuracy.
* GPU memory usage.

These results are stored in a data file and later used to generate performance graphs.

## Instalation

```bash
git clone https://github.com/LeviDooku/cuda-neural-inference.git
cd cuda-neural-inference.git
make
```

The makefile includes several useful rules:

```bash
make              # Compile the project
make run          # Compile and run the benchmark script
make plots        # Generate performance graphs using gnuplot
make experiment   # Run benchmarks and generate graphs
make clean        # Remove the compiled executable
make clean-results # Remove generated data and graph files
make clean-all    # Remove executable, data files and graph files
```

## Examples

## Examples

The benchmark can be executed with:

```bash
make experiment
```

## Performance results

These results were obtained using the following hardware:

- CPU: Intel Core i7-9700F
- GPU: NVIDIA GeForce RTX 2080 Ti

The following table shows a summary of the benchmark results. Only some representative dataset sizes are included to keep the table readable.

| Samples | CPU time (ms) | GPU total time (ms) | GPU kernel time (ms) | Total GPU speedup | Kernel speedup | Accuracy | GPU memory (MB) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 0.009 | 75.379 | 0.023 | 0.0001x | 0.39x | 94.00% | 0.001 |
| 10,000 | 0.775 | 74.926 | 0.023 | 0.01x | 33.48x | 92.07% | 0.115 |
| 1,000,000 | 84.102 | 67.056 | 0.043 | 1.25x | 1942.51x | 91.83% | 11.444 |
| 10,000,000 | 812.631 | 87.003 | 0.233 | 9.34x | 3489.25x | 91.87% | 114.441 |
| 50,000,000 | 4084.100 | 188.194 | 1.096 | 21.70x | 3725.74x | 91.90% | 572.205 |
| 100,000,000 | 8432.720 | 276.044 | 2.174 | 30.55x | 3878.75x | 91.89% | 1144.410 |
| 200,000,000 | 16507.000 | 483.276 | 4.329 | 34.16x | 3813.13x | 91.90% | 2288.820 |

Speedup is calculated as:

```text
Speedup = CPU time / GPU time
```

## Useful links

[https://cplusplus.com/reference/random/](https://cplusplus.com/reference/random/)

[https://learn.microsoft.com/es-es/cpp/standard-library/chrono-functions?view=msvc-170](https://learn.microsoft.com/es-es/cpp/standard-library/chrono-functions?view=msvc-170)

[https://en.cppreference.com/cpp/chrono](https://en.cppreference.com/cpp/chrono)

[https://www.geeksforgeeks.org/machine-learning/ml-neural-network-implementation-in-c-from-scratch/](https://www.geeksforgeeks.org/machine-learning/ml-neural-network-implementation-in-c-from-scratch/)

[https://en.cppreference.com/cpp/numeric/math/exp](https://en.cppreference.com/cpp/numeric/math/exp)

[https://forums.developer.nvidia.com/t/constant-use/4117/2](https://forums.developer.nvidia.com/t/constant-use/4117/2)

[https://stackoverflow.com/questions/7961792/device-constant-const](https://stackoverflow.com/questions/7961792/device-constant-const)