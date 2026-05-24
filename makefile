CXX = g++
NVCC = nvcc

CXXFLAGS = -O2 -std=c++17 -Iinclude
NVCCFLAGS = -O2 -std=c++17 -Iinclude

TEST_SRC = prueba.cpp \
           src/dataset.cpp \
           src/neural_network.cpp \
           src/timer.cpp

MAIN_SRC = src/main.cu \
           src/dataset.cpp \
           src/neural_network.cpp \
           src/timer.cpp \
           src/cpu_inference.cpp \
           src/cuda_inference.cu

TEST_TARGET = prueba
MAIN_TARGET = cuda_neural_inference

all: test

test:
	$(CXX) $(CXXFLAGS) $(TEST_SRC) -o $(TEST_TARGET)

run-test: test
	./$(TEST_TARGET)

main:
	$(NVCC) $(NVCCFLAGS) $(MAIN_SRC) -o $(MAIN_TARGET)

run-main: main
	./$(MAIN_TARGET)

clean:
	rm -f $(TEST_TARGET) $(MAIN_TARGET)