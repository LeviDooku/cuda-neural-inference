#Pedro Velasco Santana
#file: makefile

NVCC = nvcc
TARGET = test

SRC = src
INC = include
DATA = data
GRAPH = graph
SCRIPT = scripts

NVCCFLAGS = -I$(INC)

SOURCES = \
	test.cpp \
	$(SRC)/cpu_inference.cpp \
	$(SRC)/cuda_inference.cu \
	$(SRC)/dataset.cpp \
	$(SRC)/neural_network.cpp \
	$(SRC)/timer.cpp

.PHONY: all dirs run plots experiment clean clean-results clean-all

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(NVCC) $(NVCCFLAGS) $(SOURCES) -o $(TARGET)

dirs:
	mkdir -p $(DATA)
	mkdir -p $(GRAPH)

run: $(TARGET) dirs
	bash $(SCRIPT)/run_tests.sh

plots: dirs
	gnuplot $(SCRIPT)/plot_times_cpu.gp
	gnuplot $(SCRIPT)/plot_times_gpu.gp
	gnuplot $(SCRIPT)/plot_times_comparison.gp
	gnuplot $(SCRIPT)/plot_speedup.gp
	gnuplot $(SCRIPT)/plot_memory.gp

experiment: run plots

clean:
	rm -f $(TARGET)

clean-results:
	rm -rf $(DATA)/*
	rm -rf $(GRAPH)/*

clean-all: clean clean-results

