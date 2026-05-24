//Pedro Velasco Santana
//File: dataset.cpp

//Purpose: generate synthetic data

#include "dataset.h"
#include <iostream>
#include <random>

const int MAX_ASSISTANCE = 60;
const int MAX_WORK_HOURS = 40;
const float THRESHOLD = 0.55;
const float ASSISTANCE_WEIGHT = 0.4;
const float WORK_HOURS_WEIGHT = 0.6;
const float NOISE_MIN = -0.1;
const float NOISE_MAX = 0.1;
const float PASS = 1.0f;
const float FAILED = 0.0f;

void create_dataset(dataset& dataset, int num_samples){
    dataset.num_samples = num_samples;
    dataset.num_entries = 2;

    dataset.entries = new float[dataset.num_samples * dataset.num_entries];
    dataset.tags = new float[dataset.num_samples];

    std::random_device rd;
    std::mt19937 gen(rd()); 

    std::uniform_int_distribution<int> assistance(0, MAX_ASSISTANCE);
    std::uniform_int_distribution<int> work_hour(0, MAX_WORK_HOURS);
    std::uniform_real_distribution<float> noise(NOISE_MIN, NOISE_MAX);

    for(int i = 0; i < num_samples; ++i){
        int base = i * dataset.num_entries;

        dataset.entries[base] = assistance(gen);
        dataset.entries[base+1] = work_hour(gen);

        float score_raw = ASSISTANCE_WEIGHT * (dataset.entries[base] / MAX_ASSISTANCE) + WORK_HOURS_WEIGHT * (dataset.entries[base+1] / MAX_WORK_HOURS);
        float score_final = score_raw + noise(gen);

        score_final >= THRESHOLD ? dataset.tags[i] = PASS : dataset.tags[i] = FAILED;
    }
}

void release_dataset(dataset& dataset){
    delete [] dataset.entries;
    delete [] dataset.tags;

    dataset.entries = nullptr;
    dataset.tags = nullptr;

    dataset.num_entries = 0;
    dataset.num_samples = 0;
}

void normalize_dataset(dataset& dataset){
    for(int i = 0; i < dataset.num_samples; ++i){
        int base = i * dataset.num_entries;

        dataset.entries[base] /= MAX_ASSISTANCE;
        dataset.entries[base+1] /= MAX_WORK_HOURS;
    }
}

void show_sample(const dataset& dataset, int index){
    if(index < 0 || index >= dataset.num_samples){
        std::cout << "Invalid index" << std::endl;
        std::cout << "Accepted: 0 to " << dataset.num_samples - 1 << std::endl;
    }
    else{
        std::cout << "Student: " << index << std::endl;
        std::cout << "Assistance: " << dataset.entries[index * dataset.num_entries] << std::endl;
        std::cout << "Hours worked: " << dataset.entries[index * dataset.num_entries + 1] << std::endl;
        std::cout << "Passed: " << dataset.tags[index];
    }
}